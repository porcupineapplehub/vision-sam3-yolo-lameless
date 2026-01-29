"""
Video management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form, Response, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta
import cv2
import io
import httpx
import asyncio
import os
import nats
import boto3
from botocore.exceptions import ClientError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ..database import get_db, Video

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")
PROCESSED_DIR = Path("/app/data/processed")
ANNOTATED_DIR = PROCESSED_DIR / "annotated"
RESULTS_DIR = Path("/app/data/results")
TRAINING_DIR = Path("/app/data/training")

# Storage backend configuration
# STORAGE_BACKEND: "local" (default for development) or "s3" (for AWS deployment)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").lower()
S3_VIDEOS_BUCKET = os.getenv("S3_VIDEOS_BUCKET", "")
CLOUDFRONT_DOMAIN = os.getenv("CLOUDFRONT_DOMAIN", "")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

# Initialize S3 client (lazy loaded)
_s3_client = None

def get_s3_client():
    """Get S3 client (lazy initialization)"""
    global _s3_client
    if _s3_client is None and S3_VIDEOS_BUCKET:
        _s3_client = boto3.client('s3', region_name=AWS_REGION)
    return _s3_client

def is_s3_enabled():
    """Check if S3 storage is enabled.

    S3 is enabled when:
    1. STORAGE_BACKEND is set to "s3", AND
    2. S3_VIDEOS_BUCKET is configured

    This allows local development to use local storage even if S3 bucket is configured.
    """
    return STORAGE_BACKEND == "s3" and bool(S3_VIDEOS_BUCKET)

def get_storage_backend():
    """Get the current storage backend type."""
    if is_s3_enabled():
        return "s3"
    return "local"

# Annotation renderer service URL (uses service discovery namespace)
SERVICE_NAMESPACE = os.getenv("SERVICE_NAMESPACE", "cow-lameness-production.local")
ANNOTATION_RENDERER_URL = f"http://annotation-renderer.{SERVICE_NAMESPACE}:8000"

# NATS connection for triggering pipelines
_nats_client = None

async def get_nats():
    """Get NATS connection"""
    global _nats_client
    if _nats_client is None or not _nats_client.is_connected:
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        _nats_client = await nats.connect(nats_url)
    return _nats_client


async def trigger_tleap_pipeline(video_id: str, video_path: str):
    """Trigger T-LEAP pipeline for a video"""
    nc = await get_nats()
    msg = {
        'video_id': video_id,
        'processed_path': video_path
    }
    await nc.publish('video.preprocessed', json.dumps(msg).encode())
    await nc.flush()
    print(f"Triggered T-LEAP pipeline for {video_id}")


class VideoInfo(BaseModel):
    video_id: str
    filename: str
    file_path: str
    file_size: int
    uploaded_at: str
    status: str


@router.get("/storage-config")
async def get_storage_config():
    """Get current storage configuration.

    Returns the active storage backend and its configuration.
    - backend: "local" or "s3"
    - When backend is "local": videos stored in /app/data/videos (EFS in AWS, local volume in dev)
    - When backend is "s3": videos stored in S3 bucket with optional CloudFront CDN
    """
    backend = get_storage_backend()
    return {
        "backend": backend,
        "s3_enabled": is_s3_enabled(),
        "s3_bucket": S3_VIDEOS_BUCKET if is_s3_enabled() else None,
        "cloudfront_enabled": bool(CLOUDFRONT_DOMAIN) and is_s3_enabled(),
        "cloudfront_domain": CLOUDFRONT_DOMAIN if CLOUDFRONT_DOMAIN and is_s3_enabled() else None,
        "local_path": str(VIDEOS_DIR) if backend == "local" else None
    }


@router.post("/upload-url")
async def get_upload_url(
    filename: str = Query(..., description="Original filename"),
    content_type: str = Query("video/mp4", description="Content type")
):
    """Get a pre-signed URL for direct S3 upload from browser.

    This allows large video uploads directly to S3, bypassing the backend.
    """
    if not is_s3_enabled():
        raise HTTPException(
            status_code=400,
            detail="S3 storage not configured. Use /upload endpoint instead."
        )

    s3 = get_s3_client()
    video_id = str(uuid.uuid4())
    file_ext = Path(filename).suffix.lower() or ".mp4"
    s3_key = f"raw/{video_id}{file_ext}"

    try:
        # Generate pre-signed POST URL (better for browser uploads)
        presigned_post = s3.generate_presigned_post(
            Bucket=S3_VIDEOS_BUCKET,
            Key=s3_key,
            Fields={
                "Content-Type": content_type,
            },
            Conditions=[
                {"Content-Type": content_type},
                ["content-length-range", 1, 5 * 1024 * 1024 * 1024],  # Max 5GB
            ],
            ExpiresIn=3600  # 1 hour
        )

        return {
            "video_id": video_id,
            "upload_url": presigned_post["url"],
            "upload_fields": presigned_post["fields"],
            "s3_key": s3_key,
            "expires_in": 3600
        }
    except ClientError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate upload URL: {str(e)}")


@router.post("/confirm-upload")
async def confirm_upload(
    video_id: str = Query(...),
    s3_key: str = Query(...),
    label: Optional[int] = Query(None),
    original_filename: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Confirm that a video upload to S3 is complete.

    Call this after successfully uploading to S3 using the pre-signed URL.
    Creates a database record for the video.
    """
    if not is_s3_enabled():
        raise HTTPException(status_code=400, detail="S3 storage not configured")

    s3 = get_s3_client()

    # Verify the file exists in S3
    try:
        head = s3.head_object(Bucket=S3_VIDEOS_BUCKET, Key=s3_key)
        file_size = head["ContentLength"]
    except ClientError:
        raise HTTPException(status_code=404, detail="Video not found in S3")

    # Extract filename from s3_key
    filename = s3_key.split('/')[-1]
    uploaded_at = datetime.utcnow()

    # Create database record
    video_record = Video(
        id=video_id,
        filename=filename,
        original_filename=original_filename,
        file_size=file_size,
        storage_backend="s3",
        s3_key=s3_key,
        label=label if label is not None and label in [0, 1] else None,
        label_confidence="certain" if label is not None and label in [0, 1] else None,
        status="uploaded",
        uploaded_at=uploaded_at
    )
    db.add(video_record)
    await db.commit()

    return {
        "video_id": video_id,
        "s3_key": s3_key,
        "file_size": file_size,
        "uploaded_at": uploaded_at.isoformat(),
        "label": label if label is not None and label in [0, 1] else None,
        "label_saved": label is not None and label in [0, 1],
        "stream_url": get_video_stream_url(video_id, s3_key)
    }


def get_video_stream_url(video_id: str, s3_key: str = None) -> str:
    """Get the streaming URL for a video.

    If CloudFront is configured, returns CloudFront URL.
    Otherwise returns S3 pre-signed URL.
    """
    if not s3_key:
        s3_key = f"raw/{video_id}.mp4"

    if CLOUDFRONT_DOMAIN:
        return f"https://{CLOUDFRONT_DOMAIN}/{s3_key}"
    elif is_s3_enabled():
        s3 = get_s3_client()
        try:
            return s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": S3_VIDEOS_BUCKET, "Key": s3_key},
                ExpiresIn=3600  # 1 hour
            )
        except ClientError:
            return None
    return None


@router.get("/{video_id}/stream-url")
async def get_stream_url(video_id: str):
    """Get the streaming URL for a video.

    Returns CloudFront URL if configured, or falls back to S3 pre-signed URL.
    """
    if not is_s3_enabled():
        # Fall back to local streaming endpoint
        return {
            "video_id": video_id,
            "stream_url": f"/api/videos/{video_id}/stream",
            "source": "local"
        }

    s3 = get_s3_client()

    # Find the video in S3
    s3_key = None
    for prefix in ["raw/", "processed/", ""]:
        for ext in [".mp4", ".avi", ".mov", ".mkv"]:
            try_key = f"{prefix}{video_id}{ext}"
            try:
                s3.head_object(Bucket=S3_VIDEOS_BUCKET, Key=try_key)
                s3_key = try_key
                break
            except ClientError:
                continue
        if s3_key:
            break

    if not s3_key:
        raise HTTPException(status_code=404, detail="Video not found in S3")

    stream_url = get_video_stream_url(video_id, s3_key)

    return {
        "video_id": video_id,
        "stream_url": stream_url,
        "s3_key": s3_key,
        "source": "cloudfront" if CLOUDFRONT_DOMAIN else "s3"
    }


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    label: Optional[int] = Form(None),
    db: AsyncSession = Depends(get_db)
):
    """Upload a video file with optional label (0=sound, 1=lame).

    If S3 is configured, uploads directly to S3.
    Otherwise, stores locally on EFS.
    Video metadata is saved to the database.
    """
    # Validate file type
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    # Generate video ID
    video_id = str(uuid.uuid4())
    filename = f"{video_id}{file_ext}"
    uploaded_at = datetime.utcnow()

    # Upload to S3 if configured
    if is_s3_enabled():
        s3 = get_s3_client()
        s3_key = f"raw/{filename}"

        try:
            # Read file content
            file_content = await file.read()
            file_size = len(file_content)

            # Upload to S3
            s3.put_object(
                Bucket=S3_VIDEOS_BUCKET,
                Key=s3_key,
                Body=file_content,
                ContentType=file.content_type or "video/mp4"
            )

            # Create database record
            video_record = Video(
                id=video_id,
                filename=filename,
                original_filename=file.filename,
                file_size=file_size,
                storage_backend="s3",
                s3_key=s3_key,
                label=label if label is not None and label in [0, 1] else None,
                label_confidence="certain" if label is not None and label in [0, 1] else None,
                status="uploaded",
                uploaded_at=uploaded_at
            )
            db.add(video_record)
            await db.commit()

            return {
                "video_id": video_id,
                "filename": file.filename,
                "s3_key": s3_key,
                "file_size": file_size,
                "uploaded_at": uploaded_at.isoformat(),
                "label": label if label is not None and label in [0, 1] else None,
                "label_saved": label is not None and label in [0, 1],
                "storage": "s3",
                "stream_url": get_video_stream_url(video_id, s3_key)
            }
        except ClientError as e:
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {str(e)}")

    # Fall back to local storage (EFS)
    file_path = VIDEOS_DIR / filename
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    file_size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                file_size += len(chunk)

        # Create database record
        video_record = Video(
            id=video_id,
            filename=filename,
            original_filename=file.filename,
            file_size=file_size,
            storage_backend="local",
            file_path=str(file_path),
            label=label if label is not None and label in [0, 1] else None,
            label_confidence="certain" if label is not None and label in [0, 1] else None,
            status="uploaded",
            uploaded_at=uploaded_at
        )
        db.add(video_record)
        await db.commit()

        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": uploaded_at.isoformat(),
            "label": label if label is not None and label in [0, 1] else None,
            "label_saved": label is not None and label in [0, 1],
            "storage": "local"
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{video_id}")
async def get_video(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get video information from database"""
    # Query video from database
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Check for analysis results (local files for now)
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    has_analysis = video.has_analysis or fusion_file.exists()

    # Check for annotated video
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    has_annotated = video.has_annotated or annotated_file.exists()

    # Build stream URL
    stream_url = None
    if video.storage_backend == "s3" and video.s3_key:
        stream_url = get_video_stream_url(video_id, video.s3_key)
    elif video.file_path:
        stream_url = f"/api/videos/{video_id}/stream"

    return {
        "video_id": video_id,
        "filename": video.filename,
        "original_filename": video.original_filename,
        "file_size": video.file_size,
        "storage": video.storage_backend,
        "s3_key": video.s3_key,
        "file_path": video.file_path,
        "stream_url": stream_url,
        "has_analysis": has_analysis,
        "has_annotated": has_annotated,
        "label": video.label,
        "label_confidence": video.label_confidence,
        "status": video.status,
        "uploaded_at": video.uploaded_at.isoformat() if video.uploaded_at else None,
        "processed_at": video.processed_at.isoformat() if video.processed_at else None,
        "metadata": {
            "fps": video.fps,
            "frame_count": video.frame_count,
            "width": video.width,
            "height": video.height,
            "duration": video.duration
        }
    }


@router.get("/{video_id}/stream")
async def stream_video(video_id: str):
    """Stream the original video file"""
    # Find video file
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video_files[0]
    
    # Determine media type
    suffix = file_path.suffix.lower()
    media_types = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska"
    }
    media_type = media_types.get(suffix, "video/mp4")
    
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name
    )


@router.get("/{video_id}/annotated")
async def stream_annotated_video(video_id: str):
    """Stream the annotated video file"""
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    
    if not annotated_file.exists():
        raise HTTPException(
            status_code=404, 
            detail="Annotated video not found. Trigger annotation first."
        )
    
    return FileResponse(
        path=str(annotated_file),
        media_type="video/mp4",
        filename=f"{video_id}_annotated.mp4"
    )


@router.get("/{video_id}/frame/{frame_num}")
async def get_frame(video_id: str, frame_num: int, annotated: bool = False):
    """Get a specific frame from the video as an image"""
    # Choose source file
    if annotated:
        video_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if not video_path.exists():
            raise HTTPException(status_code=404, detail="Annotated video not found")
    else:
        video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
        if not video_files:
            raise HTTPException(status_code=404, detail="Video not found")
        video_path = video_files[0]
    
    # Extract frame
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_num < 0 or frame_num >= total_frames:
        cap.release()
        raise HTTPException(status_code=400, detail=f"Frame number must be 0-{total_frames-1}")
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to read frame")
    
    # Encode frame as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg"
    )


@router.post("/{video_id}/annotate")
async def trigger_annotation(
    video_id: str,
    include_yolo: bool = True,
    include_pose: bool = True,
    show_confidence: bool = True,
    show_labels: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """Trigger annotation rendering for a video.

    If pose data doesn't exist and include_pose=True, triggers T-LEAP pipeline first.
    """
    # Check video exists in database
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Determine video path or S3 key
    video_path = None
    s3_key = None

    if video.storage_backend == "s3" and video.s3_key:
        s3_key = video.s3_key
        # For S3 videos, we pass the S3 key to the annotation service
        video_path = f"s3://{S3_VIDEOS_BUCKET}/{s3_key}"
    elif video.file_path:
        video_path = video.file_path
    else:
        # Fallback to legacy path
        video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
        if video_files:
            video_path = str(video_files[0])
        else:
            raise HTTPException(status_code=404, detail="Video file not found")
    
    # Check if T-LEAP pose data exists (don't wait for it - proceed with YOLO only if not available)
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    pose_available = tleap_file.exists()

    if include_pose and not pose_available:
        print(f"Pose data not found for {video_id}, proceeding with YOLO annotations only")
        # Override include_pose to false since data doesn't exist
        include_pose = False

    # Call annotation renderer service
    try:
        request_data = {
            "video_id": video_id,
            "include_yolo": include_yolo,
            "include_pose": include_pose,
            "show_confidence": show_confidence,
            "show_labels": show_labels,
            "video_path": video_path,
        }
        # Add S3 info if video is stored in S3
        if s3_key:
            request_data["s3_bucket"] = S3_VIDEOS_BUCKET
            request_data["s3_key"] = s3_key

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ANNOTATION_RENDERER_URL}/render",
                json=request_data,
                timeout=30.0
            )
            return response.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Annotation renderer service unavailable"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{video_id}/annotation-status")
async def get_annotation_status(video_id: str):
    """Get annotation rendering status"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{ANNOTATION_RENDERER_URL}/status/{video_id}",
                timeout=10.0
            )
            return response.json()
    except httpx.ConnectError:
        # Service unavailable, check if file exists
        annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if annotated_file.exists():
            return {
                "video_id": video_id,
                "status": "completed",
                "progress": 1.0,
                "output_path": str(annotated_file)
            }
        return {
            "video_id": video_id,
            "status": "not_found"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{video_id}/annotation")
async def delete_annotation(video_id: str):
    """Delete annotation and pose data for a video"""
    deleted_files = []
    errors = []
    
    # Delete annotated video
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    if annotated_file.exists():
        try:
            annotated_file.unlink()
            deleted_files.append(str(annotated_file))
        except Exception as e:
            errors.append(f"Failed to delete annotated video: {e}")
    
    # Delete T-LEAP pose data
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    if tleap_file.exists():
        try:
            tleap_file.unlink()
            deleted_files.append(str(tleap_file))
        except Exception as e:
            errors.append(f"Failed to delete pose data: {e}")
    
    # Delete YOLO detections
    yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
    if yolo_file.exists():
        try:
            yolo_file.unlink()
            deleted_files.append(str(yolo_file))
        except Exception as e:
            errors.append(f"Failed to delete YOLO data: {e}")
    
    # Clear render status in annotation-renderer
    try:
        async with httpx.AsyncClient() as client:
            await client.delete(
                f"{ANNOTATION_RENDERER_URL}/status/{video_id}",
                timeout=5.0
            )
    except:
        pass  # Ignore if service is unavailable
    
    return {
        "video_id": video_id,
        "deleted_files": deleted_files,
        "errors": errors,
        "success": len(errors) == 0
    }


@router.get("/{video_id}/detections")
async def get_video_detections(video_id: str):
    """Get YOLO detections for a video"""
    yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
    
    if not yolo_file.exists():
        raise HTTPException(status_code=404, detail="Detections not found")
    
    with open(yolo_file) as f:
        return json.load(f)


@router.get("/{video_id}/pose")
async def get_video_pose(video_id: str):
    """Get T-LEAP pose data for a video"""
    pose_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    
    if not pose_file.exists():
        raise HTTPException(status_code=404, detail="Pose data not found")
    
    with open(pose_file) as f:
        return json.load(f)


@router.get("")
async def list_videos(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    label_filter: Optional[int] = Query(None, alias="label", description="Filter by label (0=sound, 1=lame)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db)
):
    """List all videos from database with optional filtering"""
    # Build query
    query = select(Video)

    # Apply filters
    if label_filter is not None:
        query = query.where(Video.label == label_filter)
    if status:
        query = query.where(Video.status == status)

    # Get total count (with filters applied)
    count_query = select(func.count(Video.id))
    if label_filter is not None:
        count_query = count_query.where(Video.label == label_filter)
    if status:
        count_query = count_query.where(Video.status == status)
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results, ordered by upload time (newest first)
    query = query.offset(skip).limit(limit).order_by(Video.uploaded_at.desc())
    result = await db.execute(query)
    videos = result.scalars().all()

    return {
        "videos": [
            {
                "video_id": v.id,
                "filename": v.filename,
                "original_filename": v.original_filename,
                "file_size": v.file_size,
                "storage": v.storage_backend,
                "s3_key": v.s3_key,
                "file_path": v.file_path,
                "label": v.label,
                "has_label": v.label is not None,
                "has_analysis": v.has_analysis,
                "has_annotated": v.has_annotated,
                "status": v.status,
                "uploaded_at": v.uploaded_at.isoformat() if v.uploaded_at else None
            }
            for v in videos
        ],
        "total": total,
        "skip": skip,
        "limit": limit
    }


@router.post("/migrate-to-db")
async def migrate_videos_to_db(db: AsyncSession = Depends(get_db)):
    """One-time migration of existing videos to database.

    Scans S3 bucket (if enabled) or local storage and creates database records
    for any videos not already in the database.
    """
    migrated = 0
    skipped = 0
    errors = []

    if is_s3_enabled():
        # Scan S3 bucket
        s3 = get_s3_client()
        try:
            paginator = s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_VIDEOS_BUCKET, Prefix='raw/'):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    filename = key.split('/')[-1]
                    if not filename:
                        continue
                    video_id = Path(filename).stem

                    # Check if already in DB
                    existing = await db.execute(
                        select(Video).where(Video.id == video_id)
                    )
                    if existing.scalar_one_or_none():
                        skipped += 1
                        continue

                    # Create record
                    try:
                        # Convert timezone-aware datetime to naive UTC
                        last_modified = obj.get('LastModified')
                        if last_modified and last_modified.tzinfo is not None:
                            last_modified = last_modified.replace(tzinfo=None)

                        video = Video(
                            id=video_id,
                            filename=filename,
                            file_size=obj['Size'],
                            storage_backend="s3",
                            s3_key=key,
                            status="uploaded",
                            uploaded_at=last_modified or datetime.utcnow()
                        )
                        db.add(video)
                        migrated += 1
                    except Exception as e:
                        errors.append(f"Failed to migrate {video_id}: {str(e)}")

            await db.commit()
        except ClientError as e:
            errors.append(f"S3 error: {str(e)}")
    else:
        # Scan local storage
        if VIDEOS_DIR.exists():
            for video_file in VIDEOS_DIR.glob("*.*"):
                if not video_file.is_file():
                    continue

                video_id = video_file.stem

                # Check if already in DB
                existing = await db.execute(
                    select(Video).where(Video.id == video_id)
                )
                if existing.scalar_one_or_none():
                    skipped += 1
                    continue

                # Create record
                try:
                    video = Video(
                        id=video_id,
                        filename=video_file.name,
                        file_size=video_file.stat().st_size,
                        storage_backend="local",
                        file_path=str(video_file),
                        status="uploaded",
                        uploaded_at=datetime.utcnow()
                    )
                    db.add(video)
                    migrated += 1
                except Exception as e:
                    errors.append(f"Failed to migrate {video_id}: {str(e)}")

            await db.commit()

    return {
        "migrated": migrated,
        "skipped": skipped,
        "errors": errors,
        "storage_backend": get_storage_backend()
    }
