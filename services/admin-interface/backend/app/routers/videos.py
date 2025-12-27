"""
Video management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import json
import uuid
from datetime import datetime
import cv2
import io
import httpx
import asyncio
import os
import nats

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")
PROCESSED_DIR = Path("/app/data/processed")
ANNOTATED_DIR = PROCESSED_DIR / "annotated"
RESULTS_DIR = Path("/app/data/results")
TRAINING_DIR = Path("/app/data/training")

# Annotation renderer service URL
ANNOTATION_RENDERER_URL = "http://annotation-renderer:8000"

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


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    label: Optional[int] = Form(None)
):
    """Upload a video file with optional label (0=sound, 1=lame)"""
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
    file_path = VIDEOS_DIR / filename
    
    # Ensure directories exist
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_size = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
                file_size += len(chunk)
        
        # Save label if provided
        label_saved = False
        if label is not None and label in [0, 1]:
            labels_dir = TRAINING_DIR / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            label_file = labels_dir / f"{video_id}_label.json"
            label_data = {
                "video_id": video_id,
                "label": label,
                "confidence": "certain",
                "timestamp": datetime.utcnow().isoformat(),
                "labeled_at_upload": True
            }
            with open(label_file, "w") as f:
                json.dump(label_data, f)
            label_saved = True
        
        return {
            "video_id": video_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": file_size,
            "uploaded_at": datetime.utcnow().isoformat(),
            "label": label if label_saved else None,
            "label_saved": label_saved
        }
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/{video_id}")
async def get_video(video_id: str):
    """Get video information"""
    # Find video file
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    file_path = video_files[0]
    file_size = file_path.stat().st_size
    
    # Check for analysis results
    fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
    has_analysis = fusion_file.exists()
    
    # Check for annotated video
    annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
    has_annotated = annotated_file.exists()
    
    # Get label if exists
    label = None
    labels_dir = TRAINING_DIR / "labels"
    label_file = labels_dir / f"{video_id}_label.json"
    if label_file.exists():
        try:
            with open(label_file) as f:
                label_data = json.load(f)
                label = label_data.get("label")
        except:
            pass
    
    # Get video metadata
    cap = cv2.VideoCapture(str(file_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()
    
    return {
        "video_id": video_id,
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_size": file_size,
        "has_analysis": has_analysis,
        "has_annotated": has_annotated,
        "label": label,
        "status": "analyzed" if has_analysis else "uploaded",
        "metadata": {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
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
    show_labels: bool = True
):
    """Trigger annotation rendering for a video.
    
    If pose data doesn't exist and include_pose=True, triggers T-LEAP pipeline first.
    """
    # Check video exists
    video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
    if not video_files:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video_path = str(video_files[0])
    
    # Check if T-LEAP pose data exists
    tleap_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
    
    if include_pose and not tleap_file.exists():
        # Trigger T-LEAP pipeline and wait for it to complete
        print(f"Pose data not found for {video_id}, triggering T-LEAP pipeline...")
        
        try:
            await trigger_tleap_pipeline(video_id, video_path)
            
            # Wait for T-LEAP to complete (poll for up to 120 seconds)
            max_wait = 120
            poll_interval = 2
            waited = 0
            
            while waited < max_wait:
                await asyncio.sleep(poll_interval)
                waited += poll_interval
                
                if tleap_file.exists():
                    print(f"T-LEAP completed for {video_id} after {waited}s")
                    break
                    
                if waited % 10 == 0:
                    print(f"Waiting for T-LEAP... {waited}s/{max_wait}s")
            
            if not tleap_file.exists():
                print(f"T-LEAP timeout for {video_id}, proceeding without pose data")
                
        except Exception as e:
            print(f"Failed to trigger T-LEAP: {e}")
            # Continue without pose data
    
    # Call annotation renderer service
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{ANNOTATION_RENDERER_URL}/render",
                json={
                    "video_id": video_id,
                    "include_yolo": include_yolo,
                    "include_pose": include_pose,
                    "show_confidence": show_confidence,
                    "show_labels": show_labels
                },
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
    limit: int = Query(100, ge=1, le=1000)
):
    """List all videos"""
    videos = []
    labels_dir = TRAINING_DIR / "labels"
    
    for video_file in list(VIDEOS_DIR.glob("*.*"))[:skip+limit]:
        if video_file.is_file():
            video_id = video_file.stem.split("_")[0]  # Extract ID from filename
            fusion_file = RESULTS_DIR / "fusion" / f"{video_id}_fusion.json"
            label_file = labels_dir / f"{video_id}_label.json"
            annotated_file = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
            
            # Get label if exists
            label = None
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        label_data = json.load(f)
                        label = label_data.get("label")
                except:
                    pass
            
            videos.append({
                "video_id": video_id,
                "filename": video_file.name,
                "file_size": video_file.stat().st_size,
                "has_analysis": fusion_file.exists(),
                "has_annotated": annotated_file.exists(),
                "label": label,
                "has_label": label is not None
            })
    
    return {
        "videos": videos[skip:skip+limit],
        "total": len(videos),
        "skip": skip,
        "limit": limit
    }
