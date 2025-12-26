"""
Video management endpoints
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query, Form
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import json
import uuid
from datetime import datetime

router = APIRouter()

VIDEOS_DIR = Path("/app/data/videos")
RESULTS_DIR = Path("/app/data/results")
TRAINING_DIR = Path("/app/data/training")


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
    
    return {
        "video_id": video_id,
        "filename": file_path.name,
        "file_path": str(file_path),
        "file_size": file_size,
        "has_analysis": has_analysis,
        "status": "analyzed" if has_analysis else "uploaded"
    }


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
                "label": label,
                "has_label": label is not None
            })
    
    return {
        "videos": videos[skip:skip+limit],
        "total": len(videos),
        "skip": skip,
        "limit": limit
    }

