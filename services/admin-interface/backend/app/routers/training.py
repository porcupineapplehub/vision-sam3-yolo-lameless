"""
Training endpoints
"""
import os
import json
import nats
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from datetime import datetime

router = APIRouter()

TRAINING_DIR = Path("/app/data/training")
RESULTS_DIR = Path("/app/data/results")

# NATS connection
nats_client = None


async def get_nats():
    """Get NATS connection"""
    global nats_client
    if nats_client is None or not nats_client.is_connected:
        nats_url = os.getenv("NATS_URL", "nats://nats:4222")
        nats_client = await nats.connect(nats_url)
    return nats_client


class LabelRequest(BaseModel):
    label: int  # 0 = sound, 1 = lame
    confidence: Optional[str] = "certain"  # certain, uncertain


@router.post("/videos/{video_id}/label")
async def label_video(video_id: str, label_request: LabelRequest):
    """Submit label for a video"""
    # Store label
    labels_dir = TRAINING_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    label_file = labels_dir / f"{video_id}_label.json"
    
    label_data = {
        "video_id": video_id,
        "label": label_request.label,
        "confidence": label_request.confidence,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    with open(label_file, "w") as f:
        json.dump(label_data, f)
    
    # Publish new label event to trigger auto-training check
    try:
        nc = await get_nats()
        await nc.publish(
            "training.data.added",
            json.dumps({
                "video_id": video_id,
                "label": label_request.label,
                "timestamp": datetime.utcnow().isoformat()
            }).encode()
        )
    except Exception as e:
        print(f"Failed to publish label event: {e}")
    
    return {
        "video_id": video_id,
        "label": label_request.label,
        "status": "saved"
    }


@router.get("/queue")
async def get_training_queue():
    """Get videos that need labeling (active learning queue)"""
    videos = []
    
    fusion_dir = RESULTS_DIR / "fusion"
    labels_dir = TRAINING_DIR / "labels"
    
    if fusion_dir.exists():
        for fusion_file in fusion_dir.glob("*_fusion.json"):
            video_id = fusion_file.stem.replace("_fusion", "")
            label_file = labels_dir / f"{video_id}_label.json"
            
            if not label_file.exists():
                with open(fusion_file) as f:
                    fusion_data = json.load(f)
                    fusion_result = fusion_data.get("fusion_result", {})
                    
                    # Prioritize uncertain predictions
                    prob = fusion_result.get("final_probability", 0.5)
                    uncertainty = abs(0.5 - prob)  # Lower uncertainty = more uncertain
                    
                    videos.append({
                        "video_id": video_id,
                        "predicted_probability": prob,
                        "uncertainty": uncertainty
                    })
    
    # Sort by uncertainty (most uncertain first)
    videos.sort(key=lambda x: x["uncertainty"])
    
    return {
        "videos": videos[:50],  # Top 50 most uncertain
        "total": len(videos)
    }


@router.get("/stats")
async def get_training_stats():
    """Get training dataset statistics"""
    labels_dir = TRAINING_DIR / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    total_labels = 0
    sound_count = 0
    lame_count = 0
    
    for label_file in labels_dir.glob("*_label.json"):
        try:
            with open(label_file) as f:
                label_data = json.load(f)
                total_labels += 1
                if label_data.get("label") == 0:
                    sound_count += 1
                elif label_data.get("label") == 1:
                    lame_count += 1
        except:
            pass
    
    return {
        "total_labels": total_labels,
        "sound_count": sound_count,
        "lame_count": lame_count,
        "balance_ratio": sound_count / lame_count if lame_count > 0 else None,
        "ready_for_training": total_labels >= 10 and sound_count > 0 and lame_count > 0
    }


@router.get("/status")
async def get_training_status():
    """Get training job status"""
    status_file = TRAINING_DIR / "training_status.json"
    
    if status_file.exists():
        try:
            with open(status_file) as f:
                return json.load(f)
        except:
            pass
    
    return {
        "status": "idle",
        "last_trained": None,
        "samples_used": 0,
        "metrics": {},
        "models": []
    }


@router.post("/ml/start")
async def start_ml_training():
    """Trigger ML training manually"""
    try:
        nc = await get_nats()
        await nc.publish(
            "training.ml.requested",
            json.dumps({
                "requested_at": datetime.utcnow().isoformat(),
                "manual": True
            }).encode()
        )
        
        return {
            "status": "training_requested",
            "message": "ML training request sent. Training will start shortly."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send training request: {e}")


@router.post("/yolo/start")
async def start_yolo_training():
    """Trigger YOLO training"""
    try:
        nc = await get_nats()
        await nc.publish(
            "training.yolo.requested",
            json.dumps({
                "requested_at": datetime.utcnow().isoformat(),
                "manual": True
            }).encode()
        )
        
        return {
            "status": "training_requested",
            "message": "YOLO training request sent."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send training request: {e}")


@router.get("/models")
async def get_trained_models():
    """List trained models"""
    models_dir = TRAINING_DIR / "models"
    models = []
    
    if models_dir.exists():
        for model_file in models_dir.glob("*.joblib"):
            stat = model_file.stat()
            models.append({
                "name": model_file.stem,
                "file": model_file.name,
                "size_kb": stat.st_size / 1024,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    return {
        "models": models,
        "total": len(models)
    }
