"""
Annotation Renderer Service
Renders YOLO bounding boxes and pose keypoints on videos
Like in the research papers: T-LEAP, BiLSTM lameness detection
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import yaml

app = FastAPI(title="Annotation Renderer Service")

# Directories
VIDEOS_DIR = Path("/app/data/videos")
PROCESSED_DIR = Path("/app/data/processed")
RESULTS_DIR = Path("/app/data/results")
ANNOTATED_DIR = PROCESSED_DIR / "annotated"

# Ensure directories exist
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

# Keypoint names (20 keypoints from trained model)
KEYPOINT_NAMES = [
    'left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear',
    'left_front_elbow', 'right_front_elbow', 'left_back_elbow', 'right_back_elbow',
    'left_front_knee', 'right_front_knee', 'left_back_knee', 'right_back_knee',
    'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
    'throat', 'withers', 'tailbase'
]

# Cow skeleton connections (20 keypoints)
COW_SKELETON = [
    (0, 1),   # left_eye - right_eye
    (0, 2),   # left_eye - nose
    (1, 2),   # right_eye - nose
    (0, 3),   # left_eye - left_ear
    (1, 4),   # right_eye - right_ear
    (2, 17),  # nose - throat
    (17, 18), # throat - withers
    (18, 19), # withers - tailbase
    (5, 9),   # left_front_elbow - left_front_knee
    (6, 10),  # right_front_elbow - right_front_knee
    (7, 11),  # left_back_elbow - left_back_knee
    (8, 12),  # right_back_elbow - right_back_knee
    (9, 13),  # left_front_knee - left_front_paw
    (10, 14), # right_front_knee - right_front_paw
    (11, 15), # left_back_knee - left_back_paw
    (12, 16), # right_back_knee - right_back_paw
]

# Color scheme for different body parts (BGR format)
SKELETON_COLORS = {
    'face': (0, 255, 255),        # Yellow - face/head
    'spine': (0, 255, 0),         # Green - spine/back line
    'front_left': (255, 0, 0),    # Blue - front left leg
    'front_right': (0, 165, 255), # Orange - front right leg
    'back_left': (255, 0, 255),   # Magenta - back left leg
    'back_right': (0, 255, 255),  # Cyan - back right leg
}

# Map skeleton connections to color groups
SKELETON_CONNECTION_GROUPS = {
    (0, 1): 'face', (0, 2): 'face', (1, 2): 'face', (0, 3): 'face', (1, 4): 'face',
    (2, 17): 'spine', (17, 18): 'spine', (18, 19): 'spine',
    (5, 9): 'front_left', (9, 13): 'front_left',
    (6, 10): 'front_right', (10, 14): 'front_right',
    (7, 11): 'back_left', (11, 15): 'back_left',
    (8, 12): 'back_right', (12, 16): 'back_right',
}

# Keypoint colors by index
KEYPOINT_COLOR_MAP = {
    0: (0, 255, 255),   # left_eye - Yellow
    1: (0, 255, 255),   # right_eye - Yellow
    2: (0, 0, 255),     # nose - Red (most visible)
    3: (0, 255, 255),   # left_ear - Yellow
    4: (0, 255, 255),   # right_ear - Yellow
    5: (255, 0, 0),     # left_front_elbow - Blue
    6: (0, 165, 255),   # right_front_elbow - Orange
    7: (255, 0, 255),   # left_back_elbow - Magenta
    8: (0, 255, 255),   # right_back_elbow - Cyan
    9: (255, 0, 0),     # left_front_knee - Blue
    10: (0, 165, 255),  # right_front_knee - Orange
    11: (255, 0, 255),  # left_back_knee - Magenta
    12: (0, 255, 255),  # right_back_knee - Cyan
    13: (255, 0, 0),    # left_front_paw - Blue
    14: (0, 165, 255),  # right_front_paw - Orange
    15: (255, 0, 255),  # left_back_paw - Magenta
    16: (0, 255, 255),  # right_back_paw - Cyan
    17: (0, 255, 0),    # throat - Green
    18: (0, 255, 0),    # withers - Green
    19: (0, 255, 0),    # tailbase - Green
}


class RenderRequest(BaseModel):
    video_id: str
    include_yolo: bool = True
    include_pose: bool = True
    show_confidence: bool = False
    show_labels: bool = True
    output_fps: Optional[float] = None


# Track rendering progress
render_status: Dict[str, Dict] = {}


def draw_skeleton_on_frame(
    frame: np.ndarray,
    keypoints: List[Dict],
    bbox: Optional[List[float]] = None,
    confidence_threshold: float = 0.3,
    show_labels: bool = True,
    show_confidence: bool = False
) -> np.ndarray:
    """Draw cow skeleton on a single frame - like in the papers."""
    
    # Draw bounding box
    if bbox:
        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if len(bbox) > 4:  # Has confidence
            cv2.putText(frame, f"Cow {bbox[4]:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if not keypoints:
        return frame
    
    # Draw skeleton connections first (behind keypoints)
    for (start_idx, end_idx) in COW_SKELETON:
        if start_idx >= len(keypoints) or end_idx >= len(keypoints):
            continue
        
        kp1 = keypoints[start_idx]
        kp2 = keypoints[end_idx]
        
        # Get confidence
        conf1 = kp1.get('confidence', 0)
        conf2 = kp2.get('confidence', 0)
        
        if conf1 > confidence_threshold and conf2 > confidence_threshold:
            x1, y1 = int(kp1.get('x', 0)), int(kp1.get('y', 0))
            x2, y2 = int(kp2.get('x', 0)), int(kp2.get('y', 0))
            
            # Get color for this connection
            group = SKELETON_CONNECTION_GROUPS.get((start_idx, end_idx), 'spine')
            color = SKELETON_COLORS.get(group, (255, 255, 255))
            
            cv2.line(frame, (x1, y1), (x2, y2), color, 3)
    
    # Draw keypoints on top
    for i, kp in enumerate(keypoints):
        conf = kp.get('confidence', 0)
        
        if conf > confidence_threshold:
            x, y = int(kp.get('x', 0)), int(kp.get('y', 0))
            
            # Get color for this keypoint
            color = KEYPOINT_COLOR_MAP.get(i, (255, 255, 255))
            
            # Draw keypoint circle
            cv2.circle(frame, (x, y), 5, color, -1)
            cv2.circle(frame, (x, y), 6, (255, 255, 255), 1)  # White border
            
            # Draw label for important keypoints
            if show_labels and i in [2, 18, 19]:  # nose, withers, tailbase
                label = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f"kp{i}"
                cv2.putText(frame, label, (x + 8, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw confidence
            if show_confidence:
                cv2.putText(frame, f"{conf:.2f}", (x + 8, y + 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    return frame


def draw_yolo_on_frame(
    frame: np.ndarray,
    detections: List[Dict],
    show_confidence: bool = True
) -> np.ndarray:
    """Draw YOLO detection boxes on frame."""
    for det in detections:
        bbox = det.get('bbox', [])
        if len(bbox) < 4:
            continue
        
        x1, y1, x2, y2 = [int(c) for c in bbox[:4]]
        confidence = det.get('confidence', 0)
        class_name = det.get('class', 'cow')
        
        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = class_name
        if show_confidence:
            label = f"{class_name} {confidence:.2f}"
        
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 5, y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame


def draw_info_overlay(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    lameness_score: Optional[float] = None
) -> np.ndarray:
    """Draw information overlay on frame."""
    h, w = frame.shape[:2]
    
    # Draw semi-transparent background for info
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    
    # Frame info
    time_sec = frame_idx / fps if fps > 0 else 0
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Time: {time_sec:.2f}s", (20, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Lameness score
    if lameness_score is not None:
        score_pct = lameness_score * 100
        if score_pct < 30:
            color = (0, 255, 0)  # Green
            status = "Normal"
        elif score_pct < 60:
            color = (0, 165, 255)  # Orange
            status = "Mild"
        else:
            color = (0, 0, 255)  # Red
            status = "Lame"
        
        cv2.putText(frame, f"Lameness: {score_pct:.1f}% ({status})", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw color legend
    legend_x = w - 160
    legend_y = h - 140
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(frame, (legend_x - 10, legend_y - 20), (w - 10, h - 10), (255, 255, 255), 1)
    
    cv2.putText(frame, "Legend:", (legend_x, legend_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    legend_items = [
        ("Face", SKELETON_COLORS['face']),
        ("Spine", SKELETON_COLORS['spine']),
        ("Front L", SKELETON_COLORS['front_left']),
        ("Front R", SKELETON_COLORS['front_right']),
        ("Back L", SKELETON_COLORS['back_left']),
        ("Back R", SKELETON_COLORS['back_right']),
    ]
    
    for i, (name, color) in enumerate(legend_items):
        y = legend_y + 18 + i * 18
        cv2.circle(frame, (legend_x + 8, y - 5), 5, color, -1)
        cv2.putText(frame, name, (legend_x + 20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame


async def render_annotated_video(request: RenderRequest):
    """Render annotated video with pose and/or YOLO detections."""
    video_id = request.video_id
    
    # Update status
    render_status[video_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Loading data...'
    }
    
    try:
        # Find video file
        video_files = list(VIDEOS_DIR.glob(f"{video_id}.*"))
        if not video_files:
            render_status[video_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Video file not found'
            }
            return
        
        video_path = video_files[0]
        
        # Load YOLO results
        yolo_data = None
        if request.include_yolo:
            yolo_file = RESULTS_DIR / "yolo" / f"{video_id}_yolo.json"
            if yolo_file.exists():
                with open(yolo_file) as f:
                    yolo_data = json.load(f)
        
        # Load T-LEAP pose results
        pose_data = None
        lameness_score = None
        if request.include_pose:
            pose_file = RESULTS_DIR / "tleap" / f"{video_id}_tleap.json"
            if pose_file.exists():
                with open(pose_file) as f:
                    pose_data = json.load(f)
                    lameness_score = pose_data.get('locomotion_features', {}).get('lameness_score')
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            render_status[video_id] = {
                'status': 'error',
                'progress': 0,
                'message': 'Failed to open video'
            }
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_fps = request.output_fps or fps
        
        # Create output video
        output_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (width, height))
        
        # Create frame lookup from pose data
        pose_by_frame = {}
        if pose_data and 'pose_sequences' in pose_data:
            for seq in pose_data['pose_sequences']:
                frame_idx = seq.get('frame', 0)
                pose_by_frame[frame_idx] = seq
        
        # Create frame lookup from YOLO data
        yolo_by_frame = {}
        if yolo_data and 'detections' in yolo_data:
            for det in yolo_data['detections']:
                if isinstance(det, dict):
                    frame_idx = det.get('frame', 0)
                    yolo_by_frame[frame_idx] = det.get('detections', [])
        
        render_status[video_id] = {
            'status': 'rendering',
            'progress': 0,
            'message': 'Rendering frames...'
        }
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Draw YOLO detections
            if request.include_yolo and frame_idx in yolo_by_frame:
                frame = draw_yolo_on_frame(frame, yolo_by_frame[frame_idx], 
                                          request.show_confidence)
            
            # Draw pose skeleton
            if request.include_pose and frame_idx in pose_by_frame:
                pose_seq = pose_by_frame[frame_idx]
                keypoints = pose_seq.get('keypoints', [])
                bbox = pose_seq.get('bbox')
                
                # Add confidence to bbox if available
                if bbox and pose_seq.get('detection_confidence'):
                    bbox = bbox + [pose_seq['detection_confidence']]
                
                frame = draw_skeleton_on_frame(
                    frame, keypoints, bbox,
                    show_labels=request.show_labels,
                    show_confidence=request.show_confidence
                )
            
            # Draw info overlay
            frame = draw_info_overlay(frame, frame_idx, fps, lameness_score)
            
            out.write(frame)
            frame_idx += 1
            
            # Update progress
            if frame_idx % 30 == 0:
                progress = frame_idx / total_frames * 100
                render_status[video_id] = {
                    'status': 'rendering',
                    'progress': progress,
                    'message': f'Rendering frame {frame_idx}/{total_frames}'
                }
        
        cap.release()
        out.release()
        
        # Convert to web-compatible format using ffmpeg if available
        try:
            import subprocess
            web_output = ANNOTATED_DIR / f"{video_id}_annotated_web.mp4"
            subprocess.run([
                'ffmpeg', '-y', '-i', str(output_path),
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-movflags', '+faststart',
                str(web_output)
            ], capture_output=True, check=True)
            # Replace with web-compatible version
            output_path.unlink()
            web_output.rename(output_path)
        except Exception:
            pass  # Keep original if ffmpeg fails
        
        render_status[video_id] = {
            'status': 'complete',
            'progress': 100,
            'message': 'Rendering complete',
            'output_path': str(output_path)
        }
        
        print(f"✅ Rendered annotated video: {output_path}")
        
    except Exception as e:
        print(f"❌ Error rendering video {video_id}: {e}")
        import traceback
        traceback.print_exc()
        render_status[video_id] = {
            'status': 'error',
            'progress': 0,
            'message': str(e)
        }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/render")
async def start_render(request: RenderRequest, background_tasks: BackgroundTasks):
    """Start rendering an annotated video."""
    video_id = request.video_id
    
    # Check if already rendering
    if video_id in render_status and render_status[video_id].get('status') == 'rendering':
        return {"status": "already_rendering", "video_id": video_id}
    
    # Start background rendering
    background_tasks.add_task(render_annotated_video, request)
    
    return {"status": "started", "video_id": video_id}


@app.get("/status/{video_id}")
async def get_status(video_id: str):
    """Get rendering status for a video."""
    if video_id not in render_status:
        # Check if annotated file exists
        output_path = ANNOTATED_DIR / f"{video_id}_annotated.mp4"
        if output_path.exists():
            return {
                "status": "complete",
                "progress": 100,
                "message": "Annotated video available",
                "output_path": str(output_path)
            }
        return {"status": "not_found", "progress": 0, "message": "No render status found"}
    
    return render_status[video_id]


@app.delete("/status/{video_id}")
async def clear_status(video_id: str):
    """Clear rendering status for a video."""
    if video_id in render_status:
        del render_status[video_id]
    return {"status": "cleared", "video_id": video_id}


@app.get("/")
async def root():
    return {
        "service": "Annotation Renderer",
        "version": "1.0",
        "keypoints": len(KEYPOINT_NAMES),
        "skeleton_connections": len(COW_SKELETON)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
