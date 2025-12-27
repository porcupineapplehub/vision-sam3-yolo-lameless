"""
T-LEAP Style Pose Estimation Pipeline
Implements cow pose estimation using YOLOv8-Pose trained on cow dataset.

Based on papers:
- T-LEAP: Occlusion-robust pose estimation of walking cows using temporal information
- Lameness detection in dairy cows using pose estimation and bidirectional LSTMs
- Video-based Automatic Lameness Detection using Pose Estimation and Multiple Locomotion Traits

Cow Keypoints (20 points from Animal Pose Dataset):
0: left_eye
1: right_eye
2: nose
3: left_ear
4: right_ear
5: left_front_elbow
6: right_front_elbow
7: left_back_elbow
8: right_back_elbow
9: left_front_knee
10: right_front_knee
11: left_back_knee
12: right_back_knee
13: left_front_paw
14: right_front_paw
15: left_back_paw
16: right_back_paw
17: throat
18: withers
19: tailbase
"""
import asyncio
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import yaml
from shared.utils.nats_client import NATSClient

# Keypoint names (20 keypoints from Animal Pose Dataset)
KEYPOINT_NAMES = [
    'left_eye', 'right_eye', 'nose', 'left_ear', 'right_ear',
    'left_front_elbow', 'right_front_elbow', 'left_back_elbow', 'right_back_elbow',
    'left_front_knee', 'right_front_knee', 'left_back_knee', 'right_back_knee',
    'left_front_paw', 'right_front_paw', 'left_back_paw', 'right_back_paw',
    'throat', 'withers', 'tailbase'
]

# Skeleton connections
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
    'face': (0, 255, 255),       # Yellow - face/head
    'spine': (0, 255, 0),        # Green - spine/back line
    'front_left': (255, 0, 0),   # Blue - front left leg
    'front_right': (0, 165, 255), # Orange - front right leg
    'back_left': (255, 0, 255),  # Magenta - back left leg
    'back_right': (0, 255, 255), # Cyan - back right leg
}

# Map skeleton connections to colors
SKELETON_CONNECTION_COLORS = {
    (0, 1): 'face', (0, 2): 'face', (1, 2): 'face', (0, 3): 'face', (1, 4): 'face',
    (2, 17): 'spine', (17, 18): 'spine', (18, 19): 'spine',
    (5, 9): 'front_left', (9, 13): 'front_left',
    (6, 10): 'front_right', (10, 14): 'front_right',
    (7, 11): 'back_left', (11, 15): 'back_left',
    (8, 12): 'back_right', (12, 16): 'back_right',
}


class CowPoseEstimator:
    """
    Cow pose estimator using trained YOLOv8-Pose model.
    Falls back to heuristic estimation if model not available.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.use_trained_model = False
        
        # Try to load trained cow pose model
        try:
            from ultralytics import YOLO
            
            # Check for trained cow pose model
            cow_pose_model = Path("/app/data/models/cow_pose_full.pt")
            if cow_pose_model.exists():
                self.model = YOLO(str(cow_pose_model))
                self.use_trained_model = True
                print(f"✅ Loaded trained cow pose model: {cow_pose_model}")
            elif model_path and Path(model_path).exists():
                self.model = YOLO(model_path)
                self.use_trained_model = True
                print(f"✅ Loaded custom model: {model_path}")
            else:
                # Fall back to pretrained COCO model for detection only
                self.model = YOLO("yolov8n.pt")
                self.use_trained_model = False
                print("⚠️ Using pretrained YOLO (no cow pose model found)")
                print("   Pose estimation will use heuristic approach")
        except Exception as e:
            print(f"❌ Could not load YOLO model: {e}")
            self.model = None
    
    def detect_with_trained_model(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect cows and keypoints using trained YOLOv8-Pose model.
        
        Uses hybrid approach: model keypoints for face (high confidence),
        heuristic estimation for body (when model confidence is low).
        """
        detections = []
        height, width = frame.shape[:2]
        results = self.model(frame, verbose=False, conf=0.3)
        
        for result in results:
            if result.boxes is None or len(result.boxes) == 0:
                continue
            
            for j, box in enumerate(result.boxes):
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Get keypoints from pose model
                model_keypoints = {}
                if result.keypoints is not None and j < len(result.keypoints):
                    kpts = result.keypoints[j].data[0].cpu().numpy()
                    for i, kp in enumerate(kpts):
                        name = KEYPOINT_NAMES[i] if i < len(KEYPOINT_NAMES) else f'kp_{i}'
                        model_keypoints[name] = {
                            'name': name,
                            'x': float(kp[0]),
                            'y': float(kp[1]),
                            'confidence': float(kp[2]) if len(kp) > 2 else 1.0
                        }
                
                # Get heuristic keypoints based on bounding box
                heuristic_kps = self.estimate_pose_from_bbox(frame, bbox, height, width)
                heuristic_dict = {kp['name']: kp for kp in heuristic_kps}
                
                # Hybrid approach: use model keypoints if confidence > 0.3, else heuristic
                final_keypoints = []
                for name in KEYPOINT_NAMES:
                    if name in model_keypoints and model_keypoints[name]['confidence'] > 0.3:
                        # Use model keypoint (high confidence)
                        final_keypoints.append(model_keypoints[name])
                    elif name in heuristic_dict:
                        # Fall back to heuristic
                        final_keypoints.append(heuristic_dict[name])
                    elif name in model_keypoints:
                        # Use model keypoint even if low confidence (better than nothing)
                        final_keypoints.append(model_keypoints[name])
                
                detections.append({
                    'bbox': bbox,
                    'confidence': confidence,
                    'class': 'cow',
                    'keypoints': final_keypoints
                })
        
        return detections
    
    def estimate_pose_from_bbox(
        self, 
        frame: np.ndarray, 
        bbox: List[float],
        frame_height: int,
        frame_width: int
    ) -> List[Dict[str, Any]]:
        """
        Estimate cow keypoints from bounding box using anatomical proportions.
        Fallback when no trained pose model is available.
        """
        x1, y1, x2, y2 = [int(c) for c in bbox]
        width = x2 - x1
        height = y2 - y1
        
        keypoints = []
        
        # Head area (front of cow)
        head_x = x1 + width * 0.1
        head_y = y1 + height * 0.3
        
        # Left eye
        keypoints.append({'name': 'left_eye', 'x': head_x - width * 0.02, 'y': head_y - height * 0.05, 'confidence': 0.7})
        # Right eye
        keypoints.append({'name': 'right_eye', 'x': head_x + width * 0.02, 'y': head_y - height * 0.05, 'confidence': 0.7})
        # Nose
        keypoints.append({'name': 'nose', 'x': head_x, 'y': head_y + height * 0.05, 'confidence': 0.8})
        # Left ear
        keypoints.append({'name': 'left_ear', 'x': head_x - width * 0.05, 'y': head_y - height * 0.1, 'confidence': 0.6})
        # Right ear
        keypoints.append({'name': 'right_ear', 'x': head_x + width * 0.05, 'y': head_y - height * 0.1, 'confidence': 0.6})
        
        # Front leg joints
        front_x = x1 + width * 0.25
        # Left front elbow
        keypoints.append({'name': 'left_front_elbow', 'x': front_x - width * 0.05, 'y': y1 + height * 0.4, 'confidence': 0.7})
        # Right front elbow
        keypoints.append({'name': 'right_front_elbow', 'x': front_x + width * 0.05, 'y': y1 + height * 0.4, 'confidence': 0.7})
        
        # Back leg joints
        back_x = x1 + width * 0.75
        # Left back elbow
        keypoints.append({'name': 'left_back_elbow', 'x': back_x - width * 0.05, 'y': y1 + height * 0.4, 'confidence': 0.7})
        # Right back elbow
        keypoints.append({'name': 'right_back_elbow', 'x': back_x + width * 0.05, 'y': y1 + height * 0.4, 'confidence': 0.7})
        
        # Front knees
        keypoints.append({'name': 'left_front_knee', 'x': front_x - width * 0.03, 'y': y1 + height * 0.6, 'confidence': 0.7})
        keypoints.append({'name': 'right_front_knee', 'x': front_x + width * 0.07, 'y': y1 + height * 0.6, 'confidence': 0.7})
        
        # Back knees
        keypoints.append({'name': 'left_back_knee', 'x': back_x - width * 0.07, 'y': y1 + height * 0.6, 'confidence': 0.7})
        keypoints.append({'name': 'right_back_knee', 'x': back_x + width * 0.03, 'y': y1 + height * 0.6, 'confidence': 0.7})
        
        # Paws (hooves)
        ground_y = y2 - height * 0.05
        keypoints.append({'name': 'left_front_paw', 'x': front_x - width * 0.02, 'y': ground_y, 'confidence': 0.7})
        keypoints.append({'name': 'right_front_paw', 'x': front_x + width * 0.08, 'y': ground_y, 'confidence': 0.7})
        keypoints.append({'name': 'left_back_paw', 'x': back_x - width * 0.08, 'y': ground_y, 'confidence': 0.7})
        keypoints.append({'name': 'right_back_paw', 'x': back_x + width * 0.02, 'y': ground_y, 'confidence': 0.7})
        
        # Spine points
        keypoints.append({'name': 'throat', 'x': x1 + width * 0.15, 'y': y1 + height * 0.25, 'confidence': 0.8})
        keypoints.append({'name': 'withers', 'x': x1 + width * 0.3, 'y': y1 + height * 0.15, 'confidence': 0.8})
        keypoints.append({'name': 'tailbase', 'x': x1 + width * 0.9, 'y': y1 + height * 0.25, 'confidence': 0.7})
        
        return keypoints
    
    def detect_with_heuristic(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect cows using YOLO and estimate pose using heuristics."""
        height, width = frame.shape[:2]
        detections = []
        
        if self.model is not None:
            results = self.model(frame, verbose=False, conf=0.3)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[cls] if hasattr(self.model, 'names') else ''
                    
                    # Look for cow class (class 19 in COCO is 'cow')
                    if cls == 19 or 'cow' in class_name.lower():
                        bbox = box.xyxy[0].cpu().numpy().tolist()
                        confidence = float(box.conf[0].cpu().numpy())
                        keypoints = self.estimate_pose_from_bbox(frame, bbox, height, width)
                        
                        detections.append({
                            'bbox': bbox,
                            'confidence': confidence,
                            'class': class_name,
                            'keypoints': keypoints
                        })
        
        # Fallback if no detection
        if not detections and height > 0 and width > 0:
            margin = 0.1
            bbox = [width * margin, height * margin, width * (1 - margin), height * (1 - margin)]
            keypoints = self.estimate_pose_from_bbox(frame, bbox, height, width)
            detections.append({
                'bbox': bbox,
                'confidence': 0.5,
                'class': 'cow_assumed',
                'keypoints': keypoints
            })
        
        return detections
    
    def detect_and_estimate(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect cows and estimate their poses."""
        if self.use_trained_model:
            return self.detect_with_trained_model(frame)
        else:
            return self.detect_with_heuristic(frame)


class TLEAPPipeline:
    """T-LEAP style pose estimation pipeline for cow lameness detection"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        self.results_dir = Path("/app/data/results/tleap")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pose estimator
        model_path = self.config.get("models", {}).get("tleap", {}).get("checkpoint_path")
        self.pose_estimator = CowPoseEstimator(model_path)
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def compute_locomotion_features(self, pose_sequences: List[Dict]) -> Dict[str, float]:
        """
        Compute locomotion features from pose sequences.
        
        Features based on lameness literature:
        - Back arch angle (arched back indicates pain)
        - Head bob magnitude (excessive bobbing indicates lameness)
        - Stride length variability
        - Leg symmetry (asymmetric gait)
        """
        if not pose_sequences or len(pose_sequences) < 2:
            return {}
        
        features = {}
        
        # Extract positions over time
        head_positions = []  # Using nose position
        hoof_positions = {'fl': [], 'fr': [], 'rl': [], 'rr': []}
        spine_angles = []
        
        for frame_data in pose_sequences:
            keypoints = frame_data.get('keypoints', [])
            if len(keypoints) < 20:
                continue
            
            # Create keypoint dict for easier access
            kp_dict = {kp['name']: kp for kp in keypoints}
            
            # Head position for head bob (using nose)
            nose = kp_dict.get('nose', {})
            if nose.get('confidence', 0) > 0.3:
                head_positions.append(nose.get('y', 0))
            
            # Spine angle (throat -> withers -> tailbase)
            throat = kp_dict.get('throat', {})
            withers = kp_dict.get('withers', {})
            tailbase = kp_dict.get('tailbase', {})
            
            if all(k.get('confidence', 0) > 0.3 for k in [throat, withers, tailbase]):
                v1 = np.array([throat['x'] - withers['x'], throat['y'] - withers['y']])
                v2 = np.array([tailbase['x'] - withers['x'], tailbase['y'] - withers['y']])
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                spine_angles.append(angle)
            
            # Hoof positions for stride analysis
            for leg, kp_name in [('fl', 'left_front_paw'), ('fr', 'right_front_paw'), 
                                  ('rl', 'left_back_paw'), ('rr', 'right_back_paw')]:
                kp = kp_dict.get(kp_name, {})
                if kp.get('confidence', 0) > 0.3:
                    hoof_positions[leg].append(kp.get('x', 0))
        
        # Calculate features
        if spine_angles:
            features['back_arch_mean'] = float(np.mean(spine_angles))
            features['back_arch_std'] = float(np.std(spine_angles))
            features['back_arch_score'] = float(1.0 - (np.mean(spine_angles) / 180.0))
        
        if len(head_positions) > 1:
            features['head_bob_magnitude'] = float(np.std(head_positions))
            head_diff = np.diff(head_positions)
            features['head_bob_frequency'] = float(np.sum(np.abs(np.diff(np.sign(head_diff)))) / 2)
            features['head_bob_score'] = float(min(1.0, features['head_bob_magnitude'] / 50.0))
        
        # Stride analysis
        for leg, positions in hoof_positions.items():
            if len(positions) > 1:
                strides = np.diff(positions)
                features[f'stride_{leg}_mean'] = float(np.mean(np.abs(strides)))
                features[f'stride_{leg}_std'] = float(np.std(strides))
        
        # Leg symmetry
        if 'stride_fl_mean' in features and 'stride_fr_mean' in features:
            features['front_leg_asymmetry'] = float(
                abs(features['stride_fl_mean'] - features['stride_fr_mean']) /
                (features['stride_fl_mean'] + features['stride_fr_mean'] + 1e-6)
            )
        
        if 'stride_rl_mean' in features and 'stride_rr_mean' in features:
            features['rear_leg_asymmetry'] = float(
                abs(features['stride_rl_mean'] - features['stride_rr_mean']) /
                (features['stride_rl_mean'] + features['stride_rr_mean'] + 1e-6)
            )
        
        # Overall lameness score (0-1, higher = more likely lame)
        score_components = []
        if 'back_arch_score' in features:
            score_components.append(features['back_arch_score'])
        if 'head_bob_score' in features:
            score_components.append(features['head_bob_score'])
        if 'front_leg_asymmetry' in features:
            score_components.append(features['front_leg_asymmetry'])
        if 'rear_leg_asymmetry' in features:
            score_components.append(features['rear_leg_asymmetry'])
        
        if score_components:
            features['lameness_score'] = float(np.mean(score_components))
        
        return features
    
    async def process_video(self, video_data: dict):
        """Process video for pose estimation"""
        video_id = video_data.get("video_id")
        processed_path = Path(video_data.get("processed_path", ""))
        
        print(f"T-LEAP pipeline processing video {video_id}")
        
        # Find video file
        videos_dir = Path("/app/data/videos")
        video_files = list(videos_dir.glob(f"{video_id}.*"))
        if processed_path.exists():
            video_path = processed_path
        elif video_files:
            video_path = video_files[0]
        else:
            print(f"Video not found: {video_id}")
            return
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Failed to open video: {video_path}")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            pose_sequences = []
            frame_count = 0
            
            # Process every frame for better accuracy
            frame_interval = max(1, int(fps // 5))  # 5 FPS processing
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    detections = self.pose_estimator.detect_and_estimate(frame)
                    
                    for det in detections:
                        pose_sequences.append({
                            'frame': frame_count,
                            'time': frame_count / fps if fps > 0 else 0,
                            'bbox': det['bbox'],
                            'keypoints': det['keypoints'],
                            'detection_confidence': det['confidence']
                        })
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            
            # Compute locomotion features
            features = self.compute_locomotion_features(pose_sequences)
            
            # Save results
            result = {
                'video_id': video_id,
                'pipeline': 'tleap',
                'total_frames': total_frames,
                'fps': fps,
                'frames_processed': len(pose_sequences),
                'pose_sequences': pose_sequences,
                'locomotion_features': features,
                'model_type': 'trained' if self.pose_estimator.use_trained_model else 'heuristic',
                'skeleton_definition': {
                    'keypoint_names': KEYPOINT_NAMES,
                    'skeleton_connections': COW_SKELETON,
                    'colors': {k: list(v) for k, v in SKELETON_COLORS.items()}
                }
            }
            
            results_file = self.results_dir / f"{video_id}_tleap.json"
            with open(results_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            # Publish results
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["pipeline_tleap"],
                {
                    'video_id': video_id,
                    'pipeline': 'tleap',
                    'results_path': str(results_file),
                    'features': features,
                    'frames_processed': len(pose_sequences),
                    'model_type': result['model_type']
                }
            )
            
            print(f"✅ T-LEAP pipeline completed for {video_id}")
            print(f"   Model: {result['model_type']}")
            print(f"   Processed {len(pose_sequences)} frames")
            if features:
                print(f"   Lameness score: {features.get('lameness_score', 'N/A'):.3f}")
            
        except Exception as e:
            print(f"❌ Error in T-LEAP pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the T-LEAP pipeline service"""
        await self.nats_client.connect()
        subject = self.config["nats"]["subjects"]["video_preprocessed"]
        print(f"T-LEAP pipeline subscribed to {subject}")
        await self.nats_client.subscribe(subject, self.process_video)
        print("=" * 60)
        print("T-LEAP Pipeline Service Started")
        print("=" * 60)
        print(f"Model: {'Trained YOLOv8-Pose' if self.pose_estimator.use_trained_model else 'Heuristic'}")
        print(f"Keypoints: {len(KEYPOINT_NAMES)}")
        print("Keypoints tracked: " + ", ".join(KEYPOINT_NAMES[:5]) + "...")
        print("=" * 60)
        await asyncio.Event().wait()


async def main():
    pipeline = TLEAPPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())
