"""
SAM3 Segmentation Pipeline
Uses SAM3 (Segment Anything Model 3) for precise cow segmentation
"""
import asyncio
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from shared.utils.nats_client import NATSClient
from typing import List, Dict, Any

# Try to import SAM3, fallback to basic segmentation if not available
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 not available. Using basic segmentation fallback.")


class SAM3Pipeline:
    """SAM3 segmentation pipeline"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Initialize SAM3 model if available
        self.sam_predictor = None
        if SAM3_AVAILABLE:
            self._load_sam3_model()
        
        # Directories
        self.processed_dir = Path("/app/data/processed")
        self.results_dir = Path("/app/data/results/sam3")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for YOLO results
        self.yolo_results_cache = {}
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_sam3_model(self):
        """Load SAM3 model"""
        try:
            checkpoint_path = Path("/app/shared/models/sam3")
            if checkpoint_path.exists() and list(checkpoint_path.glob("*.pth")):
                checkpoint_file = list(checkpoint_path.glob("*.pth"))[0]
                # Determine model type from checkpoint name
                if "vit_h" in checkpoint_file.name:
                    model_type = "vit_h"
                elif "vit_l" in checkpoint_file.name:
                    model_type = "vit_l"
                else:
                    model_type = "vit_b"
                
                sam = sam_model_registry[model_type](checkpoint=str(checkpoint_file))
                self.sam_predictor = SamPredictor(sam)
                print(f"Loaded SAM3 model: {checkpoint_file}")
            else:
                print("SAM3 checkpoint not found. Using fallback segmentation.")
        except Exception as e:
            print(f"Failed to load SAM3 model: {e}")
            print("Using fallback segmentation.")
    
    def segment_with_sam3(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Segment using SAM3 with bounding box prompt"""
        if self.sam_predictor is None:
            return self._fallback_segmentation(image, bbox)
        
        try:
            self.sam_predictor.set_image(image)
            x1, y1, x2, y2 = bbox
            box = np.array([x1, y1, x2, y2])
            masks, scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            return masks[0]  # Return best mask
        except Exception as e:
            print(f"SAM3 segmentation error: {e}")
            return self._fallback_segmentation(image, bbox)
    
    def _fallback_segmentation(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """Fallback segmentation using simple mask"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        mask[y1:y2, x1:x2] = 255
        return mask.astype(bool)
    
    def extract_segmentation_features(self, mask: np.ndarray) -> Dict[str, float]:
        """Extract features from segmentation mask"""
        mask_area = np.sum(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        area_ratio = mask_area / total_pixels if total_pixels > 0 else 0
        
        # Compute mask properties
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest_contour, True)
            # Circularity: 4π * area / perimeter²
            circularity = (4 * np.pi * cv2.contourArea(largest_contour)) / (perimeter ** 2) if perimeter > 0 else 0
            
            # Bounding box of mask
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
        else:
            circularity = 0
            aspect_ratio = 0
        
        # Compute centroid
        M = cv2.moments(mask.astype(np.uint8))
        if M["m00"] != 0:
            centroid_x = M["m10"] / M["m00"]
            centroid_y = M["m01"] / M["m00"]
        else:
            centroid_x = mask.shape[1] / 2
            centroid_y = mask.shape[0] / 2
        
        return {
            "mask_area": float(mask_area),
            "area_ratio": float(area_ratio),
            "circularity": float(circularity),
            "aspect_ratio": float(aspect_ratio),
            "centroid_x": float(centroid_x),
            "centroid_y": float(centroid_y),
            "perimeter": float(perimeter) if contours else 0.0
        }
    
    async def get_yolo_results(self, video_id: str) -> Dict[str, Any]:
        """Get YOLO results for video (from cache or file)"""
        if video_id in self.yolo_results_cache:
            return self.yolo_results_cache[video_id]
        
        # Try to load from file
        yolo_results_dir = Path("/app/data/results/yolo")
        yolo_file = yolo_results_dir / f"{video_id}_yolo.json"
        
        if yolo_file.exists():
            with open(yolo_file) as f:
                results = json.load(f)
                self.yolo_results_cache[video_id] = results
                return results
        
        return {}
    
    async def process_video(self, video_data: dict):
        """Process a preprocessed video"""
        video_id = video_data["video_id"]
        processed_path = Path(video_data["processed_path"])
        
        print(f"SAM3 pipeline processing video {video_id}")
        
        if not processed_path.exists():
            print(f"Processed video not found: {processed_path}")
            return
        
        try:
            # Get YOLO results for bounding boxes
            yolo_results = await self.get_yolo_results(video_id)
            
            # Process video frames
            cap = cv2.VideoCapture(str(processed_path))
            if not cap.isOpened():
                raise Exception(f"Failed to open video: {processed_path}")
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            segmentations = []
            frame_features = []
            frame_interval = max(1, fps // 2)  # Process 2 frames per second
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Get detection for this frame from YOLO results
                    bbox = None
                    if yolo_results and "detections" in yolo_results:
                        for det in yolo_results["detections"]:
                            if det["frame"] == frame_count:
                                if det["detections"]:
                                    # Use first detection's bbox
                                    bbox = det["detections"][0]["bbox"]
                                    break
                    
                    if bbox:
                        # Segment using SAM3
                        mask = self.segment_with_sam3(frame, bbox)
                        
                        # Extract features
                        features = self.extract_segmentation_features(mask)
                        features["frame"] = frame_count
                        features["time"] = frame_count / fps if fps > 0 else 0
                        
                        frame_features.append(features)
                        
                        segmentations.append({
                            "frame": frame_count,
                            "time": frame_count / fps if fps > 0 else 0,
                            "mask_available": True,
                            "features": features
                        })
                    else:
                        segmentations.append({
                            "frame": frame_count,
                            "time": frame_count / fps if fps > 0 else 0,
                            "mask_available": False
                        })
                
                frame_count += 1
            
            cap.release()
            
            # Aggregate features
            if frame_features:
                avg_features = {
                    "avg_mask_area": np.mean([f["mask_area"] for f in frame_features]),
                    "avg_area_ratio": np.mean([f["area_ratio"] for f in frame_features]),
                    "avg_circularity": np.mean([f["circularity"] for f in frame_features]),
                    "avg_aspect_ratio": np.mean([f["aspect_ratio"] for f in frame_features]),
                }
            else:
                avg_features = {}
            
            # Save results
            results = {
                "segmentations": segmentations,
                "aggregated_features": avg_features,
                "total_frames": total_frames,
                "fps": fps,
                "frames_processed": len(segmentations)
            }
            
            results_file = self.results_dir / f"{video_id}_sam3.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            pipeline_result = {
                "video_id": video_id,
                "pipeline": "sam3",
                "results_path": str(results_file),
                "features": avg_features,
                "num_segmentations": len(segmentations)
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["pipeline_sam3"],
                pipeline_result
            )
            
            print(f"SAM3 pipeline completed for {video_id}")
            
        except Exception as e:
            print(f"Error in SAM3 pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the SAM3 pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to video.preprocessed events
        subject = self.config["nats"]["subjects"]["video_preprocessed"]
        print(f"SAM3 pipeline subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("SAM3 pipeline service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = SAM3Pipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

