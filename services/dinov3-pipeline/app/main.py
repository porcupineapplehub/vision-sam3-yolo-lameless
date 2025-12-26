"""
DINOv3 Embedding Pipeline
Extracts DINOv3 embeddings and stores in VectorDB for similarity search
"""
import asyncio
import os
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import yaml
from shared.utils.nats_client import NATSClient
from typing import List, Dict, Any, Optional
from PIL import Image


class DINOv3Pipeline:
    """DINOv3 embedding extraction and VectorDB storage"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Initialize DINOv3 model
        model_name = self.config.get("models", {}).get("dinov3", {}).get("model_name", "facebook/dinov2-base")
        print(f"Loading DINOv3 model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Initialize Qdrant client
        qdrant_url = os.getenv("QDRANT_URL", self.config.get("qdrant", {}).get("url", "http://localhost:6333"))
        self.qdrant_client = QdrantClient(url=qdrant_url)
        
        # Ensure collection exists
        collection_name = self.config.get("qdrant", {}).get("collection_name", "cow_embeddings")
        self.collection_name = collection_name
        self._ensure_collection()
        
        # Directories
        self.processed_dir = Path("/app/data/processed")
        self.results_dir = Path("/app/data/results/dinov3")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self):
        """Load configuration"""
        import os
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {
            "qdrant": {
                "url": os.getenv("QDRANT_URL", "http://localhost:6333"),
                "collection_name": "cow_embeddings"
            },
            "models": {
                "dinov3": {
                    "model_name": "facebook/dinov2-base"
                }
            }
        }
    
    def _ensure_collection(self):
        """Ensure Qdrant collection exists"""
        try:
            # Get embedding dimension from model
            with torch.no_grad():
                dummy_input = self.processor(images=Image.new('RGB', (224, 224)), return_tensors="pt")
                dummy_output = self.model(**dummy_input.to(self.device))
                embedding_dim = dummy_output.last_hidden_state.shape[-1]
            
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                # Create collection
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
                )
                print(f"Created Qdrant collection: {self.collection_name}")
            else:
                print(f"Qdrant collection exists: {self.collection_name}")
        except Exception as e:
            print(f"Error ensuring collection: {e}")
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract DINOv3 embedding from image"""
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token or average pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        return embedding
    
    def extract_video_embeddings(self, video_path: Path) -> Dict[str, Any]:
        """Extract embeddings from video frames"""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception(f"Failed to open video: {video_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        embeddings = []
        frame_interval = max(1, fps)  # Extract 1 frame per second
        frame_count = 0
        
        # Extract canonical frames (mid-stance, mid-swing)
        canonical_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                embedding = self.extract_embedding(frame)
                embeddings.append({
                    "frame": frame_count,
                    "time": frame_count / fps if fps > 0 else 0,
                    "embedding": embedding.tolist()
                })
            
            frame_count += 1
        
        cap.release()
        
        # Extract canonical frames (first, middle, last)
        if embeddings:
            canonical_frames = [
                embeddings[0],
                embeddings[len(embeddings) // 2],
                embeddings[-1]
            ]
        
        return {
            "embeddings": embeddings,
            "canonical_frames": canonical_frames,
            "total_frames": total_frames,
            "fps": fps
        }
    
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar embeddings in VectorDB"""
        try:
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=top_k
            )
            
            similar_cases = []
            for result in results:
                similar_cases.append({
                    "video_id": result.payload.get("video_id", "unknown"),
                    "score": float(result.score),
                    "label": result.payload.get("label", None),
                    "metadata": result.payload.get("metadata", {})
                })
            
            return similar_cases
        except Exception as e:
            print(f"Error searching similar: {e}")
            return []
    
    async def process_video(self, video_data: dict):
        """Process a preprocessed video"""
        video_id = video_data["video_id"]
        processed_path = Path(video_data["processed_path"])
        
        print(f"DINOv3 pipeline processing video {video_id}")
        
        if not processed_path.exists():
            print(f"Processed video not found: {processed_path}")
            return
        
        try:
            # Extract embeddings
            embedding_data = self.extract_video_embeddings(processed_path)
            
            # Get average embedding for main representation
            if embedding_data["embeddings"]:
                avg_embedding = np.mean(
                    [np.array(e["embedding"]) for e in embedding_data["embeddings"]],
                    axis=0
                )
            else:
                print(f"No embeddings extracted for {video_id}")
                return
            
            # Search for similar cases
            similar_cases = self.search_similar(avg_embedding, top_k=5)
            
            # Compute neighbor-based evidence
            if similar_cases:
                labels = [case["label"] for case in similar_cases if case["label"] is not None]
                if labels:
                    lame_count = sum(1 for label in labels if label == 1)
                    neighbor_evidence = lame_count / len(labels)
                else:
                    neighbor_evidence = 0.5  # Unknown
            else:
                neighbor_evidence = 0.5  # No similar cases
            
            # Store in VectorDB
            try:
                point = PointStruct(
                    id=video_id,  # Use video_id as point ID
                    vector=avg_embedding.tolist(),
                    payload={
                        "video_id": video_id,
                        "filename": video_data.get("filename", ""),
                        "uploaded_at": video_data.get("uploaded_at", ""),
                        "label": None,  # Will be updated when labeled
                        "metadata": video_data.get("metadata", {})
                    }
                )
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=[point]
                )
                print(f"Stored embedding in VectorDB for {video_id}")
            except Exception as e:
                print(f"Error storing in VectorDB: {e}")
            
            # Save results
            results = {
                "video_id": video_id,
                "embedding_dim": len(avg_embedding),
                "num_embeddings": len(embedding_data["embeddings"]),
                "similar_cases": similar_cases,
                "neighbor_evidence": neighbor_evidence,
                "canonical_frames": embedding_data["canonical_frames"]
            }
            
            results_file = self.results_dir / f"{video_id}_dinov3.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            pipeline_result = {
                "video_id": video_id,
                "pipeline": "dinov3",
                "results_path": str(results_file),
                "neighbor_evidence": neighbor_evidence,
                "similar_cases": similar_cases,
                "embedding_dim": len(avg_embedding)
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["pipeline_dinov3"],
                pipeline_result
            )
            
            print(f"DINOv3 pipeline completed for {video_id}")
            
        except Exception as e:
            print(f"Error in DINOv3 pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the DINOv3 pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to video.preprocessed events
        subject = self.config["nats"]["subjects"]["video_preprocessed"]
        print(f"DINOv3 pipeline subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("DINOv3 pipeline service started. Waiting for videos...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    import os
    pipeline = DINOv3Pipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

