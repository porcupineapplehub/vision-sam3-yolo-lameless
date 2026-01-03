"""
Cow Tracking Service

Multi-object tracking service for cow lameness detection system.

Features:
- ByteTrack for within-video cow tracking
- Re-ID for cross-video cow identification using DINOv3 embeddings
- Longitudinal lameness tracking per identified cow
- Integration with YOLO detections and DINOv3 embeddings

NATS Subscriptions:
- pipeline.yolo: Process YOLO detections for tracking
- pipeline.dinov3: Associate tracks with embeddings for Re-ID

NATS Publications:
- tracking.complete: Track results for a video
- tracking.reid.match: Re-ID match results
- tracking.lameness.update: Lameness record updates
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID
import numpy as np
import yaml
from datetime import datetime

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from shared.utils.nats_client import NATSClient
from app.tracker import ByteTracker, Detection, Track
from app.reid import CowReIDMatcher, ReIDMatch
from app.database import Base, CowIdentityDB, TrackHistoryDB, LamenessRecordDB


class TrackingService:
    """
    Main tracking service orchestrating ByteTrack and Re-ID.
    """

    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))

        # ByteTracker per video (keyed by video_id)
        self.trackers: Dict[str, ByteTracker] = {}

        # Re-ID matcher
        qdrant_url = os.getenv("QDRANT_URL", self.config.get("qdrant", {}).get("url", "http://qdrant:6333"))
        self.reid_matcher = CowReIDMatcher(
            qdrant_url=qdrant_url,
            embedding_dim=768,  # DINOv3 base dimension
            auto_create_identities=True,
            embedding_momentum=0.9
        )

        # Database
        self.db_url = os.getenv(
            "POSTGRES_URL",
            self.config.get("database", {}).get("url", "postgresql://lameness_user:lameness_pass@postgres:5432/lameness_db")
        )
        # Convert to async URL
        self.async_db_url = self.db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.engine = None
        self.async_session = None

        # Results directory
        self.results_dir = Path("/app/data/results/tracking")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cache for pending embeddings (waiting for DINOv3)
        self.pending_tracks: Dict[str, List[dict]] = {}

        # Cache for video DINOv3 embeddings
        self.video_embeddings: Dict[str, np.ndarray] = {}

    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    async def _init_database(self):
        """Initialize database connection and create tables"""
        self.engine = create_async_engine(self.async_db_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        # Create tables
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        print("Database tables initialized")

    def _get_or_create_tracker(self, video_id: str) -> ByteTracker:
        """Get existing tracker or create new one for video"""
        if video_id not in self.trackers:
            self.trackers[video_id] = ByteTracker(
                high_thresh=0.6,
                low_thresh=0.1,
                match_thresh=0.8,
                track_buffer=30,
                use_appearance=True,
                appearance_weight=0.5
            )
        return self.trackers[video_id]

    async def process_yolo_results(self, message: dict):
        """
        Process YOLO detection results for tracking.

        Expected message format:
        {
            "video_id": "...",
            "results_path": "...",
            "detections": [
                {"frame": 0, "bbox": [x1,y1,x2,y2], "confidence": 0.9, ...},
                ...
            ]
        }
        """
        video_id = message.get("video_id")
        if not video_id:
            return

        print(f"Tracking service processing YOLO results for {video_id}")

        try:
            # Load full YOLO results if path provided
            results_path = message.get("results_path")
            if results_path:
                results_file = Path(results_path)
                if results_file.exists():
                    with open(results_file) as f:
                        yolo_data = json.load(f)
                else:
                    print(f"  YOLO results file not found: {results_path}")
                    return
            else:
                yolo_data = message

            # Get or create tracker for this video
            tracker = self._get_or_create_tracker(video_id)
            tracker.reset()

            # Extract detections from YOLO format
            detections_by_frame = self._parse_yolo_detections(yolo_data)

            if not detections_by_frame:
                print(f"  No detections found in YOLO results")
                return

            # Process each frame
            all_tracks = []
            for frame_idx in sorted(detections_by_frame.keys()):
                frame_dets = detections_by_frame[frame_idx]

                # Create Detection objects
                detections = [
                    Detection(
                        bbox=np.array(d["bbox"]),
                        confidence=d["confidence"],
                        class_id=d.get("class_id", 0),
                        embedding=None  # Will be added from DINOv3
                    )
                    for d in frame_dets
                ]

                # Update tracker
                active_tracks = tracker.update(detections, frame_idx)

                # Store tracks for this frame
                for track in active_tracks:
                    all_tracks.append({
                        "frame": frame_idx,
                        "track_id": track.track_id,
                        "bbox": track.bbox.tolist(),
                        "confidence": track.confidence,
                        "state": track.state.name
                    })

            # Get final track summaries
            track_summaries = []
            for track in tracker.track_manager.tracks:
                if track.hits >= 3:  # Only include tracks with sufficient detections
                    track_summaries.append({
                        "track_id": track.track_id,
                        "start_frame": track.frame_history[0] if track.frame_history else 0,
                        "end_frame": track.frame_history[-1] if track.frame_history else 0,
                        "total_frames": len(track.frame_history),
                        "avg_confidence": sum([b[4] if len(b) > 4 else track.confidence
                                              for b in track.bbox_history]) / max(1, len(track.bbox_history))
                    })

            # Store pending tracks for Re-ID (waiting for embeddings)
            self.pending_tracks[video_id] = track_summaries

            # Save tracking results
            results = {
                "video_id": video_id,
                "pipeline": "tracking",
                "total_tracks": len(track_summaries),
                "track_summaries": track_summaries,
                "frame_tracks": all_tracks,
                "statistics": tracker.get_statistics()
            }

            results_file = self.results_dir / f"{video_id}_tracking.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

            # Publish tracking complete
            await self.nats_client.publish(
                "tracking.complete",
                {
                    "video_id": video_id,
                    "results_path": str(results_file),
                    "total_tracks": len(track_summaries),
                    "pending_reid": True
                }
            )

            print(f"  Tracking complete: {len(track_summaries)} tracks detected")

        except Exception as e:
            print(f"  Error in tracking: {e}")
            import traceback
            traceback.print_exc()

    def _parse_yolo_detections(self, yolo_data: dict) -> Dict[int, List[dict]]:
        """Parse YOLO results into detections by frame"""
        detections_by_frame = {}

        # Handle different YOLO result formats
        if "detections" in yolo_data:
            for item in yolo_data["detections"]:
                frame = item.get("frame", 0)
                if frame not in detections_by_frame:
                    detections_by_frame[frame] = []
                
                # Check if this is a frame container with nested detections
                if "detections" in item and isinstance(item["detections"], list):
                    # Format: {"frame": 0, "detections": [{"bbox": [...], ...}]}
                    detections_by_frame[frame].extend(item["detections"])
                elif "bbox" in item:
                    # Flat format: {"frame": 0, "bbox": [...], ...}
                    detections_by_frame[frame].append(item)

        elif "frames" in yolo_data:
            # Nested by frame
            for frame_data in yolo_data["frames"]:
                frame = frame_data.get("frame_number", 0)
                detections_by_frame[frame] = frame_data.get("detections", [])

        elif "frame_results" in yolo_data:
            # Alternative format
            for frame_num, frame_dets in yolo_data["frame_results"].items():
                detections_by_frame[int(frame_num)] = frame_dets

        return detections_by_frame

    async def process_dinov3_results(self, message: dict):
        """
        Process DINOv3 embeddings for Re-ID association.

        Associates tracks with cow identities using embeddings.
        """
        video_id = message.get("video_id")
        if not video_id:
            return

        print(f"Tracking service processing DINOv3 results for {video_id}")

        try:
            # Load embedding from results
            results_path = message.get("results_path")
            embedding = None

            if results_path:
                results_file = Path(results_path)
                if results_file.exists():
                    with open(results_file) as f:
                        dinov3_data = json.load(f)
                    
                    # Try different embedding formats
                    if "embedding" in dinov3_data:
                        embedding = np.array(dinov3_data["embedding"])
                    elif "canonical_frames" in dinov3_data and dinov3_data["canonical_frames"]:
                        # Average embeddings from canonical frames
                        frame_embeddings = [
                            np.array(frame["embedding"]) 
                            for frame in dinov3_data["canonical_frames"]
                            if "embedding" in frame
                        ]
                        if frame_embeddings:
                            embedding = np.mean(frame_embeddings, axis=0)
                    elif "video_embedding" in dinov3_data:
                        embedding = np.array(dinov3_data["video_embedding"])

            if embedding is None or len(embedding) == 0:
                print(f"  No embedding found for {video_id}")
                return

            # Store embedding
            self.video_embeddings[video_id] = embedding

            # Check if we have pending tracks for this video
            if video_id in self.pending_tracks:
                await self._perform_reid(video_id, embedding)

        except Exception as e:
            print(f"  Error processing DINOv3 for Re-ID: {e}")
            import traceback
            traceback.print_exc()

    async def _perform_reid(self, video_id: str, embedding: np.ndarray):
        """Perform Re-ID for tracks in a video"""
        pending = self.pending_tracks.get(video_id, [])
        if not pending:
            return

        print(f"  Performing Re-ID for {len(pending)} tracks")

        reid_results = []

        for track in pending:
            # For simplicity, use video embedding for all tracks
            # In production, you'd extract per-track embeddings
            match = self.reid_matcher.match_or_create(
                embedding=embedding,
                video_id=video_id,
                track_id=track["track_id"],
                metadata={
                    "start_frame": track["start_frame"],
                    "end_frame": track["end_frame"]
                }
            )

            reid_results.append({
                "track_id": track["track_id"],
                "cow_id": match.cow_id,
                "identity_id": str(match.identity_id),
                "similarity": match.similarity,
                "confidence": match.confidence,
                "is_new": match.is_new_identity
            })

            # Save to database
            await self._save_track_to_db(video_id, track, match)

        # Update tracking results with Re-ID
        results_file = self.results_dir / f"{video_id}_tracking.json"
        if results_file.exists():
            with open(results_file) as f:
                results = json.load(f)
            results["reid_results"] = reid_results
            results["reid_complete"] = True
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)

        # Publish Re-ID results
        await self.nats_client.publish(
            "tracking.reid.match",
            {
                "video_id": video_id,
                "matches": reid_results,
                "new_identities": sum(1 for r in reid_results if r["is_new"])
            }
        )

        print(f"  Re-ID complete: {sum(1 for r in reid_results if not r['is_new'])} matched, "
              f"{sum(1 for r in reid_results if r['is_new'])} new")

        # Clean up
        del self.pending_tracks[video_id]

    async def _save_track_to_db(self, video_id: str, track: dict, match: ReIDMatch):
        """Save track and Re-ID result to database"""
        if self.async_session is None:
            return

        async with self.async_session() as session:
            # Create or update cow identity
            cow_identity = await self._get_or_create_cow_identity(session, match)

            # Create track history record
            track_record = TrackHistoryDB(
                video_id=video_id,
                track_id=track["track_id"],
                cow_id=cow_identity.id if cow_identity else None,
                reid_confidence=match.similarity,
                start_frame=track.get("start_frame"),
                end_frame=track.get("end_frame"),
                total_frames=track.get("total_frames"),
                avg_confidence=track.get("avg_confidence")
            )
            session.add(track_record)
            await session.commit()

    async def _get_or_create_cow_identity(self, session: AsyncSession, match: ReIDMatch) -> Optional[CowIdentityDB]:
        """Get or create cow identity in database"""
        from sqlalchemy import select

        # Try to find existing
        result = await session.execute(
            select(CowIdentityDB).where(CowIdentityDB.cow_id == match.cow_id)
        )
        identity = result.scalar_one_or_none()

        if identity:
            # Update last seen and sightings
            identity.last_seen = datetime.utcnow()
            identity.total_sightings += 1
        else:
            # Create new
            identity = CowIdentityDB(
                id=match.identity_id,
                cow_id=match.cow_id,
                total_sightings=1
            )
            session.add(identity)

        await session.commit()
        return identity

    async def start(self):
        """Start the tracking service"""
        # Initialize database
        await self._init_database()

        # Connect to NATS
        await self.nats_client.connect()

        # Connect to Qdrant for Re-ID
        await self.reid_matcher.connect()

        # Subscribe to YOLO results
        yolo_subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_yolo", "pipeline.yolo"
        )
        await self.nats_client.subscribe(yolo_subject, self.process_yolo_results)

        # Subscribe to DINOv3 results for Re-ID
        dinov3_subject = self.config.get("nats", {}).get("subjects", {}).get(
            "pipeline_dinov3", "pipeline.dinov3"
        )
        await self.nats_client.subscribe(dinov3_subject, self.process_dinov3_results)

        print("=" * 60)
        print("Cow Tracking Service Started")
        print("=" * 60)
        print(f"Subscriptions:")
        print(f"  - {yolo_subject} (YOLO detections)")
        print(f"  - {dinov3_subject} (DINOv3 embeddings)")
        print(f"Publications:")
        print(f"  - tracking.complete")
        print(f"  - tracking.reid.match")
        print(f"Re-ID Statistics: {self.reid_matcher.get_statistics()}")
        print("=" * 60)

        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = TrackingService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
