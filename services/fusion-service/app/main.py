"""
Fusion Service
Combines predictions from all pipelines to generate final lameness score.

Enhanced with:
- TCN, Transformer, and Graph Transformer predictions
- Human consensus integration
- Rule-based gating and stacking meta-model
- Confidence calibration
- Detailed pipeline comparison report
- Per-cow lameness aggregation and history tracking
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from uuid import UUID
import numpy as np
import yaml
import threading

from fastapi import FastAPI
import uvicorn

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import select
import uuid

from shared.utils.nats_client import NATSClient

# FastAPI app for health checks
health_app = FastAPI()

@health_app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "fusion-service"}

Base = declarative_base()


class CowIdentityDB(Base):
    """Cow identity reference for fusion service"""
    __tablename__ = "cow_identities"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cow_id = Column(String(100), unique=True, nullable=False, index=True)
    tag_number = Column(String(50), nullable=True)
    total_sightings = Column(String, default="0")
    first_seen = Column(DateTime, default=datetime.utcnow)
    last_seen = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)


class LamenessRecordDB(Base):
    """Lameness observation record for a cow"""
    __tablename__ = "lameness_records"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cow_id = Column(PGUUID(as_uuid=True), ForeignKey("cow_identities.id"), nullable=False, index=True)
    video_id = Column(String(100), nullable=False, index=True)
    observation_date = Column(DateTime, default=datetime.utcnow)

    # Pipeline scores
    fusion_score = Column(Float, nullable=True)
    tleap_score = Column(Float, nullable=True)
    tcn_score = Column(Float, nullable=True)
    transformer_score = Column(Float, nullable=True)
    gnn_score = Column(Float, nullable=True)
    graph_transformer_score = Column(Float, nullable=True)
    ml_ensemble_score = Column(Float, nullable=True)

    # Final prediction
    is_lame = Column(Boolean, nullable=True)
    confidence = Column(Float, nullable=True)
    severity_level = Column(String(20), nullable=True)

    # Human validation
    human_validated = Column(Boolean, default=False)
    human_label = Column(Boolean, nullable=True)


class FusionService:
    """
    Enhanced Fusion service combining all pipeline predictions.
    
    Pipelines integrated:
    - ML (XGBoost, CatBoost, LightGBM ensemble)
    - TCN (Temporal Convolutional Network)
    - Transformer (Gait Transformer)
    - GNN (Graph Transformer / GraphGPS)
    - Human consensus (weighted by rater reliability)
    """
    
    # Pipeline weights for weighted average fusion
    # Updated to include Graph Transformer (primary graph model)
    PIPELINE_WEIGHTS = {
        "ml": 0.15,
        "tcn": 0.12,
        "transformer": 0.12,
        "gnn": 0.08,                 # Reduced - GraphGPS as secondary
        "graph_transformer": 0.18,   # New - Primary graph model (Graphormer)
        "human": 0.35                # High weight for human consensus
    }
    
    # Confidence thresholds for gating
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    LOW_CONFIDENCE_THRESHOLD = 0.55
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model storage
        self.models_dir = Path("/app/shared/models/fusion")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load stacking meta-model if available
        self.stacking_model = None
        self._load_stacking_model()
        
        # Directories
        self.results_dir = Path("/app/data/results/fusion")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.cow_results_dir = Path("/app/data/results/cow_predictions")
        self.cow_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for pipeline results
        self.pipeline_results = {}
        
        # Cow ID mapping cache
        self.cow_id_mapping: Dict[str, str] = {}
        
        # Database connection
        self.db_url = os.getenv(
            "POSTGRES_URL",
            self.config.get("database", {}).get("url", "postgresql://lameness_user:lameness_pass@postgres:5432/lameness_db")
        )
        self.async_db_url = self.db_url.replace("postgresql://", "postgresql+asyncpg://")
        self.engine = None
        self.async_session = None
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_stacking_model(self):
        """Load stacking meta-model if available"""
        stacking_file = self.models_dir / "stacking_model.pkl"
        if stacking_file.exists():
            try:
                import pickle
                with open(stacking_file, "rb") as f:
                    self.stacking_model = pickle.load(f)
                print(f"✅ Loaded stacking model: {stacking_file}")
            except Exception as e:
                print(f"⚠️ Failed to load stacking model: {e}")
    
    async def _init_database(self):
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(self.async_db_url, echo=False)
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
            # Create tables if they don't exist
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            print("✅ Database connection initialized")
        except Exception as e:
            print(f"⚠️ Database init failed (will continue without DB): {e}")
            self.engine = None
            self.async_session = None
    
    def load_cow_id_mapping(self) -> Dict[str, str]:
        """Load video_id -> cow_id mapping from tracking results"""
        mapping = {}
        tracking_dir = Path("/app/data/results/tracking")
        
        if not tracking_dir.exists():
            return mapping
        
        for tracking_file in tracking_dir.glob("*_tracking.json"):
            try:
                with open(tracking_file) as f:
                    data = json.load(f)
                
                video_id = data.get("video_id")
                if not video_id:
                    continue
                
                reid_results = data.get("reid_results", [])
                for reid in reid_results:
                    cow_id = reid.get("cow_id")
                    if cow_id:
                        mapping[video_id] = cow_id
                        break
            except Exception:
                continue
        
        self.cow_id_mapping = mapping
        return mapping
    
    def get_cow_for_video(self, video_id: str) -> Optional[str]:
        """Get the cow_id for a given video_id"""
        if not self.cow_id_mapping:
            self.load_cow_id_mapping()
        return self.cow_id_mapping.get(video_id)
    
    def get_videos_for_cow(self, cow_id: str) -> List[str]:
        """Get all video_ids belonging to a specific cow"""
        if not self.cow_id_mapping:
            self.load_cow_id_mapping()
        return [vid for vid, cid in self.cow_id_mapping.items() if cid == cow_id]
    
    async def aggregate_cow_predictions(self, cow_id: str) -> Dict[str, Any]:
        """
        Aggregate all video predictions for a cow into one cow-level prediction.
        
        Uses weighted mean by confidence, prioritizing recent observations.
        """
        videos = self.get_videos_for_cow(cow_id)
        
        if not videos:
            return {
                "cow_id": cow_id,
                "aggregated_score": 0.5,
                "confidence": 0.0,
                "num_videos": 0,
                "prediction": 0,
                "severity_level": "unknown"
            }
        
        # Collect fusion results for each video
        scores = []
        confidences = []
        timestamps = []
        
        for video_id in videos:
            fusion_file = self.results_dir / f"{video_id}_fusion.json"
            if fusion_file.exists():
                try:
                    with open(fusion_file) as f:
                        data = json.load(f)
                    fusion_result = data.get("fusion_result", {})
                    scores.append(fusion_result.get("final_probability", 0.5))
                    confidences.append(fusion_result.get("confidence", 0.5))
                    timestamps.append(fusion_file.stat().st_mtime)
                except Exception:
                    continue
        
        if not scores:
            return {
                "cow_id": cow_id,
                "aggregated_score": 0.5,
                "confidence": 0.0,
                "num_videos": len(videos),
                "prediction": 0,
                "severity_level": "unknown"
            }
        
        # Weight by confidence and recency
        scores = np.array(scores)
        confidences = np.array(confidences)
        timestamps = np.array(timestamps)
        
        # Normalize timestamps to [0, 1] range with recent = higher weight
        if len(timestamps) > 1:
            ts_min, ts_max = timestamps.min(), timestamps.max()
            if ts_max > ts_min:
                recency_weights = (timestamps - ts_min) / (ts_max - ts_min)
            else:
                recency_weights = np.ones_like(timestamps)
        else:
            recency_weights = np.ones_like(timestamps)
        
        # Combined weight: confidence * (0.5 + 0.5 * recency)
        weights = confidences * (0.5 + 0.5 * recency_weights)
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights) / len(weights)
        
        # Weighted average
        aggregated_score = float(np.sum(scores * weights))
        aggregated_confidence = float(np.mean(confidences))
        
        # Determine severity level
        if aggregated_score < 0.3:
            severity_level = "healthy"
        elif aggregated_score < 0.5:
            severity_level = "mild"
        elif aggregated_score < 0.7:
            severity_level = "moderate"
        else:
            severity_level = "severe"
        
        return {
            "cow_id": cow_id,
            "aggregated_score": aggregated_score,
            "confidence": aggregated_confidence,
            "num_videos": len(scores),
            "total_videos": len(videos),
            "prediction": int(aggregated_score > 0.5),
            "severity_level": severity_level,
            "video_ids": videos
        }
    
    async def save_lameness_record(self, video_id: str, cow_id: str, 
                                    fusion_result: Dict, predictions: Dict):
        """Save lameness record to database"""
        if self.async_session is None:
            return
        
        try:
            async with self.async_session() as session:
                # Find or create cow identity
                result = await session.execute(
                    select(CowIdentityDB).where(CowIdentityDB.cow_id == cow_id)
                )
                cow_identity = result.scalar_one_or_none()
                
                if not cow_identity:
                    # Cow should already exist from tracking service
                    print(f"  Cow identity {cow_id} not found in database")
                    return
                
                # Create lameness record
                record = LamenessRecordDB(
                    cow_id=cow_identity.id,
                    video_id=video_id,
                    observation_date=datetime.utcnow(),
                    fusion_score=fusion_result.get("final_probability"),
                    tleap_score=predictions.get("tleap", {}).get("lameness_score"),
                    tcn_score=predictions.get("tcn", {}).get("probability"),
                    transformer_score=predictions.get("transformer", {}).get("probability"),
                    gnn_score=predictions.get("gnn", {}).get("probability"),
                    graph_transformer_score=predictions.get("graph_transformer", {}).get("probability"),
                    ml_ensemble_score=predictions.get("ml", {}).get("probability"),
                    is_lame=fusion_result.get("final_prediction") == 1,
                    confidence=fusion_result.get("confidence"),
                    severity_level=self._get_severity_level(fusion_result.get("final_probability", 0.5))
                )
                session.add(record)
                await session.commit()
                print(f"  ✅ Saved lameness record for cow {cow_id}")
        except Exception as e:
            print(f"  ⚠️ Failed to save lameness record: {e}")
    
    def _get_severity_level(self, score: float) -> str:
        """Convert score to severity level"""
        if score < 0.3:
            return "healthy"
        elif score < 0.5:
            return "mild"
        elif score < 0.7:
            return "moderate"
        else:
            return "severe"
    
    def collect_pipeline_predictions(self, video_id: str) -> Dict[str, Any]:
        """Collect predictions from all pipelines including new DL models"""
        predictions = {}
        
        # ML pipeline predictions (XGBoost, CatBoost, LightGBM)
        ml_file = Path(f"/app/data/results/ml/{video_id}_ml.json")
        if ml_file.exists():
            with open(ml_file) as f:
                ml_data = json.load(f)
                if "predictions" in ml_data:
                    predictions["ml"] = {
                        "probability": ml_data["predictions"].get("ensemble", {}).get("probability", 0.5),
                        "uncertainty": 0.1,  # Default uncertainty
                        "model_predictions": ml_data["predictions"]
                    }
        
        # TCN pipeline predictions
        tcn_file = Path(f"/app/data/results/tcn/{video_id}_tcn.json")
        if tcn_file.exists():
            with open(tcn_file) as f:
                tcn_data = json.load(f)
                predictions["tcn"] = {
                    "probability": tcn_data.get("severity_score", 0.5),
                    "uncertainty": tcn_data.get("uncertainty", 0.1)
                }
        
        # Transformer pipeline predictions
        transformer_file = Path(f"/app/data/results/transformer/{video_id}_transformer.json")
        if transformer_file.exists():
            with open(transformer_file) as f:
                transformer_data = json.load(f)
                predictions["transformer"] = {
                    "probability": transformer_data.get("severity_score", 0.5),
                    "uncertainty": transformer_data.get("uncertainty", 0.1),
                    "temporal_saliency": transformer_data.get("temporal_saliency", [])
                }
        
        # GNN (GraphGPS) pipeline predictions
        gnn_file = Path(f"/app/data/results/gnn/{video_id}_gnn.json")
        if gnn_file.exists():
            with open(gnn_file) as f:
                gnn_data = json.load(f)
                predictions["gnn"] = {
                    "probability": gnn_data.get("severity_score", 0.5),
                    "uncertainty": gnn_data.get("uncertainty", 0.1),
                    "neighbor_influence": gnn_data.get("neighbor_influence", [])
                }

        # Graph Transformer (Graphormer) pipeline predictions
        gt_file = Path(f"/app/data/results/graph_transformer/{video_id}_graph_transformer.json")
        if gt_file.exists():
            with open(gt_file) as f:
                gt_data = json.load(f)
                predictions["graph_transformer"] = {
                    "probability": gt_data.get("graph_prediction", 0.5),
                    "uncertainty": gt_data.get("uncertainty", 0.1),
                    "node_prediction": gt_data.get("node_prediction", 0.5),
                    "attention_info": gt_data.get("attention_info", {})
                }

        # Human consensus (from rater reliability service)
        human_file = Path(f"/app/data/rater_reliability/consensus/{video_id}.json")
        if human_file.exists():
            with open(human_file) as f:
                human_data = json.load(f)
                predictions["human"] = {
                    "probability": human_data.get("probability", 0.5),
                    "confidence": human_data.get("confidence", 0.5),
                    "num_raters": human_data.get("num_raters", 0)
                }
        
        # Also load feature-level data for SHAP
        # YOLO features
        yolo_file = Path(f"/app/data/results/yolo/{video_id}_yolo.json")
        if yolo_file.exists():
            with open(yolo_file) as f:
                yolo_data = json.load(f)
                if "features" in yolo_data:
                    predictions["yolo"] = yolo_data["features"]
        
        # T-LEAP features
        tleap_file = Path(f"/app/data/results/tleap/{video_id}_tleap.json")
        if tleap_file.exists():
            with open(tleap_file) as f:
                tleap_data = json.load(f)
                predictions["tleap"] = tleap_data.get("locomotion_features", {})
        
        return predictions
    
    def apply_gating_rules(self, predictions: Dict[str, Any]) -> Tuple[str, str]:
        """
        Apply rule-based gating to determine fusion strategy.
        
        Returns:
            decision_mode: 'human', 'automated', 'hybrid', 'uncertain'
            explanation: Reason for the decision mode
        """
        human_pred = predictions.get("human", {})
        human_conf = human_pred.get("confidence", 0)
        human_num_raters = human_pred.get("num_raters", 0)
        
        # Collect automated predictions
        auto_preds = []
        for key in ["ml", "tcn", "transformer", "gnn", "graph_transformer"]:
            if key in predictions:
                auto_preds.append(predictions[key].get("probability", 0.5))
        
        if not auto_preds:
            if human_num_raters > 0:
                return "human", "No automated predictions available; using human consensus"
            return "uncertain", "Insufficient data from all sources"
        
        auto_mean = np.mean(auto_preds)
        auto_std = np.std(auto_preds)
        auto_agreement = 1.0 - auto_std  # Higher when models agree
        
        # Rule 1: High human confidence with sufficient raters
        if human_conf >= self.HIGH_CONFIDENCE_THRESHOLD and human_num_raters >= 3:
            return "human", f"High human consensus confidence ({human_conf:.2f}) with {human_num_raters} raters"
        
        # Rule 2: High model agreement with high confidence
        if auto_agreement >= 0.9 and all(
            abs(p - 0.5) > 0.3 for p in auto_preds
        ):
            return "automated", f"Strong model agreement ({auto_agreement:.2f}) with high confidence"
        
        # Rule 3: Model disagreement - request more human labels
        if auto_std > 0.25:
            return "uncertain", f"Model disagreement (std={auto_std:.2f}); more human labels recommended"
        
        # Rule 4: Hybrid approach for moderate cases
        return "hybrid", "Moderate confidence; combining human and automated predictions"
    
    def fuse_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fusion combining all pipelines with gating rules.
        """
        # Apply gating rules
        decision_mode, gate_explanation = self.apply_gating_rules(predictions)
        
        # Collect pipeline probabilities and uncertainties
        pipeline_probs = {}
        pipeline_uncertainties = {}
        
        for key in ["ml", "tcn", "transformer", "gnn", "graph_transformer", "human"]:
            if key in predictions:
                pipeline_probs[key] = predictions[key].get("probability", 0.5)
                pipeline_uncertainties[key] = predictions[key].get("uncertainty",
                    1.0 - predictions[key].get("confidence", 0.5))
        
        # Determine fusion probability based on decision mode
        if decision_mode == "human" and "human" in pipeline_probs:
            fusion_prob = pipeline_probs["human"]
            confidence = predictions["human"].get("confidence", 0.5)
        
        elif decision_mode == "automated":
            # Use stacking model if available
            if self.stacking_model:
                features = [pipeline_probs.get(k, 0.5) for k in ["ml", "tcn", "transformer", "gnn", "graph_transformer"]]
                try:
                    fusion_prob = float(self.stacking_model.predict_proba([features])[0, 1])
                except:
                    fusion_prob = np.mean(list(pipeline_probs.values()))
            else:
                # Weighted average of automated pipelines
                weighted_sum = 0.0
                total_weight = 0.0
                for key in ["ml", "tcn", "transformer", "gnn", "graph_transformer"]:
                    if key in pipeline_probs:
                        weight = self.PIPELINE_WEIGHTS.get(key, 0.1)
                        # Reduce weight for high uncertainty
                        uncertainty = pipeline_uncertainties.get(key, 0.5)
                        adjusted_weight = weight * (1.0 - uncertainty * 0.5)
                        weighted_sum += pipeline_probs[key] * adjusted_weight
                        total_weight += adjusted_weight
                
                fusion_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
            
            # Compute confidence from agreement
            auto_probs = [v for k, v in pipeline_probs.items() if k != "human"]
            confidence = 1.0 - np.std(auto_probs) if auto_probs else 0.5
        
        elif decision_mode == "hybrid":
            # Combine human and automated with configured weights
            weighted_sum = 0.0
            total_weight = 0.0
            
            for key, prob in pipeline_probs.items():
                weight = self.PIPELINE_WEIGHTS.get(key, 0.1)
                uncertainty = pipeline_uncertainties.get(key, 0.5)
                adjusted_weight = weight * (1.0 - uncertainty * 0.5)
                weighted_sum += prob * adjusted_weight
                total_weight += adjusted_weight
            
            fusion_prob = weighted_sum / total_weight if total_weight > 0 else 0.5
            confidence = 1.0 - np.std(list(pipeline_probs.values()))
        
        else:  # uncertain
            fusion_prob = 0.5
            confidence = 0.0
        
        # Compute agreement metrics
        all_probs = list(pipeline_probs.values())
        model_agreement = 1.0 - np.std(all_probs) if all_probs else 0.0
        all_predictions = [int(p > 0.5) for p in all_probs]
        unanimous = len(set(all_predictions)) == 1 if all_predictions else False
        
        # Determine recommendation
        if confidence < 0.3 or decision_mode == "uncertain":
            recommendation = "Request more human labels for this video"
        elif fusion_prob > 0.7:
            recommendation = "High lameness probability - consider veterinary examination"
        elif fusion_prob < 0.3:
            recommendation = "Low lameness probability - monitor routine"
        else:
            recommendation = "Moderate lameness indication - continue observation"
        
        return {
            "final_probability": float(fusion_prob),
            "final_prediction": int(fusion_prob > 0.5),
            "confidence": float(confidence),
            "decision_mode": decision_mode,
            "gate_explanation": gate_explanation,
            "model_agreement": float(model_agreement),
            "unanimous": unanimous,
            "recommendation": recommendation,
            "pipeline_contributions": {
                key: {
                    "probability": float(pipeline_probs.get(key, 0.5)),
                    "uncertainty": float(pipeline_uncertainties.get(key, 0.5)),
                    "prediction": int(pipeline_probs.get(key, 0.5) > 0.5),
                    "weight": self.PIPELINE_WEIGHTS.get(key, 0.1)
                }
                for key in ["ml", "tcn", "transformer", "gnn", "graph_transformer", "human"]
                if key in pipeline_probs
            },
            "pipelines_used": list(pipeline_probs.keys()),
            "tleap_features": predictions.get("tleap", {}),
            "yolo_features": predictions.get("yolo", {})
        }
    
    async def process_video(self, video_data: dict):
        """Process video through fusion service with cow-level aggregation"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"Fusion service processing video {video_id}")
        
        try:
            # Get cow_id for this video
            cow_id = self.get_cow_for_video(video_id)
            
            # Collect predictions from all pipelines
            predictions = self.collect_pipeline_predictions(video_id)
            
            if not predictions:
                print(f"No pipeline predictions found for {video_id}")
                return
            
            # Fuse predictions (video-level)
            fusion_result = self.fuse_predictions(predictions)
            
            # Add cow_id to fusion result
            fusion_result["cow_id"] = cow_id
            
            # Aggregate cow-level prediction if cow_id is known
            cow_prediction = None
            if cow_id:
                # First save the video result so it's included in aggregation
                temp_results = {
                    "video_id": video_id,
                    "cow_id": cow_id,
                    "fusion_result": fusion_result,
                    "pipeline_predictions": predictions,
                    "timestamp": video_data.get("timestamp", "")
                }
                results_file = self.results_dir / f"{video_id}_fusion.json"
                with open(results_file, "w") as f:
                    json.dump(temp_results, f, indent=2)
                
                # Now aggregate all videos for this cow
                cow_prediction = await self.aggregate_cow_predictions(cow_id)
                
                # Save cow-level prediction
                cow_results_file = self.cow_results_dir / f"{cow_id}_prediction.json"
                with open(cow_results_file, "w") as f:
                    json.dump({
                        "cow_id": cow_id,
                        "prediction": cow_prediction,
                        "last_updated": datetime.utcnow().isoformat(),
                        "latest_video": video_id
                    }, f, indent=2)
                
                # Save lameness record to database
                await self.save_lameness_record(video_id, cow_id, fusion_result, predictions)
                
                print(f"  Cow {cow_id}: aggregated score = {cow_prediction['aggregated_score']:.3f} "
                      f"({cow_prediction['num_videos']} videos)")
            
            # Save final results (including cow prediction)
            results = {
                "video_id": video_id,
                "cow_id": cow_id,
                "fusion_result": fusion_result,
                "cow_prediction": cow_prediction,
                "pipeline_predictions": predictions,
                "timestamp": video_data.get("timestamp", "")
            }
            
            results_file = self.results_dir / f"{video_id}_fusion.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish analysis complete event
            analysis_result = {
                "video_id": video_id,
                "cow_id": cow_id,
                "final_probability": fusion_result["final_probability"],
                "final_prediction": fusion_result["final_prediction"],
                "results_path": str(results_file),
                "pipeline_contributions": fusion_result["pipeline_contributions"]
            }
            
            # Add cow-level prediction if available
            if cow_prediction:
                analysis_result["cow_prediction"] = {
                    "aggregated_score": cow_prediction["aggregated_score"],
                    "severity_level": cow_prediction["severity_level"],
                    "num_videos": cow_prediction["num_videos"]
                }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["analysis_complete"],
                analysis_result
            )
            
            # Publish cow-level update if cow_id is known
            if cow_id and cow_prediction:
                await self.nats_client.publish(
                    "cow.prediction.updated",
                    {
                        "cow_id": cow_id,
                        "aggregated_score": cow_prediction["aggregated_score"],
                        "severity_level": cow_prediction["severity_level"],
                        "num_videos": cow_prediction["num_videos"],
                        "latest_video_id": video_id
                    }
                )
            
            print(f"Fusion service completed for {video_id}" + 
                  (f" (cow: {cow_id})" if cow_id else ""))
            
        except Exception as e:
            print(f"Error in fusion service for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the fusion service"""
        # Initialize database connection
        await self._init_database()
        
        # Load cow_id mapping
        self.load_cow_id_mapping()
        print(f"Loaded {len(self.cow_id_mapping)} video->cow mappings")
        
        await self.nats_client.connect()
        
        # Subscribe to ML pipeline results (last in sequence)
        subject = self.config["nats"]["subjects"]["pipeline_ml"]
        print(f"Fusion service subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("=" * 60)
        print("Fusion Service Started")
        print("=" * 60)
        print(f"Features:")
        print(f"  - Video-level fusion from 6 pipelines")
        print(f"  - Cow-level aggregation enabled")
        print(f"  - Database recording: {'enabled' if self.async_session else 'disabled'}")
        print("=" * 60)
        await asyncio.Event().wait()


def run_health_server():
    """Run health server in a separate thread"""
    uvicorn.run(health_app, host="0.0.0.0", port=8006, log_level="warning")


async def main():
    """Main entry point"""
    # Start health server in background thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print("Health server started on port 8006")

    service = FusionService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

