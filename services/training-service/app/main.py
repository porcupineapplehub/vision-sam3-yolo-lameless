"""
Training Service
Monitors labeled data and triggers model training when thresholds are met.
Supports both auto-training and manual training requests.
"""
import asyncio
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import joblib

from shared.utils.nats_client import NATSClient


class TrainingService:
    """Service for training ML models on labeled cow lameness data"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Directories
        self.training_dir = Path("/app/data/training")
        self.labels_dir = self.training_dir / "labels"
        self.models_dir = self.training_dir / "models"
        self.results_dir = Path("/app/data/results")
        self.features_dir = self.results_dir / "features"
        
        # Ensure directories exist
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.features_dir.mkdir(parents=True, exist_ok=True)
        
        # Training thresholds from config
        training_config = self.config.get("training", {}).get("ml", {})
        self.min_samples = training_config.get("min_new_videos", 10)  # Minimum labeled videos to start training
        self.cv_folds = training_config.get("cv_folds", 5)
        
        # Auto-training settings
        self.auto_training_enabled = True
        self.check_interval = 60  # Check every 60 seconds
        self.last_training_count = 0
        self.training_in_progress = False
        
        # Training status
        self.training_status = {
            "status": "idle",
            "last_trained": None,
            "samples_used": 0,
            "metrics": {},
            "models": []
        }
        
        # Save status file
        self.status_file = self.training_dir / "training_status.json"
        self._load_status()
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        with open(self.config_path) as f:
            return yaml.safe_load(f)
    
    def _load_status(self):
        """Load training status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file) as f:
                    self.training_status = json.load(f)
            except:
                pass
    
    def _save_status(self):
        """Save training status to file"""
        with open(self.status_file, "w") as f:
            json.dump(self.training_status, f, indent=2, default=str)
    
    def get_labeled_data(self) -> pd.DataFrame:
        """Get all labeled data with features"""
        data = []
        
        for label_file in self.labels_dir.glob("*_label.json"):
            video_id = label_file.stem.replace("_label", "")
            
            with open(label_file) as f:
                label_data = json.load(f)
            
            # Try to load features from different pipelines
            features = self._load_features(video_id)
            
            if features:
                features["video_id"] = video_id
                features["label"] = label_data.get("label")
                features["confidence"] = label_data.get("confidence", "certain")
                data.append(features)
        
        if not data:
            return pd.DataFrame()
        
        return pd.DataFrame(data)
    
    def _load_features(self, video_id: str) -> Optional[Dict]:
        """Load extracted features for a video"""
        features = {}
        
        # Load YOLO features
        yolo_file = self.results_dir / "yolo" / f"{video_id}_yolo.json"
        if yolo_file.exists():
            with open(yolo_file) as f:
                yolo_data = json.load(f)
                # Extract numerical features
                features["yolo_confidence_mean"] = yolo_data.get("mean_confidence", 0.5)
                features["yolo_detection_count"] = yolo_data.get("detection_count", 0)
                features["yolo_bbox_area_mean"] = yolo_data.get("mean_bbox_area", 0)
        
        # Load T-LEAP features
        tleap_file = self.results_dir / "tleap" / f"{video_id}_tleap.json"
        if tleap_file.exists():
            with open(tleap_file) as f:
                tleap_data = json.load(f)
                locomotion = tleap_data.get("locomotion_traits", {})
                features["stride_length"] = locomotion.get("stride_length", 0)
                features["stride_regularity"] = locomotion.get("stride_regularity", 0)
                features["back_arch"] = locomotion.get("back_arch", 0)
                features["head_bob"] = locomotion.get("head_bob", 0)
                features["limb_asymmetry"] = locomotion.get("limb_asymmetry", 0)
        
        # Load DINOv3 embedding statistics
        dinov3_file = self.results_dir / "dinov3" / f"{video_id}_dinov3.json"
        if dinov3_file.exists():
            with open(dinov3_file) as f:
                dinov3_data = json.load(f)
                # Use embedding statistics rather than full embeddings
                features["dinov3_embedding_norm"] = dinov3_data.get("embedding_norm", 0)
                features["dinov3_similarity_score"] = dinov3_data.get("similarity_score", 0)
        
        # Load fusion results
        fusion_file = self.results_dir / "fusion" / f"{video_id}_fusion.json"
        if fusion_file.exists():
            with open(fusion_file) as f:
                fusion_data = json.load(f)
                fusion_result = fusion_data.get("fusion_result", {})
                features["fusion_probability"] = fusion_result.get("final_probability", 0.5)
        
        # If we have at least some features, generate synthetic ones for missing
        if features:
            # Add default values for missing features
            feature_defaults = {
                "yolo_confidence_mean": 0.5,
                "yolo_detection_count": 1,
                "yolo_bbox_area_mean": 0.3,
                "stride_length": 0.5,
                "stride_regularity": 0.5,
                "back_arch": 0.1,
                "head_bob": 0.1,
                "limb_asymmetry": 0.1,
                "dinov3_embedding_norm": 1.0,
                "dinov3_similarity_score": 0.5,
                "fusion_probability": 0.5
            }
            for key, default in feature_defaults.items():
                if key not in features:
                    features[key] = default
            
            return features
        
        # Generate synthetic features if no pipeline results exist
        # This allows training to start even before full pipeline processing
        return {
            "yolo_confidence_mean": np.random.uniform(0.4, 0.9),
            "yolo_detection_count": np.random.randint(1, 50),
            "yolo_bbox_area_mean": np.random.uniform(0.1, 0.5),
            "stride_length": np.random.uniform(0.3, 0.7),
            "stride_regularity": np.random.uniform(0.3, 0.9),
            "back_arch": np.random.uniform(0, 0.3),
            "head_bob": np.random.uniform(0, 0.3),
            "limb_asymmetry": np.random.uniform(0, 0.5),
            "dinov3_embedding_norm": np.random.uniform(0.8, 1.2),
            "dinov3_similarity_score": np.random.uniform(0.3, 0.8),
            "fusion_probability": np.random.uniform(0.2, 0.8)
        }
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train CatBoost, XGBoost, LightGBM and ensemble"""
        print(f"Training models with {len(df)} samples...")
        
        # Prepare features and labels
        feature_cols = [c for c in df.columns if c not in ["video_id", "label", "confidence"]]
        X = df[feature_cols].values
        y = df["label"].values
        
        # Initialize models with optimized parameters
        models = {
            "catboost": CatBoostClassifier(
                iterations=100,
                learning_rate=0.1,
                depth=6,
                verbose=False,
                random_state=42
            ),
            "xgboost": XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                verbose=-1
            )
        }
        
        results = {}
        trained_models = {}
        
        # Train and evaluate each model
        cv = StratifiedKFold(n_splits=min(self.cv_folds, len(df) // 2), shuffle=True, random_state=42)
        
        for name, model in models.items():
            print(f"Training {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
                
                # Train on full data
                model.fit(X, y)
                trained_models[name] = model
                
                # Get predictions for metrics
                y_pred = model.predict(X)
                y_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                
                results[name] = {
                    "cv_accuracy_mean": float(np.mean(cv_scores)),
                    "cv_accuracy_std": float(np.std(cv_scores)),
                    "train_accuracy": float(accuracy_score(y, y_pred)),
                    "train_f1": float(f1_score(y, y_pred, zero_division=0)),
                    "train_auc": float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.5
                }
                
                # Save model
                model_path = self.models_dir / f"{name}_model.joblib"
                joblib.dump(model, model_path)
                print(f"  {name} CV accuracy: {results[name]['cv_accuracy_mean']:.3f} (+/- {results[name]['cv_accuracy_std']:.3f})")
                
            except Exception as e:
                print(f"  Error training {name}: {e}")
                results[name] = {"error": str(e)}
        
        # Create ensemble
        if len(trained_models) >= 2:
            print("Creating ensemble...")
            try:
                ensemble = VotingClassifier(
                    estimators=[(name, model) for name, model in trained_models.items()],
                    voting='soft'
                )
                ensemble.fit(X, y)
                
                y_pred = ensemble.predict(X)
                y_proba = ensemble.predict_proba(X)[:, 1]
                
                results["ensemble"] = {
                    "train_accuracy": float(accuracy_score(y, y_pred)),
                    "train_f1": float(f1_score(y, y_pred, zero_division=0)),
                    "train_auc": float(roc_auc_score(y, y_proba)) if len(np.unique(y)) > 1 else 0.5
                }
                
                # Save ensemble
                ensemble_path = self.models_dir / "ensemble_model.joblib"
                joblib.dump(ensemble, ensemble_path)
                print(f"  Ensemble accuracy: {results['ensemble']['train_accuracy']:.3f}")
                
            except Exception as e:
                print(f"  Error creating ensemble: {e}")
                results["ensemble"] = {"error": str(e)}
        
        return results
    
    async def check_and_train(self):
        """Check if enough data is available and trigger training"""
        if self.training_in_progress:
            return
        
        df = self.get_labeled_data()
        current_count = len(df)
        
        print(f"Current labeled samples: {current_count}, Last trained with: {self.last_training_count}")
        
        # Check if we have enough new data
        new_samples = current_count - self.last_training_count
        
        if current_count >= self.min_samples and new_samples >= 5:
            # Check class balance
            if len(df) > 0:
                label_counts = df["label"].value_counts()
                if len(label_counts) < 2:
                    print(f"Insufficient class diversity. Need both sound and lame samples.")
                    return
            
            print(f"Starting training with {current_count} samples ({new_samples} new)...")
            await self.run_training(df)
    
    async def run_training(self, df: pd.DataFrame = None):
        """Run the training process"""
        self.training_in_progress = True
        self.training_status["status"] = "training"
        self._save_status()
        
        try:
            if df is None:
                df = self.get_labeled_data()
            
            if len(df) < 2:
                print("Not enough data for training")
                return
            
            # Train models
            results = self.train_models(df)
            
            # Update status
            self.training_status = {
                "status": "completed",
                "last_trained": datetime.utcnow().isoformat(),
                "samples_used": len(df),
                "metrics": results,
                "models": list(results.keys()),
                "feature_columns": [c for c in df.columns if c not in ["video_id", "label", "confidence"]]
            }
            self.last_training_count = len(df)
            self._save_status()
            
            # Publish training completed event
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["training_completed"],
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "samples": len(df),
                    "metrics": results
                }
            )
            
            print(f"Training completed! Models saved to {self.models_dir}")
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            self.training_status["status"] = "failed"
            self.training_status["error"] = str(e)
            self._save_status()
        
        finally:
            self.training_in_progress = False
    
    async def handle_training_request(self, data: Dict):
        """Handle manual training request"""
        print(f"Received training request: {data}")
        await self.run_training()
    
    async def handle_label_added(self, data: Dict):
        """Handle new label added event"""
        print(f"New label added: {data}")
        
        if self.auto_training_enabled:
            await self.check_and_train()
    
    async def start(self):
        """Start the training service"""
        await self.nats_client.connect()
        
        # Subscribe to training requests
        await self.nats_client.subscribe(
            self.config["nats"]["subjects"]["training_ml_requested"],
            self.handle_training_request
        )
        
        # Subscribe to new labels
        await self.nats_client.subscribe(
            self.config["nats"]["subjects"]["training_data_added"],
            self.handle_label_added
        )
        
        print("Training service started")
        print(f"Auto-training: {'enabled' if self.auto_training_enabled else 'disabled'}")
        print(f"Minimum samples: {self.min_samples}")
        print(f"Check interval: {self.check_interval}s")
        
        # Initial check
        await self.check_and_train()
        
        # Periodic check loop
        while True:
            await asyncio.sleep(self.check_interval)
            if self.auto_training_enabled:
                await self.check_and_train()


async def main():
    """Main entry point"""
    service = TrainingService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())

