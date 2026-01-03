"""
ML Pipeline Service
Uses CatBoost, XGBoost, LightGBM, and Ensemble for lameness prediction
"""
import asyncio
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import yaml
from shared.utils.nats_client import NATSClient

# ML libraries
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class MLPipeline:
    """ML pipeline with multiple models and ensemble"""
    
    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # Model storage
        self.models_dir = Path("/app/shared/models/ml")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Load trained models
        self.catboost_model = None
        self.xgboost_model = None
        self.lightgbm_model = None
        self.ensemble_weights = None
        self.scaler = StandardScaler()
        self.feature_names = []
        
        self._load_models()
        
        # Directories
        self.results_dir = Path("/app/data/results/ml")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache for pipeline results
        self.pipeline_results_cache = {}
    
    def _load_config(self):
        """Load configuration"""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_models(self):
        """Load trained models if available"""
        # Try to load CatBoost
        catboost_file = self.models_dir / "catboost_latest.cbm"
        if catboost_file.exists():
            try:
                self.catboost_model = CatBoostClassifier()
                self.catboost_model.load_model(str(catboost_file))
                print(f"Loaded CatBoost model: {catboost_file}")
            except Exception as e:
                print(f"Failed to load CatBoost model: {e}")
        
        # Try to load XGBoost
        xgboost_file = self.models_dir / "xgboost_latest.json"
        if xgboost_file.exists():
            try:
                self.xgboost_model = xgb.Booster()
                self.xgboost_model.load_model(str(xgboost_file))
                print(f"Loaded XGBoost model: {xgboost_file}")
            except Exception as e:
                print(f"Failed to load XGBoost model: {e}")
        
        # Try to load LightGBM
        lightgbm_file = self.models_dir / "lightgbm_latest.txt"
        if lightgbm_file.exists():
            try:
                self.lightgbm_model = lgb.Booster(model_file=str(lightgbm_file))
                print(f"Loaded LightGBM model: {lightgbm_file}")
            except Exception as e:
                print(f"Failed to load LightGBM model: {e}")
        
        # Load ensemble weights if available
        weights_file = self.models_dir / "ensemble_weights.json"
        if weights_file.exists():
            with open(weights_file) as f:
                self.ensemble_weights = json.load(f)
                print(f"Loaded ensemble weights: {self.ensemble_weights}")
        else:
            # Default equal weights
            self.ensemble_weights = {"catboost": 0.33, "xgboost": 0.33, "lightgbm": 0.34}
        
        if not any([self.catboost_model, self.xgboost_model, self.lightgbm_model]):
            print("Warning: No trained models found. Predictions will use default values.")
    
    async def get_pipeline_results(self, video_id: str) -> Dict[str, Any]:
        """Get results from all pipelines for feature extraction"""
        if video_id in self.pipeline_results_cache:
            return self.pipeline_results_cache[video_id]
        
        results = {
            "yolo": None,
            "sam3": None,
            "dinov3": None,
            "tleap": None
        }
        
        # Load from result files
        result_dirs = {
            "yolo": Path("/app/data/results/yolo"),
            "sam3": Path("/app/data/results/sam3"),
            "dinov3": Path("/app/data/results/dinov3"),
            "tleap": Path("/app/data/results/tleap")
        }
        
        for pipeline, result_dir in result_dirs.items():
            result_file = result_dir / f"{video_id}_{pipeline}.json"
            if result_file.exists():
                try:
                    with open(result_file) as f:
                        results[pipeline] = json.load(f)
                except Exception as e:
                    print(f"Error loading {pipeline} results: {e}")
        
        self.pipeline_results_cache[video_id] = results
        return results
    
    def extract_features(self, pipeline_results: Dict[str, Any]) -> np.ndarray:
        """Extract features from all pipeline results"""
        features = []
        feature_names = []
        
        # YOLO features
        if pipeline_results.get("yolo") and "features" in pipeline_results["yolo"]:
            yolo_features = pipeline_results["yolo"]["features"]
            features.extend([
                yolo_features.get("avg_confidence", 0),
                yolo_features.get("position_stability", 0),
                yolo_features.get("avg_box_area", 0),
                yolo_features.get("detection_rate", 0)
            ])
            feature_names.extend(["yolo_conf", "yolo_stability", "yolo_area", "yolo_rate"])
        
        # SAM3 features
        if pipeline_results.get("sam3") and "features" in pipeline_results["sam3"]:
            sam3_features = pipeline_results["sam3"]["features"]
            features.extend([
                sam3_features.get("avg_area_ratio", 0),
                sam3_features.get("avg_circularity", 0),
                sam3_features.get("avg_aspect_ratio", 0)
            ])
            feature_names.extend(["sam3_area_ratio", "sam3_circularity", "sam3_aspect"])
        
        # DINOv3 features
        if pipeline_results.get("dinov3"):
            dinov3_data = pipeline_results["dinov3"]
            features.extend([
                dinov3_data.get("neighbor_evidence", 0.5),
                len(dinov3_data.get("similar_cases", []))
            ])
            feature_names.extend(["dinov3_neighbor_evidence", "dinov3_similar_count"])
        
        # T-LEAP features (if available)
        #
        # Note: the T-LEAP pipeline currently writes `locomotion_features` in the JSON,
        # but older versions used `locomotion_traits`. Support both.
        if pipeline_results.get("tleap"):
            tleap_data = pipeline_results["tleap"] or {}
            locomotion = (
                tleap_data.get("locomotion_traits")
                or tleap_data.get("locomotion_features")
                or {}
            )

            # Preferred (legacy) keys if present
            if any(k in locomotion for k in ["avg_stride_length", "avg_head_bob", "asymmetry_score"]):
                features.extend([
                    locomotion.get("avg_stride_length", 0),
                    locomotion.get("avg_head_bob", 0),
                    locomotion.get("asymmetry_score", 0)
                ])
            else:
                # Derive a compact 3-feature summary from current locomotion_features keys
                stride_means = [
                    locomotion.get("stride_fl_mean"),
                    locomotion.get("stride_fr_mean"),
                    locomotion.get("stride_rl_mean"),
                    locomotion.get("stride_rr_mean"),
                ]
                stride_means = [float(x) for x in stride_means if x is not None]
                avg_stride_length = float(np.mean(stride_means)) if stride_means else 0.0

                # head bob: prefer magnitude (pixels), else use normalized score if that's all we have
                avg_head_bob = float(
                    locomotion.get("head_bob_magnitude")
                    if locomotion.get("head_bob_magnitude") is not None
                    else locomotion.get("head_bob_score", 0.0)
                )

                # asymmetry: average front/rear asymmetry if present
                asym_vals = [
                    locomotion.get("front_leg_asymmetry"),
                    locomotion.get("rear_leg_asymmetry"),
                ]
                asym_vals = [float(x) for x in asym_vals if x is not None]
                asymmetry_score = float(np.mean(asym_vals)) if asym_vals else 0.0

                features.extend([avg_stride_length, avg_head_bob, asymmetry_score])

            feature_names.extend(["tleap_stride", "tleap_head_bob", "tleap_asymmetry"])
        
        # If no features extracted, return default
        if not features:
            # Return default feature vector
            features = [0.5] * 10
            feature_names = [f"default_{i}" for i in range(10)]
        
        self.feature_names = feature_names
        return np.array(features)
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make predictions using all models and ensemble"""
        predictions = {}
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # CatBoost prediction
        if self.catboost_model:
            try:
                catboost_proba = self.catboost_model.predict_proba(features)[0]
                predictions["catboost"] = {
                    "probability": float(catboost_proba[1]),  # Probability of class 1 (lame)
                    "prediction": int(catboost_proba[1] > 0.5)
                }
            except Exception as e:
                print(f"CatBoost prediction error: {e}")
        
        # XGBoost prediction
        if self.xgboost_model:
            try:
                dmatrix = xgb.DMatrix(features)
                xgboost_proba = self.xgboost_model.predict(dmatrix)[0]
                predictions["xgboost"] = {
                    "probability": float(xgboost_proba),
                    "prediction": int(xgboost_proba > 0.5)
                }
            except Exception as e:
                print(f"XGBoost prediction error: {e}")
        
        # LightGBM prediction
        if self.lightgbm_model:
            try:
                lightgbm_proba = self.lightgbm_model.predict(features)[0]
                predictions["lightgbm"] = {
                    "probability": float(lightgbm_proba),
                    "prediction": int(lightgbm_proba > 0.5)
                }
            except Exception as e:
                print(f"LightGBM prediction error: {e}")
        
        # Ensemble prediction
        ensemble_proba = 0.0
        total_weight = 0.0
        
        for model_name, weight in self.ensemble_weights.items():
            if model_name in predictions:
                ensemble_proba += predictions[model_name]["probability"] * weight
                total_weight += weight
        
        if total_weight > 0:
            ensemble_proba /= total_weight
        else:
            ensemble_proba = 0.5  # Default
        
        predictions["ensemble"] = {
            "probability": float(ensemble_proba),
            "prediction": int(ensemble_proba > 0.5),
            "weights": self.ensemble_weights
        }
        
        return predictions
    
    async def process_video(self, video_data: dict):
        """Process video through ML pipeline"""
        video_id = video_data.get("video_id")
        if not video_id:
            return
        
        print(f"ML pipeline processing video {video_id}")
        
        try:
            # Get results from all pipelines
            pipeline_results = await self.get_pipeline_results(video_id)
            
            # Extract features
            features = self.extract_features(pipeline_results)
            
            # Make predictions
            predictions = self.predict(features)
            
            # Save results
            results = {
                "video_id": video_id,
                "features": features.tolist(),
                "feature_names": self.feature_names,
                "predictions": predictions,
                "pipeline_results_available": {
                    k: v is not None for k, v in pipeline_results.items()
                }
            }
            
            results_file = self.results_dir / f"{video_id}_ml.json"
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
            
            # Publish results
            pipeline_result = {
                "video_id": video_id,
                "pipeline": "ml",
                "results_path": str(results_file),
                "predictions": predictions,
                "ensemble_probability": predictions.get("ensemble", {}).get("probability", 0.5)
            }
            
            await self.nats_client.publish(
                self.config["nats"]["subjects"]["pipeline_ml"],
                pipeline_result
            )
            
            print(f"ML pipeline completed for {video_id}")
            
        except Exception as e:
            print(f"Error in ML pipeline for {video_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the ML pipeline service"""
        await self.nats_client.connect()
        
        # Subscribe to all pipeline results
        subjects = [
            self.config["nats"]["subjects"]["pipeline_yolo"],
            self.config["nats"]["subjects"]["pipeline_sam3"],
            self.config["nats"]["subjects"]["pipeline_dinov3"],
            self.config["nats"]["subjects"]["pipeline_tleap"]
        ]
        
        # Process when we have results from key pipelines
        # For now, subscribe to dinov3 as trigger (last in sequence)
        subject = self.config["nats"]["subjects"]["pipeline_dinov3"]
        print(f"ML pipeline subscribed to {subject}")
        
        await self.nats_client.subscribe(subject, self.process_video)
        
        # Keep running
        print("ML pipeline service started. Waiting for pipeline results...")
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    pipeline = MLPipeline()
    await pipeline.start()


if __name__ == "__main__":
    asyncio.run(main())

