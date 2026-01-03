"""
LLM Explanation Service
Generates human-readable explanations of lameness predictions.

Key Features:
- Evidence-based summaries (no hallucination)
- Structured prompts with strict input constraints
- Executive summary, guidance, and action recommendations
- Priority: OpenAI API > Ollama Local LLM > Skip (no template fallback)
"""
import asyncio
import json
import os
import httpx
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
from shared.utils.nats_client import NATSClient


class LLMExplanationService:
    """
    Service for generating LLM-based explanations of lameness predictions.
    
    Priority:
    1. OpenAI API (if OPENAI_API_KEY is set)
    2. Ollama Local LLM (if running)
    3. Skip explanation generation (no template fallback)
    
    Constraints:
    - Only reference provided inputs (no external knowledge)
    - Explicitly state when evidence is missing or conflicting
    - Keep explanations concise and actionable
    """
    
    # System prompt template
    SYSTEM_PROMPT = """You are a veterinary AI assistant explaining lameness predictions for dairy cows.

STRICT RULES:
1. ONLY reference the data provided in the user message
2. NEVER invent or assume information not in the input
3. If evidence is missing or conflicting, explicitly say so
4. Keep explanations clear and actionable for farm staff
5. Use simple language, avoid jargon

OUTPUT FORMAT (use exact headers):
## Executive Summary
(2-3 sentences: Main conclusion with confidence level)

## Key Evidence
(Bullet points of supporting data from pipelines)

## Uncertainties
(Any missing data or model disagreements)

## Recommended Action
(Clear next step for farm staff)"""

    EXPLANATION_TEMPLATE = """Generate an explanation for this lameness prediction:

## Final Decision
- Prediction: {prediction_label} ({probability:.1%} probability)
- Confidence: {confidence_level} ({confidence:.1%})
- Decision Mode: {decision_mode}

## Pipeline Predictions
{pipeline_summary}

## Quality Indicators
- Clip Quality: {clip_quality}
- Pose Quality: {pose_quality}
- Detection Confidence: {detection_confidence}

## Gait Features (from T-LEAP)
{gait_features}

## Top SHAP Features
{shap_features}

## Human Consensus
{human_consensus}

## Model Agreement
- Agreement Level: {agreement_level}
- Models in agreement: {models_agree}

Generate a clear explanation following the output format specified."""

    def __init__(self):
        self.config_path = Path("/app/shared/config/config.yaml")
        self.config = self._load_config()
        self.nats_client = NATSClient(str(self.config_path))
        
        # OpenAI configuration
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # Ollama configuration
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2")  # Good balance of speed/quality
        
        # Results directory
        self.results_dir = Path("/app/data/results/explanations")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize providers
        self.openai_client = None
        self.ollama_available = False
        self.llm_provider = None
        
        self._init_providers()
    
    def _init_providers(self):
        """Initialize LLM providers in priority order"""
        
        # 1. Try OpenAI first
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                # Test connection
                self.openai_client.models.list()
                self.llm_provider = "openai"
                print(f"✅ OpenAI initialized with model: {self.openai_model}")
                return
            except ImportError:
                print("⚠️ OpenAI library not installed")
            except Exception as e:
                print(f"⚠️ OpenAI connection failed: {e}")
        
        # 2. Try Ollama as fallback
        self._check_ollama()
        if self.ollama_available:
            self.llm_provider = "ollama"
            print(f"✅ Ollama initialized with model: {self.ollama_model}")
            return
        
        # 3. No LLM available
        self.llm_provider = None
        print("⚠️ No LLM available (no OpenAI key, no Ollama running)")
        print("   Explanations will NOT be generated")
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available and has models"""
        try:
            import httpx
            response = httpx.get(f"{self.ollama_host}/api/tags", timeout=5.0)
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                
                if models:
                    # Check if preferred model is available
                    if self.ollama_model not in models:
                        # Use first available model
                        self.ollama_model = models[0]
                        print(f"   Using available Ollama model: {self.ollama_model}")
                    
                    self.ollama_available = True
                    return True
                else:
                    print(f"⚠️ Ollama running but no models installed")
                    print(f"   Run: ollama pull {self.ollama_model}")
            return False
        except Exception as e:
            print(f"⚠️ Ollama not available at {self.ollama_host}: {e}")
            return False
    
    def _load_config(self) -> dict:
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def format_pipeline_summary(self, pipeline_contributions: Dict[str, Any]) -> str:
        """Format pipeline predictions for the prompt"""
        lines = []
        for pipeline, data in pipeline_contributions.items():
            if isinstance(data, dict):
                prob = data.get("probability", 0.5)
                pred = "Lame" if prob > 0.5 else "Sound"
                uncertainty = data.get("uncertainty", 0.5)
                lines.append(f"- {pipeline.upper()}: {pred} ({prob:.1%} probability, uncertainty: {uncertainty:.1%})")
            else:
                lines.append(f"- {pipeline.upper()}: {data}")
        
        return "\n".join(lines) if lines else "No pipeline predictions available"
    
    def format_gait_features(self, tleap_features: Dict[str, Any]) -> str:
        """Format T-LEAP gait features for the prompt"""
        if not tleap_features:
            return "No gait features available"
        
        feature_descriptions = {
            "back_arch_mean": "Back arch angle",
            "back_arch_score": "Back arch severity",
            "head_bob_magnitude": "Head bobbing intensity",
            "head_bob_score": "Head bob severity",
            "front_leg_asymmetry": "Front leg asymmetry",
            "rear_leg_asymmetry": "Rear leg asymmetry",
            "lameness_score": "Overall lameness score"
        }
        
        lines = []
        for key, value in tleap_features.items():
            if key in feature_descriptions:
                severity = "High" if value > 0.7 else "Medium" if value > 0.4 else "Low"
                lines.append(f"- {feature_descriptions[key]}: {value:.2f} ({severity})")
        
        return "\n".join(lines) if lines else "No significant gait abnormalities detected"
    
    def format_shap_features(self, shap_data: Dict[str, Any]) -> str:
        """Format SHAP feature importance for the prompt"""
        if not shap_data or "top_features" not in shap_data:
            return "SHAP analysis not available"
        
        lines = []
        for feature in shap_data.get("top_features", [])[:5]:
            name = feature.get("name", "Unknown")
            value = feature.get("value", 0)
            importance = feature.get("importance", 0)
            direction = "increases" if importance > 0 else "decreases"
            lines.append(f"- {name}: {value:.2f} ({direction} lameness probability)")
        
        return "\n".join(lines) if lines else "No significant feature contributions"
    
    def format_human_consensus(self, human_data: Optional[Dict[str, Any]]) -> str:
        """Format human consensus data for the prompt"""
        if not human_data or human_data.get("num_raters", 0) == 0:
            return "No human labels available for this video"
        
        prob = human_data.get("probability", 0.5)
        confidence = human_data.get("confidence", 0.5)
        num_raters = human_data.get("num_raters", 0)
        
        consensus = "Lame" if prob > 0.5 else "Sound"
        conf_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        
        return f"Human assessors ({num_raters} raters): {consensus} with {conf_level} confidence ({confidence:.1%})"
    
    def build_prompt(self, fusion_result: Dict[str, Any], 
                     shap_data: Optional[Dict[str, Any]] = None,
                     quality_data: Optional[Dict[str, Any]] = None) -> str:
        """Build the full prompt for LLM explanation"""
        
        # Extract key information
        probability = fusion_result.get("final_probability", 0.5)
        confidence = fusion_result.get("confidence", 0.5)
        prediction_label = "Lame" if probability > 0.5 else "Sound"
        confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
        decision_mode = fusion_result.get("decision_mode", "unknown")
        
        # Pipeline summary
        pipeline_contributions = fusion_result.get("pipeline_contributions", {})
        pipeline_summary = self.format_pipeline_summary(pipeline_contributions)
        
        # Quality indicators
        quality_data = quality_data or {}
        clip_quality = quality_data.get("clip_quality", "Unknown")
        pose_quality = quality_data.get("pose_quality", "Unknown")
        detection_conf = quality_data.get("detection_confidence", "Unknown")
        
        # Gait features
        tleap_features = fusion_result.get("tleap_features", {})
        gait_features = self.format_gait_features(tleap_features)
        
        # SHAP features
        shap_features = self.format_shap_features(shap_data or {})
        
        # Human consensus
        human_data = pipeline_contributions.get("human", {})
        human_consensus = self.format_human_consensus(human_data)
        
        # Agreement level
        model_agreement = fusion_result.get("model_agreement", 0.5)
        unanimous = fusion_result.get("unanimous", False)
        agreement_level = "Unanimous" if unanimous else "High" if model_agreement > 0.8 else "Medium" if model_agreement > 0.5 else "Low"
        models_agree = "All models agree" if unanimous else f"{len(pipeline_contributions)} models with {agreement_level.lower()} agreement"
        
        return self.EXPLANATION_TEMPLATE.format(
            prediction_label=prediction_label,
            probability=probability,
            confidence_level=confidence_level,
            confidence=confidence,
            decision_mode=decision_mode,
            pipeline_summary=pipeline_summary,
            clip_quality=clip_quality,
            pose_quality=pose_quality,
            detection_confidence=detection_conf,
            gait_features=gait_features,
            shap_features=shap_features,
            human_consensus=human_consensus,
            agreement_level=agreement_level,
            models_agree=models_agree
        )
    
    async def generate_explanation_openai(self, prompt: str) -> Optional[str]:
        """Generate explanation using OpenAI API"""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Low temperature for consistency
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return None
    
    async def generate_explanation_ollama(self, prompt: str) -> Optional[str]:
        """Generate explanation using Ollama local LLM"""
        if not self.ollama_available:
            return None
        
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 500
                        }
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("response", "")
                else:
                    print(f"Ollama API error: {response.status_code}")
                    return None
        except Exception as e:
            print(f"Ollama error: {e}")
            # Mark as unavailable for future requests
            self.ollama_available = False
            return None
    
    async def generate_explanation(self, video_id: str, 
                                   fusion_result: Dict[str, Any],
                                   shap_data: Optional[Dict[str, Any]] = None,
                                   quality_data: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Generate complete explanation for a video"""
        
        # Check if any LLM is available
        if not self.llm_provider:
            # Re-check Ollama in case it was started
            self._check_ollama()
            if self.ollama_available:
                self.llm_provider = "ollama"
            else:
                print(f"  ⏭️ Skipping explanation for {video_id} (no LLM available)")
                return None
        
        # Build prompt
        prompt = self.build_prompt(fusion_result, shap_data, quality_data)
        
        # Generate explanation based on provider
        explanation_text = None
        
        if self.llm_provider == "openai":
            explanation_text = await self.generate_explanation_openai(prompt)
            if not explanation_text:
                # Try Ollama as fallback
                self._check_ollama()
                if self.ollama_available:
                    explanation_text = await self.generate_explanation_ollama(prompt)
                    if explanation_text:
                        self.llm_provider = "ollama"
        
        elif self.llm_provider == "ollama":
            explanation_text = await self.generate_explanation_ollama(prompt)
        
        # If no explanation generated, skip
        if not explanation_text:
            print(f"  ⏭️ Skipping explanation for {video_id} (LLM generation failed)")
            return None
        
        # Parse sections from explanation
        sections = {
            "executive_summary": "",
            "key_evidence": "",
            "uncertainties": "",
            "recommended_action": ""
        }
        
        current_section = None
        for line in explanation_text.split("\n"):
            line_lower = line.lower()
            if "executive summary" in line_lower:
                current_section = "executive_summary"
            elif "key evidence" in line_lower:
                current_section = "key_evidence"
            elif "uncertainties" in line_lower:
                current_section = "uncertainties"
            elif "recommended action" in line_lower:
                current_section = "recommended_action"
            elif current_section:
                sections[current_section] += line + "\n"
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        result = {
            "video_id": video_id,
            "explanation": explanation_text,
            "sections": sections,
            "prompt_used": prompt,
            "llm_provider": self.llm_provider,
            "llm_model": self.openai_model if self.llm_provider == "openai" else self.ollama_model,
            "fusion_summary": {
                "prediction": "Lame" if fusion_result.get("final_probability", 0.5) > 0.5 else "Sound",
                "probability": fusion_result.get("final_probability", 0.5),
                "confidence": fusion_result.get("confidence", 0.5),
                "decision_mode": fusion_result.get("decision_mode", "unknown")
            }
        }
        
        # Save explanation
        output_path = self.results_dir / f"{video_id}_explanation.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
    
    async def handle_explanation_request(self, data: dict):
        """Handle incoming explanation request"""
        video_id = data.get("video_id")
        if not video_id:
            return
        
        print(f"Generating explanation for {video_id}")
        
        try:
            # Load fusion results
            fusion_path = Path(f"/app/data/results/fusion/{video_id}_fusion.json")
            if not fusion_path.exists():
                print(f"  No fusion results for {video_id}")
                return
            
            with open(fusion_path) as f:
                fusion_data = json.load(f)
            
            fusion_result = fusion_data.get("fusion_result", {})
            
            # Load SHAP data if available
            shap_path = Path(f"/app/data/results/shap/{video_id}_shap.json")
            shap_data = None
            if shap_path.exists():
                with open(shap_path) as f:
                    shap_data = json.load(f)
            
            # Generate explanation
            explanation = await self.generate_explanation(
                video_id, fusion_result, shap_data
            )
            
            if explanation:
                print(f"  ✅ Explanation generated for {video_id} via {self.llm_provider}")
                
                # Publish result
                await self.nats_client.publish(
                    "explanation.generated",
                    {
                        "video_id": video_id,
                        "explanation_path": str(self.results_dir / f"{video_id}_explanation.json"),
                        "summary": explanation["sections"]["executive_summary"][:200],
                        "provider": self.llm_provider
                    }
                )
            
        except Exception as e:
            print(f"  ❌ Error generating explanation: {e}")
            import traceback
            traceback.print_exc()
    
    async def start(self):
        """Start the LLM explanation service"""
        await self.nats_client.connect()
        
        # Subscribe to analysis complete events
        subject = self.config.get("nats", {}).get("subjects", {}).get(
            "analysis_complete", "analysis.complete"
        )
        print(f"LLM Explanation Service subscribing to: {subject}")
        
        await self.nats_client.subscribe(subject, self.handle_explanation_request)
        
        print("=" * 60)
        print("LLM Explanation Service Started")
        print("=" * 60)
        if self.llm_provider == "openai":
            print(f"Provider: OpenAI ({self.openai_model})")
        elif self.llm_provider == "ollama":
            print(f"Provider: Ollama Local ({self.ollama_model})")
        else:
            print("Provider: NONE - Explanations will be skipped")
            print("  To enable: Set OPENAI_API_KEY or run Ollama")
        print("=" * 60)
        
        await asyncio.Event().wait()


async def main():
    """Main entry point"""
    service = LLMExplanationService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
