"""
Worklist Triage Agent.

Prioritizes radiology worklist items using CV models and clinical context.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import json

from ..config import Config
from ..models import (
    WorklistTriageRequest,
    WorklistTriageResponse,
    WorklistTriageItem,
    WorklistItem,
    TriageReason,
    TriageReasonType,
    TriageLabel,
    TriageConfig,
    TriageThresholdConfig,
    ModalityGroup,
    CVHighlightRequest
)
from ..llm_client import LLMClient
from ..cv.models import ChestXrayAnomalyModel
from ..agents.visual_highlighter import VisualHighlightingAgent

logger = logging.getLogger(__name__)


class CVRouter:
    """Routes studies to the appropriate CV model based on modality/region."""

    def __init__(self, config: TriageConfig):
        self.config = config
        self._models: Dict[str, Any] = {}
        # Pre-load XR chest model if configured (lazy loading in refined version, but simple here)
        # In production this would be more sophisticated resource management.
        
    def get_model_for_item(self, item: WorklistItem) -> Optional[VisualHighlightingAgent]:
        """
        Return an initialized VisualHighlightingAgent for the item, or None if no model available.
        """
        key = f"{item.modality}/{item.body_region}"
        
        # Check explicit mapping in config first if implemented
        # For now, hardcode the rule for XR/Chest as per plan
        if item.modality == "XR" and item.body_region == "Chest":
            if "xr_chest" not in self._models:
                try:
                    # In a real app, we'd load weights specified in config
                    cv_model = ChestXrayAnomalyModel(device="cpu") 
                    self._models["xr_chest"] = VisualHighlightingAgent(cv_model)
                    logger.info("Loaded XR/Chest model for triage.")
                except Exception as e:
                    logger.error(f"Failed to load XR/Chest model: {e}")
                    return None
            return self._models["xr_chest"]
            
        return None


class WorklistTriageAgent:
    """Agent for triaging worklist items."""

    def __init__(self, config: TriageConfig, llm_client: Optional[LLMClient] = None):
        self.config = config
        self.llm_client = llm_client
        self.cv_router = CVRouter(config)
        self.logger = logging.getLogger(__name__)
        
        # Validate config
        if not self.config.thresholds:
            self.logger.warning("No triage thresholds configured!")

    def _get_thresholds(self, modality: str, body_region: Optional[str]) -> Optional[TriageThresholdConfig]:
        """Find matching threshold config."""
        for t in self.config.thresholds:
            # Simple matching logic
            if t.modality_group.value == modality:
                if t.body_region is None or t.body_region.lower() == (body_region or "").lower():
                    return t
        
        # Fallback: try to match just modality group if body region specific one not found
        for t in self.config.thresholds:
             if t.modality_group.value == modality and t.body_region is None:
                 return t
                 
        return None

    def _calculate_triage(
        self, 
        item: WorklistItem, 
        cv_result: Optional[Dict[str, Any]], 
        thresholds: TriageThresholdConfig
    ) -> WorklistTriageItem:
        """
        Calculate triage score and label for a single item.
        """
        reasons: List[TriageReason] = []
        score = 0.0
        
        # 1. Base score from CV
        cv_max_prob = 0.0
        if cv_result:
            # Assuming CV result has a summary or regions with scores
            # We'll extract the max score from regions
            # The VisualHighlightingAgent returns CVHighlightResult
            regions = cv_result.regions
            if regions:
                best_region = max(regions, key=lambda r: r.score)
                cv_max_prob = best_region.score
                score = max(score, cv_max_prob)
                
                reasons.append(TriageReason(
                    type=TriageReasonType.MODEL_PREDICTION,
                    description=f"Model detected {best_region.label} with confidence {best_region.score:.2f}",
                    weight=cv_max_prob
                ))
            else:
                 reasons.append(TriageReason(
                    type=TriageReasonType.MODEL_PREDICTION,
                    description="Model detected no significant anomalies",
                    weight=0.0
                ))

        # 2. Adjust for Clinical Priority (STAT)
        if item.priority_flag_from_order and item.priority_flag_from_order.upper() == "STAT":
            # Boost score to at least high threshold if STAT
            stat_boost = 0.8  # Arbitrary high base
            if score < stat_boost:
                score = stat_boost
            reasons.append(TriageReason(
                type=TriageReasonType.CLINICAL_INDICATION,
                description="Order marked as STAT",
                weight=0.5
            ))
            
        # 3. Adjust for Time in Queue (Linear ramp? Simple bump?)
        # For simplicity, we won't implement time parsing yet unless requested, 
        # but the hook is here.
        
        # 4. Assign Label
        if score >= thresholds.critical_threshold:
            label = TriageLabel.CRITICAL
        elif score >= thresholds.high_threshold:
            label = TriageLabel.HIGH
        elif score >= thresholds.low_threshold:
            label = TriageLabel.ROUTINE
        else:
            label = TriageLabel.LOW
            
        # 5. Metadata
        metadata = {}
        if cv_result:
            metadata["cv_model"] = "ChestXrayAnomalyModel" # Hardcoded for now
            # Only include succinct metadata
            metadata["regions_count"] = len(cv_result.regions)

        return WorklistTriageItem(
            study_id=item.study_id,
            triage_score=score,
            triage_label=label,
            reasons=reasons,
            model_metadata=metadata
        )

    def _generate_explanation(self, item: WorklistTriageItem, worklist_item: WorklistItem) -> Optional[str]:
        """Generate LLM explanation."""
        if not self.config.enable_llm_explanation or not self.llm_client:
            return None
            
        if item.triage_label == TriageLabel.LOW or item.triage_label == TriageLabel.ROUTINE:
             return None # Skip explanation for routine cases to save tokens/time

        try:
            prompt = f"""Generate a single short sentence (max 20 words) explaining why this radiology study is triaged as {item.triage_label.value}.
Context:
- Clinical Indication: {worklist_item.clinical_indication or 'None'}
- AI Findings: {', '.join([r.description for r in item.reasons if r.type == TriageReasonType.MODEL_PREDICTION])}
- Order Priority: {worklist_item.priority_flag_from_order or 'Routine'}

Explanation:"""
            
            explanation = self.llm_client.generate(prompt, temperature=0.3, max_tokens=50)
            return explanation.strip()
        except Exception as e:
            self.logger.warning(f"LLM explanation failed for {item.study_id}: {e}")
            return None

    def triage(self, request: WorklistTriageRequest) -> WorklistTriageResponse:
        """
        Main entry point for batch triage.
        """
        start_time = time.monotonic()
        results: List[WorklistTriageItem] = []
        
        self.logger.info(f"Received batch of {len(request.worklist_items)} items for triage.")
        
        # Group items? Or just iterate? 
        # Since we load models dynamically, simple iteration is fine for MVP/v1.
        
        for item in request.worklist_items:
            try:
                # 1. Config Lookup
                thresholds = self._get_thresholds(item.modality, item.body_region)
                if not thresholds:
                    # Config missing -> UNTRIAGED
                    results.append(WorklistTriageItem(
                        study_id=item.study_id,
                        triage_score=0.0,
                        triage_label=TriageLabel.UNTRIAGED,
                        reasons=[TriageReason(type=TriageReasonType.FALLBACK, description="No triage configuration for modality/region")],
                        error="Missing configuration"
                    ))
                    continue

                # 2. CV Model
                cv_agent = self.cv_router.get_model_for_item(item)
                cv_result = None
                
                if cv_agent and item.image_reference:
                    # Load and run inference
                    # We assume item.image_reference contains path or bytes we can read
                    # For this implementation, we need to handle image loading.
                    # Since we are "receiving a list of studies", the image_reference likely has a path.
                    # If it's a real file path, load it.
                    
                    try:
                        image_path = item.image_reference.get("thumbnail_path")
                        if image_path:
                             with open(image_path, "rb") as f:
                                 image_bytes = f.read()
                             
                             # Use VisualHighlightingAgent's highlight method
                             # It expects a CVHighlightRequest
                             hl_req = CVHighlightRequest(
                                 modality=item.modality,
                                 body_part=item.body_region
                             )
                             cv_result = cv_agent.highlight(hl_req, image_bytes)
                    except Exception as e:
                        self.logger.error(f"CV Inference failed for {item.study_id}: {e}")
                        # We continue without CV result
                
                # 3. Calculate
                triage_item = self._calculate_triage(item, cv_result, thresholds)
                
                # 4. Explain (optional)
                triage_item.explanation_text = self._generate_explanation(triage_item, item)
                
                results.append(triage_item)

            except Exception as e:
                self.logger.exception(f"Unexpected error triaging item {item.study_id}")
                # Fail-soft: Return error item
                results.append(WorklistTriageItem(
                    study_id=item.study_id,
                    triage_score=0.0,
                    triage_label=TriageLabel.UNTRIAGED,
                    reasons=[],
                    error=f"Internal error: {str(e)}"
                ))

        duration = time.monotonic() - start_time
        self.logger.info(f"Triage batch completed in {duration:.2f}s")
        
        return WorklistTriageResponse(
            version="v1",
            items=results
        )
