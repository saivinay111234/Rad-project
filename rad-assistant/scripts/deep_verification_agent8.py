
"""
Deep Verification Script for Agent 8.
Verifies integration of real agent classes with mock LLM responses.
"""
import sys
import os
import json
import logging
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.models import (
    StudyOrchestrationRequest, ExamMetadata, ClinicalContext,
    PipelineOptions, PipelineStatus, StageStatus
)
from radiology_assistant.agents.study_orchestrator import StudyOrchestratorAgent
# Import actual agent classes
from radiology_assistant.agents.report_drafter import ReportDraftingAgent
from radiology_assistant.agents.report_qa_agent import ReportQAAgent
from radiology_assistant.agents.followup_extractor import FollowUpExtractorAgent
from radiology_assistant.agents.patient_report_explainer import PatientReportExplainerAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepVerifier")

class MockLLMClient:
    """Mock LLM that returns valid responses for all agent prompts."""
    
    def generate(self, prompt: str, temperature: float = 0.0, max_tokens: int = None, system_prompt: str = None) -> str:
        """Return valid responses for all agent types."""
        sp = (system_prompt or "").lower()
        up = (prompt or "").lower()
        
        # Report Drafter - check for drafting keywords
        if "draft" in sp or "radiologist" in up:
            return json.dumps({
                "report_text": "FINDINGS: The lungs are clear. The heart size is normal.\n\nIMPRESSION: Normal chest radiograph.",
                "key_findings": [
                    {"label": "Clear lungs", "category": "normal", "severity": "normal"}
                ],
                "confidence_score": 0.95,
                "used_cv_signals": []
            })
            
        # QA Agent - check for QA/review keywords
        if "qa" in sp or "review" in sp or "exam_metadata" in up:
            return json.dumps({
                "version": "v1",
                "original_report_text": "Original report text",
                "normalized_report_text": None,
                "issues": [],
                "summary": {
                    "overall_quality": "good",
                    "num_critical": 0,
                    "num_major": 0,
                    "num_minor": 0,
                    "comments": None
                }
            })
            
        # Follow-up Extractor
        if "incidental" in sp or "follow" in sp:
            return json.dumps({
                "incidental_findings": [],
                "has_any_followup": False,
                "global_followup_comment": None
            })
             
        # Patient Explainer
        if "patient" in sp or "friendly" in sp:
            return json.dumps({
                "version": "v1",
                "patient_summary_text": "Your chest X-ray looks normal. No problems were found.",
                "key_points": ["Clear lungs", "Normal heart size"],
                "next_steps": [],
                "glossary": [],
                "original_report_text": "Original report"
            })
        
        # Default fallback - return minimal valid JSON
        logger.warning(f"Mock LLM received unrecognized prompt pattern")
        return json.dumps({"status": "ok"})

def run_deep_verification():
    logger.info("Starting Deep Integration Verification for Agent 8")
    
    # 1. Setup Mock LLM
    mock_llm = MockLLMClient()
    
    # 2. Instantiate Real Sub-Agents
    drafter = ReportDraftingAgent(mock_llm)
    qa = ReportQAAgent(mock_llm)
    followup = FollowUpExtractorAgent(mock_llm)
    explainer = PatientReportExplainerAgent(mock_llm)
    
    # Mock CV Agent (Hardware dependency) and Repo (DB dependency)
    cv_agent = None 
    repo = MagicMock()
    
    # 3. Instantiate Orchestrator
    orchestrator = StudyOrchestratorAgent(
        report_drafter=drafter,
        qa_agent=qa,
        followup_agent=followup,
        patient_explainer=explainer,
        cv_agent=cv_agent,
        learning_repo=repo
    )
    
    # 4. Create Valid Request
    request = StudyOrchestrationRequest(
        study_id="DEEP_TEST_001",
        exam_metadata=ExamMetadata(modality="XR", accession="ACC_TEST", body_region="Chest"),
        clinical_context=ClinicalContext(
            patient_info="Male, 30y", 
            clinical_presentation="Routine checkup"
        ),
        pipeline_options=PipelineOptions(
            run_cv_analysis=False, # Skip CV
            run_qa_review=True,
            run_followup_extraction=True,
            run_patient_summary=True
        )
    )
    
    # 5. Run Execution
    try:
        response = orchestrator.orchestrate_study(request)
        
        logger.info(f"Pipeline Status: {response.pipeline_status}")
        
        # Verify Pipeline Success
        if response.pipeline_status != PipelineStatus.SUCCESS:
            logger.error(f"Pipeline Failed! Stages: {response.stages}")
            sys.exit(1)
            
        # Verify Bundle Content
        if not response.bundle.report_draft:
            logger.error("Missing Draft!")
            sys.exit(1)
            
        if not response.bundle.qa_result:
            logger.error("Missing QA Result!")
            sys.exit(1)
            
        logger.info("Verification Passed: Orchestrator successfully integrated all agents.")
        
        # Output result for inspection
        print(json.dumps(response.model_dump(mode='json'), indent=2))
        
    except Exception as e:
        logger.exception(f"Orchestration threw unexpected exception: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_deep_verification()
