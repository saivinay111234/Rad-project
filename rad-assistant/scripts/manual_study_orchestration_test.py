
"""
Manual Verification Script for Agent 8 (Study Orchestrator).
"""
import sys
import os
import logging
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.models import (
    StudyOrchestrationRequest, ExamMetadata, ClinicalContext,
    PipelineOptions, ReportDraft, ReportQAResponse, QASummary,
    FollowUpExtractionResponse, PatientReportSummaryResponse
)
from radiology_assistant.agents.study_orchestrator import StudyOrchestratorAgent

# Setup logging
logging.basicConfig(level=logging.INFO)

def run_manual_test():
    print("--- Starting Manual Orchestration Test ---")
    try:
        ctx = ClinicalContext(patient_info="Male, 45y", clinical_presentation="Cough")
        print(f"Created ClinicalContext: {ctx}")
    except Exception as e:
        print(f"Failed to create ClinicalContext: {e}")
        return
    
    # 1. Mock Sub-Agents
    mock_drafter = MagicMock()
    mock_drafter.draft_report.return_value = ReportDraft(
        report_text="Findings: Clear lungs. Impression: Normal.",
        key_findings=[],
        used_cv_signals=[],
        confidence_score=0.95
    )
    
    mock_qa = MagicMock()
    mock_qa.review_report.return_value = ReportQAResponse(
        version="v1",
        original_report_text="...",
        normalized_report_text=None,
        issues=[],
        summary=QASummary(overall_quality="good", num_critical=0, num_major=0, num_minor=0)
    )
    
    mock_followup = MagicMock()
    mock_followup.extract_followups.return_value = FollowUpExtractionResponse(
        incidental_findings=[], has_any_followup=False, global_followup_comment="None"
    )
    
    mock_explainer = MagicMock()
    mock_explainer.explain.return_value = PatientReportSummaryResponse(
        version="v1",
        patient_summary_text="Everything looks normal.",
        key_points=["Lungs clear"],
        next_steps=[],
        glossary=[],
        original_report_text="..."
    )
    
    mock_cv = MagicMock() # Not used as we won't pass images
    
    mock_repo = MagicMock()
    
    # 2. Instantiate Orchestrator
    orchestrator = StudyOrchestratorAgent(
        report_drafter=mock_drafter,
        qa_agent=mock_qa,
        followup_agent=mock_followup,
        patient_explainer=mock_explainer,
        cv_agent=mock_cv,
        learning_repo=mock_repo
    )
    
    try:
        ctx = ClinicalContext(patient_info="Male, 45y", clinical_presentation="Cough")
        print(f"Created ClinicalContext: {ctx}")
    except Exception as e:
        print(f"Failed to create ClinicalContext: {e}")
        return
    
    # 1. Mock Sub-Agents ... (omitted)

    # 3. Create Request
    req = StudyOrchestrationRequest(
        study_id="MANUAL_TEST_001",
        exam_metadata=ExamMetadata(modality="XR", accession="ACC123", body_region="Chest"),
        clinical_context=ctx,
        pipeline_options=PipelineOptions(run_cv_analysis=False)
    )
    print(f"Request created: {req.model_dump()}")
    
    # 4. Run
    try:
        response = orchestrator.orchestrate_study(req)
        
        print(f"\nPipeline Status: {response.pipeline_status}")
        print(f"Study ID: {response.study_id}")
        print(f"Final Report: {response.bundle.final_report_text}")
        print("\nStages:")
        for s in response.stages:
            print(f"- {s.stage}: {s.status}")
            
    except Exception as e:
        print(f"Orchestration Failed: {e}")

if __name__ == "__main__":
    run_manual_test()
