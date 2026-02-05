
import pytest
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.models import (
    StudyOrchestrationRequest, PipelineOptions, ExamMetadata, ClinicalContext,
    PipelineStatus, StageStatus, StudyPipelineStage,
    ReportDraft, ReportQAResponse, FollowUpExtractionResponse, PatientReportSummaryResponse,
    QASummary
)
from radiology_assistant.agents.study_orchestrator import StudyOrchestratorAgent

# Mocks
@pytest.fixture
def mock_subagents():
    return {
        "drafter": MagicMock(),
        "qa": MagicMock(),
        "followup": MagicMock(),
        "explainer": MagicMock(),
        "cv": MagicMock(),
        "repo": MagicMock()
    }

@pytest.fixture
def orchestrator(mock_subagents):
    return StudyOrchestratorAgent(
        report_drafter=mock_subagents["drafter"],
        qa_agent=mock_subagents["qa"],
        followup_agent=mock_subagents["followup"],
        patient_explainer=mock_subagents["explainer"],
        cv_agent=mock_subagents["cv"],
        learning_repo=mock_subagents["repo"]
    )

@pytest.fixture
def sample_request():
    return StudyOrchestrationRequest(
        study_id="S123",
        exam_metadata=ExamMetadata(modality="CX", accession="A123"),
        clinical_context=ClinicalContext(patient_info="Patient A", clinical_presentation="Cough"),
        pipeline_options=PipelineOptions() # All true by default
    )

def test_orchestrate_happy_path(orchestrator, mock_subagents, sample_request):
    """Test full pipeline compliance."""
    # Setup mocks with real Pydantic objects
    mock_subagents["drafter"].draft_report.return_value = ReportDraft(
        report_text="Draft Report", key_findings=[], used_cv_signals=[], confidence_score=0.9
    )
    mock_subagents["qa"].review_report.return_value = ReportQAResponse(
        version="v1", original_report_text="Draft Report", normalized_report_text="Clean Report",
        issues=[], summary=QASummary(overall_quality="good", num_critical=0, num_major=0, num_minor=0, comments=None)
    )
    mock_subagents["followup"].extract_followups.return_value = FollowUpExtractionResponse(
        incidental_findings=[], has_any_followup=False
    )
    mock_subagents["explainer"].explain.return_value = PatientReportSummaryResponse(
        version="v1", patient_summary_text="Clean Report", key_points=[], next_steps=[], 
        glossary=[], original_report_text="Clean Report"
    )
    
    # Run
    response = orchestrator.orchestrate_study(sample_request)
    
    assert response.pipeline_status == PipelineStatus.SUCCESS
    assert response.bundle.final_report_text == "Clean Report"
    
    # Check stages
    assert len(response.stages) >= 4
    for stage in response.stages:
        if stage.stage != StudyPipelineStage.CV_ANALYSIS: # CV skipped as no image
             assert stage.status == StageStatus.SUCCESS
        else:
             assert stage.status == StageStatus.SKIPPED

def test_orchestrate_draft_failure(orchestrator, mock_subagents, sample_request):
    """Test critical failure at draft stage."""
    mock_subagents["drafter"].draft_report.side_effect = Exception("LLM Error")
    
    response = orchestrator.orchestrate_study(sample_request)
    
    assert response.pipeline_status == PipelineStatus.FAILED
    
    draft_stage = next(s for s in response.stages if s.stage == StudyPipelineStage.REPORT_DRAFT)
    assert draft_stage.status == StageStatus.FAILED
    
    qa_stage = next(s for s in response.stages if s.stage == StudyPipelineStage.QA_REVIEW)
    assert qa_stage.status == StageStatus.SKIPPED

def test_orchestrate_qa_failure_fail_soft(orchestrator, mock_subagents, sample_request):
    """Test non-critical failure at QA stage."""
    mock_subagents["drafter"].draft_report.return_value = ReportDraft(
        report_text="Draft Report", key_findings=[], used_cv_signals=[], confidence_score=0.9
    )
    mock_subagents["qa"].review_report.side_effect = Exception("QA Error")
    
    response = orchestrator.orchestrate_study(sample_request)
    
    assert response.pipeline_status == PipelineStatus.PARTIAL_SUCCESS
    assert response.bundle.final_report_text == "Draft Report" # Fallback
    
    qa_stage = next(s for s in response.stages if s.stage == StudyPipelineStage.QA_REVIEW)
    assert qa_stage.status == StageStatus.FAILED
    
    # Followup should still run
    mock_subagents["followup"].extract_followups.assert_called()

def test_options_skip_stages(orchestrator, mock_subagents, sample_request):
    """Test disabling stages via options."""
    sample_request.pipeline_options.run_qa_review = False
    
    mock_subagents["drafter"].draft_report.return_value = ReportDraft(
        report_text="Draft Report", key_findings=[], used_cv_signals=[], confidence_score=0.9
    )
    
    response = orchestrator.orchestrate_study(sample_request)
    
    qa_stage = next(s for s in response.stages if s.stage == StudyPipelineStage.QA_REVIEW)
    assert qa_stage.status == StageStatus.SKIPPED
    mock_subagents["qa"].review_report.assert_not_called()
