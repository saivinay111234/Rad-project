import pytest
from unittest.mock import Mock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.agents.patient_report_explainer import (
    PatientReportExplainerAgent, 
    LLMJsonParseError
)
from radiology_assistant.models import (
    PatientReportSummaryRequest,
    PatientReportSummaryResponse,
    ExamMetadata,
    PatientReadingLevel,
    PatientSummaryTone,
    FollowUpExtractionResponse,
    IncidentalFinding,
    IncidentalFindingCategory,
    FollowUpType,
    RecommendationStrength,
    FollowUpInterval
)

SAMPLE_GOOD_RESPONSE = {
    "version": "v1",
    "patient_summary_text": "Normal exam.",
    "key_points": ["No issues."],
    "next_steps": [],
    "glossary": [],
    "original_report_text": "Normal."
}

SAMPLE_WITH_FOLLOWUP = {
    "version": "v1",
    "patient_summary_text": "A spot was found.",
    "key_points": ["Nodule found."],
    "next_steps": [
        {
            "description": "Follow-up scan recommended.",
            "urgency": "soon",
            "followup_interval": {"months": 6},
            "source_finding_id": "IF1"
        }
    ],
    "glossary": [{"term": "nodule", "explanation": "a small lump"}],
    "original_report_text": "Nodule found."
}

class TestPatientReportExplainerAgent:

    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def agent(self, mock_llm_client):
        return PatientReportExplainerAgent(llm_client=mock_llm_client)

    @pytest.fixture
    def base_request(self):
        return PatientReportSummaryRequest(
            exam_metadata=ExamMetadata(modality="XR", body_region="Chest"),
            report_text="Normal chest x-ray."
        )

    def test_explain_happy_path(self, agent, mock_llm_client, base_request):
        # Arrange
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_GOOD_RESPONSE)
        
        # Act
        response = agent.explain(base_request)
        
        # Assert
        assert isinstance(response, PatientReportSummaryResponse)
        assert response.patient_summary_text == "Normal exam."
        assert len(response.next_steps) == 0

    def test_explain_with_followup_data(self, agent, mock_llm_client):
        # Arrange
        followup_data = FollowUpExtractionResponse(
            incidental_findings=[
                IncidentalFinding(
                    id="IF1", category=IncidentalFindingCategory.pulmonary_nodule,
                    description="nodule", followup_required=True,
                    followup_type=FollowUpType.imaging, recommendation_strength=RecommendationStrength.explicit,
                    verbatim_snippet="nodule",
                    followup_interval=FollowUpInterval(months=6)
                )
            ],
            has_any_followup=True
        )
        request = PatientReportSummaryRequest(
            exam_metadata=ExamMetadata(modality="CT"),
            report_text="Nodule found.",
            followup_data=followup_data,
            reading_level=PatientReadingLevel.STANDARD
        )
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_WITH_FOLLOWUP)
        
        # Act
        response = agent.explain(request)
        
        # Assert
        assert len(response.next_steps) == 1
        # In a real LLM call we'd expect the agent to map IF1, but here we just mock the return
        assert response.next_steps[0].source_finding_id == "IF1"

    def test_no_followup_data_empty_steps(self, agent, mock_llm_client, base_request):
        # Arrange: Prompt returns no steps if report is normal
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_GOOD_RESPONSE)
        
        # Act
        response = agent.explain(base_request)
        
        # Assert
        assert len(response.next_steps) == 0

    def test_retry_on_malformed_json(self, agent, mock_llm_client, base_request):
        # Arrange
        mock_llm_client.generate.side_effect = ["BAD JSON", json.dumps(SAMPLE_GOOD_RESPONSE)]
        
        # Act
        response = agent.explain(base_request)
        
        # Assert
        assert mock_llm_client.generate.call_count == 2
        assert response.patient_summary_text == "Normal exam."

    def test_failure_after_retries(self, agent, mock_llm_client, base_request):
        # Arrange
        mock_llm_client.generate.return_value = "REALLY BAD JSON"
        
        # Act & Assert
        with pytest.raises(LLMJsonParseError):
            agent.explain(base_request)
