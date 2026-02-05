import pytest
from unittest.mock import Mock, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.agents.followup_extractor import FollowUpExtractorAgent
from radiology_assistant.models import (
    FollowUpExtractionRequest, 
    FollowUpExtractionResponse, 
    ExamMetadata, 
    IncidentalFinding,
    IncidentalFindingCategory,
    FollowUpType,
    RecommendationStrength
)

# Sample Valid JSON Response from LLM
SAMPLE_LLM_RESPONSE = {
    "incidental_findings": [
        {
            "id": "IF1",
            "description": "Pulmonary nodule",
            "verbatim_snippet": "6 mm solid nodule in the right upper lobe",
            "location": "Right upper lobe",
            "size": "6 mm",
            "category": "pulmonary_nodule",
            "followup_required": True,
            "followup_type": "imaging",
            "followup_modality": "CT chest",
            "followup_interval": {"months": 6},
            "followup_interval_text": "6 months",
            "followup_rationale": "Solid nodule > 4mm in high risk patient",
            "recommendation_strength": "explicit"
        }
    ],
    "global_followup_comment": None,
    "has_any_followup": True
}

class TestFollowUpExtractorAgent:
    
    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def agent(self, mock_llm_client):
        return FollowUpExtractorAgent(llm_client=mock_llm_client)

    @pytest.fixture
    def sample_request(self):
        return FollowUpExtractionRequest(
            exam_metadata=ExamMetadata(
                modality="DX",
                exam_date="2023-10-27"
            ),
            report_text="FINDINGS: 6 mm solid nodule in the right upper lobe. IMPRESSION: RUL nodule, recommend CT chest in 6 months."
        )

    def test_extract_followups_success(self, agent, mock_llm_client, sample_request):
        # Arrange
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_LLM_RESPONSE)
        
        # Act
        response = agent.extract_followups(sample_request)
        
        # Assert
        assert isinstance(response, FollowUpExtractionResponse)
        assert response.has_any_followup is True
        assert len(response.incidental_findings) == 1
        finding = response.incidental_findings[0]
        assert finding.id == "IF1"
        assert finding.category == IncidentalFindingCategory.pulmonary_nodule
        assert finding.followup_type == FollowUpType.imaging
        assert finding.recommendation_strength == RecommendationStrength.explicit
        assert finding.followup_interval.months == 6

    def test_clean_json_blocks(self, agent):
        # Helper test for markdown cleanup
        raw = "```json\n{\"foo\": \"bar\"}\n```"
        cleaned = agent._clean_json(raw)
        assert cleaned == "{\"foo\": \"bar\"}"

    def test_retry_logic_on_malformed_json(self, agent, mock_llm_client, sample_request):
        # Arrange: First call returns garbage, second returns proper JSON
        mock_llm_client.generate.side_effect = [
            "This is not JSON",
            json.dumps(SAMPLE_LLM_RESPONSE)
        ]
        
        # Act
        response = agent.extract_followups(sample_request)
        
        # Assert
        assert mock_llm_client.generate.call_count == 2
        assert response.has_any_followup is True

    def test_failure_handling(self, agent, mock_llm_client, sample_request):
        # Arrange: Always fails
        mock_llm_client.generate.side_effect = ["Bad JSON", "Still Bad JSON"]
        
        # Act
        response = agent.extract_followups(sample_request)
        
        # Assert
        assert response.has_any_followup is False
        assert len(response.incidental_findings) == 0
        assert "Extraction failed" in str(response.global_followup_comment)
