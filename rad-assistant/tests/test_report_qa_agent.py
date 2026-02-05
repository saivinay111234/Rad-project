import pytest
from unittest.mock import Mock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.agents.report_qa_agent import ReportQAAgent, LLMJsonParseError
from radiology_assistant.models import (
    ReportQARequest,
    ReportQAResponse,
    ExamMetadata, 
    QASeverity,
    QAType,
    QASection,
    QAChangeType
)

SAMPLE_GOOD_RESPONSE = {
    "version": "v1",
    "original_report_text": "text",
    "issues": [],
    "summary": {
        "overall_quality": "good",
        "num_critical": 0,
        "num_major": 0,
        "num_minor": 0,
        "comments": "Great report."
    }
}

SAMPLE_BAD_RESPONSE = {
    "version": "v1",
    "original_report_text": "text",
    "issues": [
        {
            "id": "QA1",
            "severity": "critical",
            "type": "consistency",
            "section": "IMPRESSION",
            "description": "Contradiction detected.",
            "location_hint": "Impression 1",
            "suggested_change_type": "suggest_edit",
            "suggested_text": None
        }
    ],
    "summary": {
        "overall_quality": "needs_revision",
        "num_critical": 1,
        "num_major": 0,
        "num_minor": 0,
        "comments": None
    }
}

class TestReportQAAgent:

    @pytest.fixture
    def mock_llm_client(self):
        return Mock()

    @pytest.fixture
    def agent(self, mock_llm_client):
        return ReportQAAgent(llm_client=mock_llm_client)

    @pytest.fixture
    def sample_request(self):
        return ReportQARequest(
            exam_metadata=ExamMetadata(modality="XR", body_region="Chest"),
            report_text="FINDINGS: Normal. IMPRESSION: Normal."
        )

    def test_review_report_clean_happy_path(self, agent, mock_llm_client, sample_request):
        # Arrange
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_GOOD_RESPONSE)

        # Act
        response = agent.review_report(sample_request)

        # Assert
        assert isinstance(response, ReportQAResponse)
        assert len(response.issues) == 0
        assert response.summary.overall_quality == "good"

    def test_review_report_with_issues(self, agent, mock_llm_client, sample_request):
        # Arrange
        mock_llm_client.generate.return_value = json.dumps(SAMPLE_BAD_RESPONSE)

        # Act
        response = agent.review_report(sample_request)

        # Assert
        assert len(response.issues) == 1
        assert response.issues[0].severity == QASeverity.CRITICAL

    def test_retry_on_malformed_json_success(self, agent, mock_llm_client, sample_request):
        # Arrange: Fail first, succeed second
        mock_llm_client.generate.side_effect = [
            "NOT JSON",
            json.dumps(SAMPLE_GOOD_RESPONSE)
        ]

        # Act
        response = agent.review_report(sample_request)

        # Assert
        assert mock_llm_client.generate.call_count == 2
        assert response.summary.overall_quality == "good"

    def test_failure_after_retries(self, agent, mock_llm_client, sample_request):
        # Arrange: Always fail
        mock_llm_client.generate.return_value = "NOT JSON"

        # Act & Assert
        with pytest.raises(LLMJsonParseError):
            agent.review_report(sample_request)

    def test_clean_json_blocks(self, agent):
        raw = "```json\n{\"foo\": \"bar\"}\n```"
        cleaned = agent._clean_json(raw)
        assert cleaned == "{\"foo\": \"bar\"}"
