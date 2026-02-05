import pytest
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.agents.learning_feedback import LearningFeedbackAgent
from radiology_assistant.models import (
    RadiologistLearningDigestRequest, LearningEvent, LearningEventType, 
    DiscrepancySeverity, ExamMetadata
)
from radiology_assistant.llm_client import LLMClient

# Mock Repository
class MockRepo:
    def __init__(self, events):
        self.events = events
    def get_events(self, radiologist_id, start_date, end_date):
        return self.events

@pytest.fixture
def mock_llm_client():
    client = MagicMock(spec=LLMClient)
    # Default mock response
    client.generate.return_value = """
    {
        "summary_text": "Good work this week.",
        "key_themes": [],
        "cases": []
    }
    """
    return client

@pytest.fixture
def sample_event():
    return LearningEvent(
        event_id="evt_1",
        radiologist_id="rad_123",
        exam_metadata=ExamMetadata(modality="CT", body_region="Chest", exam_date="2023-01-01"),
        event_type=LearningEventType.QA_ISSUE,
        severity=DiscrepancySeverity.MINOR,
        source="qa",
        timestamp="2023-01-01T10:00:00",
        report_text_before="Original report",
        qa_issues=[]
    )

def test_generate_digest_happy_path(mock_llm_client, sample_event):
    repo = MockRepo([sample_event])
    agent = LearningFeedbackAgent(repo, mock_llm_client)
    
    req = RadiologistLearningDigestRequest(
        radiologist_id="rad_123", start_date="2023-01-01", end_date="2023-01-07"
    )
    
    resp = agent.generate_radiologist_digest(req)
    
    assert resp.radiologist_id == "rad_123"
    assert resp.stats.num_total_events == 1
    assert resp.cases[0].event_id == "evt_1"
    assert resp.version == "v1"

def test_generate_digest_empty(mock_llm_client):
    repo = MockRepo([])
    agent = LearningFeedbackAgent(repo, mock_llm_client)
    
    req = RadiologistLearningDigestRequest(
        radiologist_id="rad_123", start_date="2023-01-01", end_date="2023-01-07"
    )
    
    resp = agent.generate_radiologist_digest(req)
    
    assert resp.stats.num_total_events == 0
    assert "No significant learning events" in resp.summary_text
    # Ensure LLM was NOT called
    mock_llm_client.generate.assert_not_called()

def test_scoring_prioritizes_critical(mock_llm_client):
    evt_minor = LearningEvent(
        event_id="minor", radiologist_id="r", exam_metadata=ExamMetadata(modality="XR"),
        event_type=LearningEventType.QA_ISSUE, severity=DiscrepancySeverity.MINOR,
        source="qa", timestamp="2023-01-01"
    )
    evt_critical = LearningEvent(
        event_id="critical", radiologist_id="r", exam_metadata=ExamMetadata(modality="XR"),
        event_type=LearningEventType.QA_ISSUE, severity=DiscrepancySeverity.CRITICAL,
        source="qa", timestamp="2023-01-01"
    )
    
    repo = MockRepo([evt_minor, evt_critical])
    agent = LearningFeedbackAgent(repo, mock_llm_client)
    
    # Request max 1 case to see which wins
    req = RadiologistLearningDigestRequest(
        radiologist_id="r", start_date="2023-01-01", end_date="2023-01-07", max_cases=1
    )
    
    resp = agent.generate_radiologist_digest(req)
    
    assert len(resp.cases) == 1
    assert resp.cases[0].event_id == "critical"

def test_llm_failure_fallback(mock_llm_client, sample_event):
    repo = MockRepo([sample_event])
    agent = LearningFeedbackAgent(repo, mock_llm_client)
    
    # Simulate LLM exception
    mock_llm_client.generate.side_effect = Exception("LLM Down")
    
    req = RadiologistLearningDigestRequest(
        radiologist_id="rad_123", start_date="2023-01-01", end_date="2023-01-07"
    )
    
    resp = agent.generate_radiologist_digest(req)
    
    assert resp.generation_metadata["generation_method"] == "fallback"
    assert len(resp.cases) == 1
    assert "Review of 1 events" in resp.summary_text
