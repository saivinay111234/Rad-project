"""
Unit tests for Agent 6: Worklist Triage.
"""

import pytest
from unittest.mock import MagicMock, patch
from radiology_assistant.models import (
    WorklistTriageRequest,
    WorklistItem,
    TriageLabel,
    TriageConfig,
    TriageThresholdConfig,
    ModalityGroup,
    TriageReasonType
)
from radiology_assistant.agents.worklist_triage import WorklistTriageAgent, CVRouter

@pytest.fixture
def mock_config():
    return TriageConfig(
        thresholds=[
            TriageThresholdConfig(
                modality_group=ModalityGroup.XR,
                body_region="Chest",
                critical_threshold=0.9,
                high_threshold=0.7,
                low_threshold=0.3
            )
        ],
        model_mapping={"XR/Chest": "mock_model"},
        enable_llm_explanation=True
    )

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.generate.return_value = "Mock explanation."
    return llm

@pytest.fixture
def triage_agent(mock_config, mock_llm):
    return WorklistTriageAgent(config=mock_config, llm_client=mock_llm)

def test_triage_happy_path(triage_agent):
    """Test standard triage with STAT priority boost."""
    # Mock CV router to return a mock agent
    mock_cv_agent = MagicMock()
    # Mock result with high score
    mock_result = MagicMock()
    mock_region = MagicMock()
    mock_region.score = 0.5
    mock_region.label = "Opacities"
    mock_result.regions = [mock_region]
    mock_cv_agent.highlight.return_value = mock_result
    
    triage_agent.cv_router.get_model_for_item = MagicMock(return_value=mock_cv_agent)
    
    # Input item: Score 0.5 (CV) -> boosted by STAT if applicable
    item = WorklistItem(
        study_id="123",
        modality="XR",
        body_region="Chest",
        priority_flag_from_order="STAT",
        image_reference={"thumbnail_path": "dummy.png"}
    )
    request = WorklistTriageRequest(worklist_items=[item])
    
    # We need to mock open() since the agent tries to read the file
    with patch("builtins.open", MagicMock()):
        response = triage_agent.triage(request)
        
    assert len(response.items) == 1
    t_item = response.items[0]
    assert t_item.study_id == "123"
    # STAT boosts score to min 0.8
    assert t_item.triage_score >= 0.8
    assert t_item.triage_label == TriageLabel.HIGH or t_item.triage_label == TriageLabel.CRITICAL
    
    # Check reasons
    reasons = [r.type for r in t_item.reasons]
    assert TriageReasonType.MODEL_PREDICTION in reasons
    assert TriageReasonType.CLINICAL_INDICATION in reasons

def test_triage_critical_cv(triage_agent):
    """Test critical CV score leads to CRITICAL label."""
    mock_cv_agent = MagicMock()
    mock_result = MagicMock()
    mock_region = MagicMock()
    mock_region.score = 0.95
    mock_region.label = "Pneumothorax"
    mock_result.regions = [mock_region]
    mock_cv_agent.highlight.return_value = mock_result
    
    triage_agent.cv_router.get_model_for_item = MagicMock(return_value=mock_cv_agent)
    
    item = WorklistItem(
        study_id="crit",
        modality="XR",
        body_region="Chest",
        image_reference={"thumbnail_path": "dummy.png"}
    )
    
    with patch("builtins.open", MagicMock()):
        response = triage_agent.triage(WorklistTriageRequest(worklist_items=[item]))
        
    t_item = response.items[0]
    assert t_item.triage_label == TriageLabel.CRITICAL
    assert t_item.triage_score >= 0.95

def test_missing_config_fallback(triage_agent):
    """Test item with no matching config."""
    item = WorklistItem(
        study_id="noconfig",
        modality="US", # No config for US
        body_region="Abdomen"
    )
    request = WorklistTriageRequest(worklist_items=[item])
    response = triage_agent.triage(request)
    
    t_item = response.items[0]
    assert t_item.triage_label == TriageLabel.UNTRIAGED
    assert t_item.error is not None

def test_fail_soft_cv_error(triage_agent):
    """Test that if CV model raises exception, item is still processed/returned with error."""
    mock_cv_agent = MagicMock()
    mock_cv_agent.highlight.side_effect = Exception("CV Model Failed")
    triage_agent.cv_router.get_model_for_item = MagicMock(return_value=mock_cv_agent)
    
    item = WorklistItem(
        study_id="failcv",
        modality="XR",
        body_region="Chest",
        image_reference={"thumbnail_path": "dummy.png"}
    )
    
    with patch("builtins.open", MagicMock()):
        response = triage_agent.triage(WorklistTriageRequest(worklist_items=[item]))
        
    t_item = response.items[0]
    # It might return LOW/ROUTINE (0 score) if logic just swallows exception and continues scoring based on 0 CV score
    # OR unhandled processing exception makes it UNTRIAGED.
    # In my implementation: "We continue without CV result" -> Score 0 -> Label LOW or ROUTINE (depending on threshold)
    # Thresholds: low=0.3. Score 0 < 0.3 -> LOW.
    assert t_item.study_id == "failcv"
    assert t_item.triage_label == TriageLabel.LOW or t_item.triage_label == TriageLabel.ROUTINE
    # Note: Logic captures specific CV fail as "continue without result", so no global error on item unless calc fails.
    assert t_item.error is None # Should proceed to calculation

def test_fail_soft_unexpected_error(triage_agent):
    """Test unexpected error during processing."""
    # Force error by making get_thresholds raise or something core
    triage_agent._get_thresholds = MagicMock(side_effect=Exception("Boom"))
    
    item = WorklistItem(study_id="boom", modality="XR", body_region="Chest")
    response = triage_agent.triage(WorklistTriageRequest(worklist_items=[item]))
    
    t_item = response.items[0]
    assert t_item.triage_label == TriageLabel.UNTRIAGED
    assert "Boom" in (t_item.error or "")

def test_llm_explanation_failure(triage_agent):
    """Test that LLM failure doesn't crash triage."""
    triage_agent.llm_client.generate.side_effect = Exception("LLM Error")
    
    item = WorklistItem(study_id="llmfail", modality="XR", body_region="Chest", priority_flag_from_order="STAT")
    # Will get STAT score -> High -> Try explanation
    
    response = triage_agent.triage(WorklistTriageRequest(worklist_items=[item]))
    
    t_item = response.items[0]
    assert t_item.explanation_text is None
    assert t_item.triage_label in [TriageLabel.HIGH, TriageLabel.CRITICAL]
