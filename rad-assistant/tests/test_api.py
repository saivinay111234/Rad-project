import json
from unittest.mock import Mock

from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import radiology_assistant.api as api
from radiology_assistant.models import ReportDraft


def test_health_endpoint_returns_ok():
    client = TestClient(api.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("status") == "ok"


def test_draft_report_endpoint_success(monkeypatch):
    # Prepare a mock agent that returns a known ReportDraft
    mock_agent = Mock()
    mock_report = ReportDraft(
        report_text="Mock report text...",
        key_findings=[],
        used_cv_signals=[],
        confidence_score=0.85,
    )
    mock_agent.draft_report.return_value = mock_report

    # Patch the module-level _agent so the API uses the mock
    monkeypatch.setattr(api, "_agent", mock_agent)

    client = TestClient(api.app)

    payload = {
        "findings": [
            {
                "location": "right lower lobe",
                "type": "opacity",
                "severity": "moderate",
                "additional_details": "Air bronchograms noted",
            }
        ],
        "clinical_context": {
            "patient_info": "65-year-old male",
            "clinical_presentation": "Fever and cough for 3 days",
            "relevant_history": "History of COPD"
        },
        "modality": "Chest X-ray",
        "view": "PA"
    }

    resp = client.post("/draft_report", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    # Ensure response contains the expected keys
    assert "report_text" in data
    assert "key_findings" in data
    assert "used_cv_signals" in data
    assert "confidence_score" in data


def test_cv_highlight_endpoint_success(monkeypatch):
    from radiology_assistant.models import CVHighlightResult, CVRegionHighlight
    
    # Mock the CV agent
    mock_agent = Mock()
    mock_result = CVHighlightResult(
        modality="DX",
        summary="Mock summary",
        regions=[
            CVRegionHighlight(label="opacity", score=0.8, bbox=(10, 10, 50, 50))
        ],
        heatmap_png_base64="mockbase64"
    )
    mock_agent.highlight.return_value = mock_result
    
    # Patch the module-level _cv_agent
    monkeypatch.setattr(api, "_cv_agent", mock_agent)
    
    client = TestClient(api.app)
    
    # Create dummy image bytes
    import io
    from PIL import Image
    img = Image.new('L', (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    # Send request
    files = {"file": ("test.png", image_bytes, "image/png")}
    params = {"modality": "DX"}
    
    resp = client.post("/cv/highlight", params=params, files=files)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    
    assert data["modality"] == "DX"
    assert data["summary"] == "Mock summary"
    assert len(data["regions"]) == 1
    assert data["regions"][0]["label"] == "opacity"
    assert data["heatmap_png_base64"] == "mockbase64"


def test_followup_extraction_endpoint_success(monkeypatch):
    from radiology_assistant.models import (
        FollowUpExtractionResponse, IncidentalFinding, IncidentalFindingCategory, 
        FollowUpType, RecommendationStrength
    )
    
    # Mock the follow-up agent
    mock_agent = Mock()
    mock_response = FollowUpExtractionResponse(
        incidental_findings=[
            IncidentalFinding(
                id="IF1",
                description="Test nodule",
                verbatim_snippet="snippet",
                category=IncidentalFindingCategory.pulmonary_nodule,
                followup_required=True,
                followup_type=FollowUpType.imaging,
                recommendation_strength=RecommendationStrength.explicit
            )
        ],
        has_any_followup=True
    )
    mock_agent.extract_followups.return_value = mock_response
    
    # Patch the module-level _followup_agent
    monkeypatch.setattr(api, "_followup_agent", mock_agent)
    
    client = TestClient(api.app)
    
    payload = {
        "exam_metadata": {"modality": "CT"},
        "report_text": "Sample report text"
    }
    
    resp = client.post("/v1/reports/extract_followups", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    
    assert data["has_any_followup"] is True
    assert len(data["incidental_findings"]) == 1
    assert data["incidental_findings"][0]["id"] == "IF1"


def test_report_qa_endpoint_success(monkeypatch):
    from radiology_assistant.models import (
        ReportQAResponse, QASummary
    )
    
    msg = ReportQAResponse(
        original_report_text="text",
        issues=[],
        summary=QASummary(
            overall_quality="good",
            num_critical=0, num_major=0, num_minor=0,
            comments="None"
        )
    )
    
    mock_agent = Mock()
    mock_agent.review_report.return_value = msg
    
    monkeypatch.setattr(api, "_qa_agent", mock_agent)
    
    client = TestClient(api.app)
    resp = client.post("/v1/reports/qa", json={
        "exam_metadata": {"modality": "XR"},
        "report_text": "Sample"
    })
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["summary"]["overall_quality"] == "good"

def test_report_qa_parse_error_502(monkeypatch):
    from radiology_assistant.agents.report_qa_agent import LLMJsonParseError
    
    mock_agent = Mock()
    mock_agent.review_report.side_effect = LLMJsonParseError("fail")
    
    monkeypatch.setattr(api, "_qa_agent", mock_agent)
    
    client = TestClient(api.app)
    resp = client.post("/v1/reports/qa", json={
        "exam_metadata": {"modality": "XR"},
        "report_text": "Sample"
    })
    
    assert resp.status_code == 502
    assert "Failed to parse" in resp.json()["detail"]


def test_patient_summary_endpoint_success(monkeypatch):
    from radiology_assistant.models import PatientReportSummaryResponse
    
    msg = PatientReportSummaryResponse(
        version="v1",
        patient_summary_text="Summary",
        key_points=[],
        next_steps=[],
        glossary=[],
        original_report_text="orig"
    )
    
    mock_agent = Mock()
    mock_agent.explain.return_value = msg
    
    monkeypatch.setattr(api, "_patient_explainer_agent", mock_agent)
    
    client = TestClient(api.app)
    resp = client.post("/v1/reports/patient_summary", json={
        "exam_metadata": {"modality": "XR"},
        "report_text": "Sample",
        "reading_level": "simple",
        "tone": "neutral"
    })
    
    assert resp.status_code == 200
    data = resp.json()
    assert data["patient_summary_text"] == "Summary"

def test_patient_summary_error_502(monkeypatch):
    from radiology_assistant.agents.patient_report_explainer import LLMJsonParseError
    
    mock_agent = Mock()
    mock_agent.explain.side_effect = LLMJsonParseError("fail")
    
    monkeypatch.setattr(api, "_patient_explainer_agent", mock_agent)
    
    client = TestClient(api.app)
    resp = client.post("/v1/reports/patient_summary", json={
        "exam_metadata": {"modality": "XR"},
        "report_text": "Sample"
    })
    
    assert resp.status_code == 502


def test_worklist_triage_endpoint_success(monkeypatch):
    from radiology_assistant.models import (
        WorklistTriageResponse, WorklistTriageItem, TriageLabel
    )
    
    mock_agent = Mock()
    mock_agent.triage.return_value = WorklistTriageResponse(
        version="v1",
        items=[
            WorklistTriageItem(
                study_id="123",
                triage_score=0.9,
                triage_label=TriageLabel.CRITICAL,
                reasons=[],
                model_metadata={}
            )
        ]
    )
    
    monkeypatch.setattr(api, "_triage_agent", mock_agent)
    
    client = TestClient(api.app)
    # The actual payload sent to the API
    resp = client.post("/v1/worklist/triage", json={
        "worklist_items": [
            {
                "study_id": "123",
                "modality": "XR",
                "body_region": "Chest"
            }
        ]
    })
    
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["version"] == "v1"
    assert len(data["items"]) == 1
    assert data["items"][0]["study_id"] == "123"
    assert data["items"][0]["triage_label"] == "CRITICAL"


def test_learning_digest_endpoint(monkeypatch):
    from radiology_assistant.models import RadiologistLearningDigestRequest
    
    # We don't need to patch _learning_agent because api.py mocks the repo by default
    # But let's act as a client
    client = TestClient(api.app)
    
    req = RadiologistLearningDigestRequest(
        radiologist_id="rad_123",
        start_date="2023-01-01",
        end_date="2023-01-07"
    )
    
    # We expect 200 OK with empty digest (since Mock Repo in api.py returns empty)
    resp = client.post("/v1/learning/radiologist_digest", json=req.model_dump())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["radiologist_id"] == "rad_123"
    assert "No significant learning events" in data["summary_text"]
    assert data["version"] == "v1"

