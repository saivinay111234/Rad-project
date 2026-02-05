"""
Unit tests for Report Drafting Agent.

Tests focus on Agent 1 behavior: input validation, report generation, and JSON output.
"""

import unittest
from unittest.mock import Mock
import json

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from radiology_assistant.models import (
    Finding, ClinicalContext, ReportDraftRequest, ReportDraft,
    KeyFinding, UsedCVSignal, CVHighlightResult, CVRegionHighlight, CVHighlightMode
)
from radiology_assistant.agents import ReportDraftingAgent
from radiology_assistant.llm_client import LLMClient


class TestReportDraftingAgent(unittest.TestCase):
    """Test cases for ReportDraftingAgent - Agent 1."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock LLM client
        self.mock_llm_client = Mock(spec=LLMClient)
        self.agent = ReportDraftingAgent(self.mock_llm_client)
        
        # Create sample request
        self.sample_findings = [
            Finding(
                location="right lower lobe",
                type="opacity",
                severity="moderate"
            ),
            Finding(
                location="cardiac silhouette",
                type="cardiomegaly",
                severity="mild"
            )
        ]
        
        self.sample_context = ClinicalContext(
            patient_info="65-year-old male",
            clinical_presentation="Fever and cough for 3 days",
            relevant_history="History of COPD"
        )
        
        # Create optional CV summary
        self.cv_summary = CVHighlightResult(
            modality="DX",
            summary="Opacity RLL",
            regions=[
                CVRegionHighlight(label="Opacity", score=0.88),
            ]
        )
        
        self.sample_request = ReportDraftRequest(
            findings=self.sample_findings,
            clinical_context=self.sample_context,
            modality="Chest X-ray",
            view="PA & lateral",
            cv_summary=self.cv_summary
        )
    
    # ========== Input Validation Tests ==========
    def test_report_draft_request_creation(self):
        """Test that ReportDraftRequest is properly created and validated."""
        request = self.sample_request
        
        self.assertEqual(len(request.findings), 2)
        self.assertEqual(request.clinical_context.patient_info, "65-year-old male")
        self.assertEqual(request.modality, "Chest X-ray")
        self.assertIsNotNone(request.cv_summary)
    
    # ========== Output Validation Tests ==========
    def test_report_draft_response_validation(self):
        """Test that ReportDraft enforces all required fields and confidence_score range."""
        report = ReportDraft(
            report_text="TECHNIQUE: ... FINDINGS: ... IMPRESSION: ...",
            key_findings=[
                KeyFinding(label="RLL Opacity", category="pathology", severity="significant")
            ],
            used_cv_signals=[
                UsedCVSignal(cv_label="Opacity", included_in_report=True, reasoning="Matched")
            ],
            confidence_score=0.85
        )
        
        # Verify all fields are present
        self.assertIsNotNone(report.report_text)
        self.assertEqual(len(report.key_findings), 1)
        self.assertEqual(len(report.used_cv_signals), 1)
        
        # Verify confidence_score is in valid range
        self.assertGreaterEqual(report.confidence_score, 0.0)
        self.assertLessEqual(report.confidence_score, 1.0)
    
    # ========== Agent Logic Tests ==========
    def test_agent_format_findings(self):
        """Test that findings are correctly formatted for prompt."""
        formatted = self.agent._format_findings(self.sample_request)
        
        # Check all finding information is present
        self.assertIn("right lower lobe", formatted.lower())
        self.assertIn("opacity", formatted.lower())
        self.assertIn("moderate", formatted.lower())
    
    def test_agent_format_cv_summary(self):
        """Test that CV summary is correctly formatted."""
        formatted = self.agent._format_cv_summary(self.sample_request)
        
        self.assertIn("Opacity RLL", formatted)
        self.assertIn("Opacity", formatted)
        self.assertIn("0.88", formatted)

    def test_agent_build_prompt(self):
        """Test that prompt includes all necessary information."""
        prompt = self.agent._build_prompt(self.sample_request)
        
        # Verify prompt contains clinical context
        self.assertIn("65-year-old male", prompt)
        self.assertIn("Fever and cough", prompt)
        
        # Verify CV summary inclusion
        self.assertIn("COMPUTER VISION ASSISTANT SUMMARY", prompt)
        self.assertIn("Opacity RLL", prompt)
    
    # ========== Report Generation Tests ==========
    def test_agent_draft_report_success(self):
        """Test successful report generation with valid LLM response."""
        # Mock LLM to return valid JSON
        mock_response = json.dumps({
            "report_text": "TECHNIQUE: Standard. FINDINGS: RLL Opacity. IMPRESSION: Pneumonia.",
            "key_findings": [
                {"label": "RLL Opacity", "category": "pathology", "severity": "significant"}
            ],
            "used_cv_signals": [
                {"cv_label": "Opacity", "included_in_report": True, "reasoning": "Consistent"}
            ],
            "confidence_score": 0.87
        })
        
        self.mock_llm_client.generate.return_value = mock_response
        
        # Generate report
        report = self.agent.draft_report(self.sample_request)
        
        # Validate output is ReportDraft
        self.assertIsInstance(report, ReportDraft)
        
        # Validate content
        self.assertIn("Pneumonia", report.report_text)
        self.assertEqual(report.confidence_score, 0.87)
        self.assertEqual(report.key_findings[0].label, "RLL Opacity")
    
    def test_agent_draft_report_fallback_on_invalid_json(self):
        """Test fallback report when JSON parsing fails completely."""
        # Mock LLM to return invalid response
        self.mock_llm_client.generate.return_value = "This is not valid JSON"
        
        # Generate report - should fallback gracefully
        report = self.agent.draft_report(self.sample_request)
        
        # Validate fallback report is returned
        self.assertIsInstance(report, ReportDraft)
        self.assertIn("Unable to generate", report.report_text)
        
        # Fallback should have low confidence
        self.assertLess(report.confidence_score, 0.2)
        self.assertEqual(len(report.key_findings), 0)


if __name__ == '__main__':
    unittest.main()
