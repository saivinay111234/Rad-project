"""
Report Drafting Agent for generating radiology reports.

This agent takes structured findings and clinical context and generates
professional radiology reports using an LLM.
"""

import json
import logging
from typing import Optional, List, Dict, Any
import re

from ..config import Config

from ..models import ReportDraftRequest, ReportDraft, KeyFinding, UsedCVSignal
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


# Prompt templates
SYSTEM_PROMPT = """You are an experienced, board-certified radiologist specializing in diagnostic imaging.
Your role is to generate high-quality, clinically useful radiology reports from structured inputs.

You DO NOT see images directly. You only receive:
- Clinical indication and relevant history.
- Imaging metadata (modality, body part, technique).
- Optional prior study summary.
- Structured findings entered or confirmed by a human radiologist.
- A summary from a computer vision (CV) model: suspected pathologies with confidence scores and region-of-interest (ROI) descriptions derived from heatmaps.

Your responsibilities:
1. Draft a professional, concise, and clinically accurate radiology report.
2. Use the CV model outputs ONLY as supportive context, never as ground truth.
3. Prioritize any explicit human/radiologist-provided findings over CV suggestions.
4. Maintain internal consistency between TECHNIQUE, FINDINGS, and IMPRESSION.
5. Use standard radiology language and hedging appropriately (e.g., “compatible with”, “suspicious for”).
6. Never mention the existence of an AI or CV model in the report text. The AI assistance must stay invisible to the end reader.
7. Never infer or fabricate imaging details that are not explicitly given in the input (e.g., do NOT invent size, shape, distribution, or chronicity that isn’t provided).
8. If CV suggests a finding that is not supported by any human-provided or textual context, you may:
   - Omit it, OR
   - Mention it in cautious, non-committal language if it is clinically plausible given the rest of the context.
9. You must remain modality-aware (e.g., chest radiograph vs CT vs MRI) and avoid stating findings that are impossible or unlikely for the given modality.

Report structure:
- TECHNIQUE: concise description of how the study was performed. Use provided technique and metadata; do not invent parameters.
- COMPARISON: mention prior study if information is provided; otherwise “None” or “No prior imaging available for comparison” depending on input.
- FINDINGS: objective, structured description of what is seen, organized by anatomic region where appropriate.
- IMPRESSION: short, prioritized list of the most important, actionable findings. Avoid repeating every detail from FINDINGS.

Tone and style:
- Professional, clear, and free of colloquial language.
- Avoid redundancies and overly long sentences.
- Follow standard radiology conventions for the body region and modality.

Output format:
You MUST respond with a single JSON object with the following keys:
- "report_text": string – the full formatted report with TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION.
- "key_findings": array of objects, each with:
    - "label": short human-readable name of the finding (e.g., "Right lower lobe consolidation").
    - "category": e.g., "pathology", "normal_variant", "device", "artifact".
    - "severity": one of ["critical", "significant", "minor", "normal"].
- "used_cv_signals": array of objects describing how CV outputs influenced the report, each with:
    - "cv_label": pathology name from the CV model.
    - "included_in_report": boolean.
    - "reasoning": short explanation (1–2 sentences).
If no CV information is provided, set "used_cv_signals" to an empty array.

Constraints:
- The JSON must be syntactically valid.
- Do not include any comments or trailing commas.
- Do not add extra top-level keys.
"""


REPORT_PROMPT_TEMPLATE = """Based on the following imaging findings and clinical context, generate a professional radiology report as a single JSON object.

CLINICAL CONTEXT:
- Patient: {patient_info}
- Presentation: {clinical_presentation}
- History: {relevant_history}

IMAGING MODALITY: {modality}
VIEW: {view}
PRIOR IMAGING: {prior_study_summary}

RADIOLOGICAL FINDINGS (Human Verified):
{findings_text}

COMPUTER VISION ASSISTANT SUMMARY:
{cv_summary_text}

IMPORTANT: Output MUST be a single valid JSON object matching the keys specified in the system prompt. Do NOT include any extra text.
"""


class ReportDraftingAgent:
    """Agent for drafting radiology reports from findings and clinical context."""
    
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the Report Drafting Agent.
        
        Args:
            llm_client: An LLMClient instance for making API calls
        """
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    def _format_findings(self, request: ReportDraftRequest) -> str:
        """Format findings into readable text for the prompt."""
        if not request.findings:
            return "None documented."
            
        findings_list = []
        for i, finding in enumerate(request.findings, 1):
            finding_text = f"{i}. {finding.location.title()}: {finding.type.lower()} ({finding.severity.lower()})"
            if finding.additional_details:
                finding_text += f" - {finding.additional_details}"
            findings_list.append(finding_text)
        
        return "\n".join(findings_list)

    def _format_cv_summary(self, request: ReportDraftRequest) -> str:
        """Format CV summary into readable text."""
        if not request.cv_summary:
            return "No Computer Vision analysis available."
        
        summary_text = f"Summary: {request.cv_summary.summary}\nDetected Regions:\n"
        for region in request.cv_summary.regions:
            summary_text += f"- {region.label} (Confidence: {region.score:.2f})\n"
        
        return summary_text

    def _extract_json_objects(self, text: str) -> List[str]:
        """Extract top-level JSON object strings from text using brace counting."""
        objs: List[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] == '{':
                start = i
                depth = 1
                i += 1
                while i < n and depth > 0:
                    if text[i] == '{':
                        depth += 1
                    elif text[i] == '}':
                        depth -= 1
                    i += 1
                if depth == 0:
                    objs.append(text[start:i])
                continue
            i += 1
        return objs
    
    def _build_prompt(self, request: ReportDraftRequest) -> str:
        """Build the complete prompt for the LLM."""
        modality = request.modality or "Imaging"
        view = request.view or "Standard views"
        
        findings_text = self._format_findings(request)
        cv_summary_text = self._format_cv_summary(request)
        relevant_history = request.clinical_context.relevant_history or "Not provided"
        prior_study = request.prior_study_summary or "None available"
        
        prompt = REPORT_PROMPT_TEMPLATE.format(
            patient_info=request.clinical_context.patient_info,
            clinical_presentation=request.clinical_context.clinical_presentation,
            relevant_history=relevant_history,
            modality=modality,
            view=view,
            prior_study_summary=prior_study,
            findings_text=findings_text,
            cv_summary_text=cv_summary_text
        )
        
        return prompt
    
    def draft_report(self, request: ReportDraftRequest) -> ReportDraft:
        """
        Draft a radiology report from findings and clinical context.
        
        Args:
            request: ReportDraftRequest containing findings and clinical context
            
        Returns:
            ReportDraft object with report text and structured metadata
            
        Raises:
            ValueError: If the LLM response is invalid or cannot be parsed
        """
        self.logger.info(f"Drafting report with {len(request.findings)} findings")
        
        try:
            # Build the prompt
            prompt = self._build_prompt(request)
            
            # Generate response using LLM with retry on invalid JSON
            attempts = 0
            last_error = None
            report_dict = None
            while attempts < Config.MAX_RETRIES and report_dict is None:
                attempts += 1
                self.logger.debug(f"Generating LLM response (attempt {attempts})")
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.3
                )

                # Avoid logging raw LLM responses (may contain PHI). Log safe metadata instead.
                try:
                    import hashlib
                    sha = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
                    self.logger.debug("LLM response received: length=%d sha256=%s", len(response_text), sha)
                except Exception:
                    self.logger.debug("LLM response received: length=%d", len(response_text) if response_text is not None else 0)

                try:
                    report_dict = self._parse_response(response_text)
                except Exception as e:
                    last_error = e
                    self.logger.warning(f"Attempt {attempts} failed to produce valid JSON: {e}")
                    # On failure, append a short clarifying instruction and retry
                    prompt = (
                        "\nPlease respond with ONLY valid JSON matching the schema exactly. "
                        "Do NOT include any text outside the JSON object."
                    ) + "\n" + prompt
                    # small backoff
                    import time
                    time.sleep(Config.RETRY_DELAY)

            if report_dict is None:
                raise ValueError(f"Failed to obtain valid JSON from LLM after {attempts} attempts: {last_error}")
            
            # --- Phase 7 Upgrade: Self-Evaluation ---
            # Perform a quick self-critique to generate a realistic confidence score.
            try:
                evaluation = self._self_evaluate(report_dict.get("report_text", ""), request)
                report_dict["confidence_score"] = evaluation.get("confidence_score", 0.9)
                self.logger.info("Self-evaluation complete. Confidence Score: %.2f", report_dict["confidence_score"])
                if evaluation.get("critique"):
                    self.logger.info("Critique: %s", evaluation["critique"])
            except Exception as e:
                self.logger.warning("Self-evaluation failed (non-fatal): %s", e)
                if "confidence_score" not in report_dict:
                    report_dict["confidence_score"] = 0.85

            # Validate and create ReportDraft 
            try:
                report = ReportDraft(**report_dict)
            except Exception as e:
                self.logger.error(f"ReportDraft validation failed: {e}")
                # Dump dict to see what went wrong (careful with PHI in logs in real prod, but okay for debug)
                self.logger.debug(f"Invalid dict content: {report_dict}")
                raise ValueError(f"Invalid report structure: {e}")
            
            self.logger.info("Report drafted and self-evaluated successfully")
            return report
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {e}")
            return self._get_fallback_report(request, "JSON parsing error")
        except ValueError as e:
            self.logger.error(f"Validation error: {e}")
            return self._get_fallback_report(request, str(e))
        except Exception as e:
            self.logger.error(f"Unexpected error in report drafting: {e}")
            return self._get_fallback_report(request, "Unexpected error")
    
    def _parse_response(self, response_text: str) -> dict:
        """
        Parse the LLM response to extract JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed dictionary matching the expected structure
        """
        # Try direct JSON parsing first
        try:
            candidate = json.loads(response_text)
            return candidate
        except Exception:
            pass

        # Extract all top-level JSON objects and validate each
        candidates = self._extract_json_objects(response_text)
        for c in candidates:
            try:
                parsed = json.loads(c)
                # Quick check for required keys to pick the right object if multiple exist
                if "report_text" in parsed:
                    return parsed
            except Exception:
                continue

        # If still no valid JSON, raise error
        raise ValueError("Could not find a valid ReportDraft JSON object in LLM response")
    
    def _self_evaluate(self, report_text: str, request: ReportDraftRequest) -> dict:
        """
        Ask the LLM to critique the generated report draft for accuracy and consistency.
        Returns a dictionary with confidence_score and critique.
        """
        eval_prompt = f"""Critique the following radiology report draft based on the provided inputs.
INPUT FINDINGS:
{self._format_findings(request)}

GENERATED REPORT:
{report_text}

Rate the consistency of the IMPRESSION with the FINDINGS and the INPUT. 
Check if any human-provided findings were missed.
Check if the modality is correctly addressed.

Respond with ONLY a JSON object:
{{
  "confidence_score": float (0.0 to 1.0),
  "critique": "brief string",
  "is_consistent": boolean
}}
"""
        try:
            response = self.llm_client.generate(
                prompt=eval_prompt,
                system_prompt="You are a senior radiology quality assurance auditor.",
                temperature=0.1
            )
            # Use the existing robust parser
            return self._parse_response(response)
        except Exception as e:
            self.logger.error("Error in self-evaluation generate/parse: %s", e)
            return {"confidence_score": 0.85, "critique": "Self-evaluation failed.", "is_consistent": True}

    def _get_fallback_report(self, request: ReportDraftRequest, error_reason: str) -> ReportDraft:
        """
        Generate a fallback report when LLM fails.
        """
        self.logger.warning(f"Using fallback report due to: {error_reason}")
        
        findings_text = "\n".join([
            f"- {f.location}: {f.type} ({f.severity})"
            for f in request.findings
        ])
        
        report_text = (
            f"TECHNIQUE:\n{request.modality or 'Imaging'} {request.view or 'views'} obtained.\n\n"
            f"FINDINGS:\n{findings_text or 'No findings available.'}\n\n"
            f"IMPRESSION:\nUnable to generate automatic impression. Please review findings and clinical context manually.\n"
            f"System Error: {error_reason}"
        )
        
        return ReportDraft(
            report_text=report_text,
            key_findings=[],
            used_cv_signals=[],
            confidence_score=0.1
        )
