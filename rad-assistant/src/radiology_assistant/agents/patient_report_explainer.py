"""
Patient Report Explainer Agent.

Translates radiology reports into patient-friendly language.
"""

import json
import logging
import time
from typing import Optional, List, Dict, Any

from ..config import Config
from ..models import (
    PatientReportSummaryRequest,
    PatientReportSummaryResponse,
    PatientReadingLevel,
    PatientSummaryTone,
    PatientNextStepUrgency,
    GlossaryItem,
    PatientNextStep,
    FollowUpInterval
)
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


PATIENT_EXPLAINER_SYSTEM_PROMPT = """You are a radiologist who is excellent at explaining imaging results to patients in simple, clear language.

Your job:
- Take a finalized radiology report and optional structured follow-up data.
- Generate a patient-friendly explanation and next steps WITHOUT changing the medical meaning.

Audience:
- Patients and non-medical readers.
- Reading level is specified in the input (e.g., very_simple, simple, standard).
- The text may be shown directly in a patient portal.

Key rules:

1. DO NOT change clinical meaning.
- You can simplify wording, but you must not add new diagnoses, probabilities, or treatments that are not supported by the report.
- Do not contradict the report.
- Do not make promises or guarantees (e.g., "this is definitely nothing to worry about").

2. DO NOT give new medical advice.
- You may restate the follow-up recommendations from the report in simpler language.
- You must not invent new tests, treatments, or timelines that are not clearly present in the report or follow-up data.
- If something is uncertain or “could represent X or Y”, say that it is uncertain in simple terms.

3. Explain clearly and kindly.
- Avoid medical jargon where possible. When you must use a medical word, provide a short explanation in the glossary.
- Keep sentences relatively short and direct.
- Respect the specified reading_level:
  - VERY_SIMPLE: aim around 4th–5th grade.
  - SIMPLE: aim around 7th–8th grade.
  - STANDARD: high school level, but still free of unnecessary jargon.
- Tone:
  - NEUTRAL: calm, factual.
  - REASSURING: calm, factual, but gently reassuring when appropriate (without overpromising).

4. Use follow-up data when provided.
- If structured follow-up data is provided (incidental findings, follow-up intervals, etc.), use it to create clear "next steps" for the patient (e.g., "Your doctor may want a follow-up CT in about 6 months to recheck this spot.").
- If no follow-up data is provided, infer next steps only if they are explicitly stated in the report.

5. Never mention AI.
- Do not mention artificial intelligence, large language models, or that an AI is generating this explanation.

Output format:
Return ONLY a single JSON object with this structure:

{
  "version": "v1",
  "patient_summary_text": "<short paragraph summary in patient-friendly language>",
  "key_points": [
    "<bullet-style key takeaway 1>",
    "<bullet-style key takeaway 2>"
    // ...
  ],
  "next_steps": [
    {
      "description": "<patient-friendly description of what may happen next>",
      "urgency": "routine" | "soon" | "urgent" | "unknown",
      "followup_interval": {
        "years": <int>,
        "months": <int>,
        "weeks": <int>
      } or null,
      "source_finding_id": "<ID of related finding from follow-up data>" or null
    }
    // ... more steps
  ],
  "glossary": [
    {
      "term": "<medical term>",
      "explanation": "<simple explanation>"
    }
    // ... more terms as needed
  ],
  "original_report_text": "<the original radiology report text exactly as provided>"
}

Rules for JSON:
- The JSON must be syntactically valid.
- No extra top-level keys.
- No comments, no trailing commas.
- If there are no next steps, use an empty array.
- If there are no glossary terms, use an empty array.
"""


class LLMJsonParseError(Exception):
    """Raised when LLM output cannot be parsed as JSON even after retries."""
    pass


class PatientReportExplainerAgent:
    def __init__(self, llm_client: LLMClient, logger: Optional[logging.Logger] = None):
        self._llm = llm_client
        self._logger = logger or logging.getLogger(__name__)
        self._system_prompt = PATIENT_EXPLAINER_SYSTEM_PROMPT

    def _build_user_prompt(self, request: PatientReportSummaryRequest) -> str:
        """Constructs the JSON input string for the LLM."""
        
        followup_dict = None
        if request.followup_data:
            followup_dict = request.followup_data.model_dump()

        input_data = {
            "exam_metadata": {
                "accession": request.exam_metadata.accession,
                "exam_date": str(request.exam_metadata.exam_date) if request.exam_metadata.exam_date else None,
                "modality": request.exam_metadata.modality,
                "body_region": request.exam_metadata.body_region,
                "age": request.exam_metadata.age,
                "sex": request.exam_metadata.sex
            },
            "report_text": request.report_text,
            "followup_data": followup_dict,
            "reading_level": request.reading_level,
            "tone": request.tone,
            "language_code": request.language_code
        }

        # Use json.dumps to handle escaping correctly
        input_json_str = json.dumps(input_data, indent=2)

        return f"""You will receive a JSON object with the exam metadata, the full radiology report text, and optional structured follow-up data.

Use it to generate a patient-friendly explanation according to the system instructions. Respond ONLY with the JSON structure described there.

Input JSON:
{input_json_str}
"""

    def _clean_json(self, text: str) -> str:
        """Clean code blocks from LLM response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text

    def explain(self, request: PatientReportSummaryRequest) -> PatientReportSummaryResponse:
        """
        Generate a patient-friendly explanation of the radiology report.
        """
        start = time.monotonic()
        
        user_prompt = self._build_user_prompt(request)

        attempts = 0
        max_retries = 1
        last_error = None
        
        while attempts <= max_retries:
            attempts += 1
            try:
                # Add retry hint if this is a retry
                current_user_prompt = user_prompt
                if attempts > 1:
                    current_user_prompt += "\n\nError: Invalid JSON returned previously. Please ensure valid JSON format only."

                raw_response = self._llm.generate(
                    system_prompt=self._system_prompt,
                    prompt=current_user_prompt,
                    temperature=0.2,
                    max_tokens=2048,
                )

                cleaned_response = self._clean_json(raw_response)
                payload = json.loads(cleaned_response)
                
                # Validation handled by Pydantic model
                response = PatientReportSummaryResponse(**payload)

                duration = time.monotonic() - start
                
                # Observability log (no PHI)
                self._logger.info(
                    "patient_explainer_completed",
                    extra={
                        "latency_ms": int(duration * 1000),
                        "model": getattr(self._llm, "model_name", "unknown"),
                        "language_code": request.language_code,
                        "num_next_steps": len(response.next_steps),
                    },
                )

                return response

            except (json.JSONDecodeError, ValueError) as e:
                self._logger.warning(f"Failed to parse Patient Summarizer response (attempt {attempts}): {e}")
                last_error = e
        
        self._logger.error("Failed to parse Patient Summarizer response after retries.")
        raise LLMJsonParseError(f"Failed to parse response: {last_error}")
