"""
Follow-Up Extractor Agent.

Parses minimized radiology reports to identify incidental findings and follow-up recommendations.
"""

import json
import logging
from typing import Optional, List, Dict, Any
import datetime

from ..config import Config
from ..models import (
    FollowUpExtractionRequest, 
    FollowUpExtractionResponse, 
    IncidentalFinding,
    FollowUpInterval
)
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are an information extraction system specialized in radiology reports.
Your only job is to identify incidental findings and follow-up recommendations from finalized radiology reports and convert them into structured JSON.

Definitions:
- "Incidental finding": a finding that is not the primary reason for the exam but may have clinical significance or require follow-up (e.g., pulmonary nodule, adrenal nodule, renal lesion, thyroid nodule, liver lesion, aortic aneurysm).
- "Follow-up recommendation": any explicit or implicit recommendation for additional imaging, clinical correlation, or specific timeframe for re-evaluation.

Input:
- A full radiology report as free text (including TECHNIQUE, FINDINGS, IMPRESSION).
- Optional metadata about exam date and patient demographics.

Your responsibilities:
1. Identify incidental findings (if any).
2. For each incidental finding, extract:
   - Anatomic location.
   - Lesion description (as close to verbatim as possible).
   - Size and key imaging features if provided (e.g., solid, cystic, calcified).
   - Whether follow-up is recommended.
   - The type of follow-up (imaging modality vs clinical).
   - Suggested interval (e.g., 3 months, 6–12 months).
3. Normalize follow-up intervals into a machine-usable form (e.g., "6 months" → {"months": 6}).
4. Distinguish between:
   - Explicit recommendations (e.g., "Recommend CT chest in 6–12 months").
   - Weak/hedged suggestions (e.g., "Consider ultrasound follow-up if clinically indicated").
5. If the report explicitly states that no follow-up is required for a known incidental finding, capture that as "followup_required": false with rationale.
6. Do NOT invent or guess follow-ups that are not clearly implied in the text.
7. If there are no incidental findings or follow-up recommendations, return empty arrays and appropriate flags.

Output format:
You MUST return a single JSON object with these keys:

- "incidental_findings": array of objects. Each object MUST have:
    - "id": string – a unique identifier for this finding within the report (e.g., "IF1", "IF2").
    - "description": string – short, human-readable summary of the finding.
    - "verbatim_snippet": string – close-to-verbatim excerpt from the report describing this finding.
    - "location": string – anatomic location (e.g., "right upper lobe", "left adrenal gland", "right kidney").
    - "size": string or null – as stated in the report (e.g., "6 mm", "2.3 cm x 1.5 cm"), or null if not specified.
    - "category": string – e.g., "pulmonary_nodule", "liver_lesion", "renal_cyst", "adrenal_nodule", "thyroid_nodule", "other".
    - "followup_required": boolean.
    - "followup_type": string or null – e.g., "imaging", "clinical", "none", or null if unclear.
    - "followup_modality": string or null – e.g., "CT chest", "ultrasound abdomen", "MRI liver", or null.
    - "followup_interval": object or null – time recommendation broken into fields:
        {
          "years": int or 0,
          "months": int or 0,
          "weeks": int or 0
        }
      If the report gives a range (e.g., "6–12 months"), choose the lower bound as the main value and record the full text in "followup_interval_text".
    - "followup_interval_text": string or null – the verbatim or near-verbatim interval phrase (e.g., "6–12 months").
    - "followup_rationale": string – brief explanation based on the report (e.g., "Small solid pulmonary nodule in high-risk patient").
    - "recommendation_strength": string – one of ["explicit", "conditional", "none"].
- "global_followup_comment": string or null – any global comments about follow-up from the report not tied to a single finding.
- "has_any_followup": boolean – true if ANY follow-up is recommended in the report.

Constraints:
- The JSON must be syntactically valid, with no trailing commas.
- Do not add any top-level keys other than those specified.
- If there are no incidental findings, "incidental_findings" MUST be an empty array.
- If no follow-up recommendations are present, set "has_any_followup" to false and "global_followup_comment" to null or a short explanation.

Important:
- You must not change clinical meaning or invent new recommendations.
- You must stay faithful to the original report text.
- Always extract from the IMPRESSION section first when both FINDINGS and IMPRESSION mention the same item; use IMPRESSION as the primary source for follow-up.
"""

USER_PROMPT_TEMPLATE = """Extract incidental findings and follow-up recommendations from the following radiology report.

Input JSON:

{{
  "exam_metadata": {{
    "accession": "{accession}",
    "exam_date": "{exam_date}",
    "modality": "{modality}",
    "body_region": "{body_region}",
    "age": {age},
    "sex": "{sex}"
  }},
  "report_text": "{report_text_escaped}"
}}

Remember:
- Only return the JSON structure described in the system instructions.
- Do not include any explanation outside the JSON.
"""

class FollowUpExtractorAgent:
    """Agent for extracting follow-up recommendations from radiology reports."""
    
    def __init__(self, llm_client: LLMClient, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

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

    def extract_followups(self, request: FollowUpExtractionRequest) -> FollowUpExtractionResponse:
        """
        Extract follow-up information from the report.
        """
        start_time = datetime.datetime.now()
        self.logger.info("Starting follow-up extraction.")

        # Escape report text for JSON safety in prompt
        report_text_escaped = json.dumps(request.report_text)[1:-1]  # remove quotes
        
        # Build prompt
        prompt = USER_PROMPT_TEMPLATE.format(
            accession=request.exam_metadata.accession or "null",
            exam_date=str(request.exam_metadata.exam_date) if request.exam_metadata.exam_date else "null",
            modality=request.exam_metadata.modality or "null",
            body_region=request.exam_metadata.body_region or "null",
            age=request.exam_metadata.age if request.exam_metadata.age is not None else "null",
            sex=request.exam_metadata.sex or "null",
            report_text_escaped=report_text_escaped
        )

        attempts = 0
        max_retries = 1  # As per plan, optional retry
        last_error = None
        
        while attempts <= max_retries:
            attempts += 1
            try:
                response_text = self.llm_client.generate(
                    prompt=prompt,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=0.0  # Strict
                )
                
                # Clean response
                cleaned_response = self._clean_json(response_text)
                data = json.loads(cleaned_response)
                
                # Parse with Pydantic
                response = FollowUpExtractionResponse.model_validate(data)
                
                latency = (datetime.datetime.now() - start_time).total_seconds()
                self.logger.info(
                    "Follow-up extraction successful.",
                    extra={
                        "findings_count": len(response.incidental_findings),
                        "has_followup": response.has_any_followup,
                        "latency": latency
                    }
                )
                return response

            except (json.JSONDecodeError, ValueError) as e:
                self.logger.warning(f"Failed to parse LLM response (attempt {attempts}): {e}")
                last_error = e
                # Setup retry prompt enhancement if needed
                if attempts <= max_retries:
                    prompt += "\n\nError: Invalid JSON returned. Please correct formatting and return valid JSON only."

        # If we failed after retries, return empty safe response or raise
        self.logger.error("Failed to extract follow-ups after retries.")
        # Returning empty response for safety as per plan suggestion 2
        return FollowUpExtractionResponse(
            incidental_findings=[],
            has_any_followup=False,
            global_followup_comment=f"Extraction failed: {str(last_error)}"
        )
