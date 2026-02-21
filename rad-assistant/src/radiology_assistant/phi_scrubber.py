"""
PHI (Protected Health Information) De-identification for the Radiology Assistant.

Scrubs common PHI patterns from text before transmitting to external LLM APIs.
Uses regex-based replacement with placeholder tokens. This is a heuristic
approach — not a certified de-identification tool — but provides a meaningful
safety layer against accidental PHI leakage.

Important: For full HIPAA Safe Harbor compliance, use a certified NLP de-id
tool (e.g., Microsoft Presidio, Stanford DeID, or AWS Comprehend Medical).
"""

import re
import logging
import copy
from typing import Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PHI Pattern Library
# ---------------------------------------------------------------------------

# Each pattern is (compiled_regex, replacement_token)
_PHI_PATTERNS: list[tuple[re.Pattern, str]] = [
    # ------- Dates -------
    # Full dates: 01/15/2024, 01-15-2024, Jan 15 2024, January 15, 2024
    (re.compile(
        r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b',
        re.IGNORECASE
    ), '[DATE]'),
    # Numeric dates: MM/DD/YYYY, MM-DD-YYYY, YYYY-MM-DD
    (re.compile(r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b'), '[DATE]'),
    (re.compile(r'\b\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\b'), '[DATE]'),
    # DOB / Age when combined with DOB
    (re.compile(r'\b(?:DOB|D\.O\.B|Date of Birth|date of birth)[:\s]*[\d/\-\.]+', re.IGNORECASE), '[DOB]'),

    # ------- Patient IDs / MRN / Accession -------
    # MRN pattern (various formats)
    (re.compile(r'\bMRN[:\s#]*[A-Z0-9\-]{4,20}\b', re.IGNORECASE), '[MRN]'),
    # Accession numbers (e.g., ACC-20240115-001, XR12345678)
    (re.compile(r'\b(?:ACC|Accession)[:\s#\-]*[A-Z0-9\-]{4,20}\b', re.IGNORECASE), '[ACCESSION]'),
    # Patient ID label patterns
    (re.compile(r'\b(?:Patient\s+ID|Pat\.?\s*ID|Pt\.\s*ID)[:\s]*[A-Z0-9\-]{4,20}\b', re.IGNORECASE), '[PATIENT_ID]'),

    # ------- Phone Numbers -------
    # (123) 456-7890, 123-456-7890, 1234567890, +1-123-456-7890
    (re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b'), '[PHONE]'),

    # ------- Social Security Numbers -------
    (re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'), '[SSN]'),

    # ------- Email Addresses -------
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),

    # ------- Physical Addresses -------
    # Street address: 123 Main St, 456 Oak Avenue Apt 7
    (re.compile(
        r'\b\d{1,5}\s+[A-Za-z\s]{2,30}(?:St|St\.|Ave|Ave\.|Blvd|Blvd\.|Dr|Dr\.|Rd|Rd\.|Ln|Ln\.|Way|Ct|Ct\.|Pl)'
        r'(?:\s+(?:Apt|Suite|Ste|Unit|#)\s*\w+)?\b',
        re.IGNORECASE
    ), '[ADDRESS]'),

    # ------- Names -------
    # "Patient: John Smith", "Referring: Dr. Jane Doe"
    (re.compile(
        r'\b(?:Patient(?:\s+Name)?|Pt\.?|Referring(?:\s+physician)?|Attending'
        r'|Radiologist|Dr\.?|Doctor|Physician)[:\s]+(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b',
        re.IGNORECASE
    ), '[PROVIDER_OR_PATIENT]'),
]


# ---------------------------------------------------------------------------
# Scrubber Class
# ---------------------------------------------------------------------------

class PHIScrubber:
    """
    Regex-based PHI scrubber for radiology report text.

    Applies a layered set of regex patterns to replace common PHI tokens
    with safe placeholder strings before content is sent to external APIs.

    Usage:
        scrubber = PHIScrubber()
        safe_text = scrubber.scrub("Patient: John Doe, DOB: 01/01/1980")
        # => "Patient: [PROVIDER_OR_PATIENT], DOB: [DOB]"
    """

    def __init__(self):
        self._patterns = _PHI_PATTERNS

    def scrub(self, text: str) -> str:
        """
        Replace PHI tokens in a string with placeholder labels.

        Args:
            text: Raw text that may contain PHI.

        Returns:
            De-identified text with PHI replaced by [TOKEN] placeholders.
        """
        if not text:
            return text
        result = text
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        if result != text:
            logger.debug("PHI scrubber made replacements in text (len_before=%d, len_after=%d)", len(text), len(result))
        return result

    def scrub_dict(self, data: dict) -> dict:
        """
        Recursively scrub all string values in a dictionary.
        Useful for scrubbing request payloads.

        Args:
            data: Dict potentially containing PHI in string values.

        Returns:
            Deep copy of dict with all string values scrubbed.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self.scrub(value)
            elif isinstance(value, dict):
                result[key] = self.scrub_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    self.scrub(item) if isinstance(item, str)
                    else (self.scrub_dict(item) if isinstance(item, dict) else item)
                    for item in value
                ]
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_default_scrubber: Optional[PHIScrubber] = None


def get_phi_scrubber() -> PHIScrubber:
    """Return a module-level singleton PHIScrubber."""
    global _default_scrubber
    if _default_scrubber is None:
        _default_scrubber = PHIScrubber()
    return _default_scrubber
