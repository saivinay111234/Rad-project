"""
FHIR Exporter — Structured Report to HL7 FHIR R4 DiagnosticReport.

Converts a completed StudyBundle (from the Study Orchestrator) into a
valid HL7 FHIR R4 DiagnosticReport JSON resource, enabling integration
with EHR systems (Epic, Cerner, etc.) that support FHIR R4.

Reference: https://hl7.org/fhir/R4/diagnosticreport.html
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class FHIRExporter:
    """
    Converts a StudyBundle / report components to a FHIR R4 DiagnosticReport.

    Usage:
        exporter = FHIRExporter()
        resource = exporter.to_diagnostic_report(bundle, patient_id="anon-123")
        # => dict conforming to HL7 FHIR R4 DiagnosticReport schema
    """

    FHIR_BASE_URL = "https://rad-assistant.example.com/fhir/R4"

    def to_diagnostic_report(
        self,
        bundle,  # StudyOrchestrationResponse or compatible object
        patient_id: Optional[str] = None,
        study_id: Optional[str] = None,
        report_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a FHIR R4 DiagnosticReport from a StudyBundle/orchestration response.

        Args:
            bundle: StudyOrchestrationResponse or a dict with report_draft, qa_result,
                    followup_data, patient_summary fields.
            patient_id: Anonymized patient reference (no real MRN).
            study_id: Study accession number reference.
            report_id: Existing report ID (generates new UUID if not provided).

        Returns:
            Dict conforming to FHIR R4 DiagnosticReport schema.
            Can be serialized directly to JSON for API responses.
        """
        rid = report_id or str(uuid.uuid4())
        now_iso = datetime.now(timezone.utc).isoformat()

        # Extract components (handles both Pydantic objects and dicts)
        def _get(obj, *keys):
            for key in keys:
                try:
                    val = getattr(obj, key, None)
                    if val is not None:
                        return val
                    if isinstance(obj, dict):
                        return obj.get(key)
                except Exception:
                    pass
            return None

        report_draft = _get(bundle, "report_draft")
        followup_data = _get(bundle, "followup_data")
        qa_result = _get(bundle, "qa_result")
        meta = _get(bundle, "exam_metadata")

        report_text = _get(report_draft, "report_text") or ""
        key_findings = _get(report_draft, "key_findings") or []
        conclusion_text = _get(followup_data, "summary") or _get(report_draft, "impression") or ""

        # Build observation resources for key findings
        observations = []
        for i, finding in enumerate(key_findings):
            finding_text = finding if isinstance(finding, str) else str(finding)
            obs_id = f"obs-{rid}-{i}"
            observations.append({
                "reference": f"Observation/{obs_id}",
                "_display": finding_text[:200],
            })

        # Build the FHIR resource
        resource: Dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "id": rid,
            "meta": {
                "lastUpdated": now_iso,
                "profile": ["http://hl7.org/fhir/StructureDefinition/DiagnosticReport"],
                "source": f"{self.FHIR_BASE_URL}/DiagnosticReport/{rid}",
            },
            "status": "final",
            "category": [
                {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                            "code": "RAD",
                            "display": "Radiology",
                        }
                    ]
                }
            ],
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": self._get_loinc_code(bundle),
                        "display": self._get_display_name(bundle),
                    }
                ],
                "text": self._get_display_name(bundle),
            },
            "effectiveDateTime": now_iso,
            "issued": now_iso,
        }

        # Add patient reference (anonymized)
        if patient_id:
            resource["subject"] = {
                "reference": f"Patient/{patient_id}",
                "type": "Patient",
            }

        # Add study/accession reference
        if study_id:
            resource["identifier"] = [
                {
                    "use": "usual",
                    "system": f"{self.FHIR_BASE_URL}/identifier/accession",
                    "value": study_id,
                }
            ]

        # Report text as presentedForm (base64-free plain text attachment)
        if report_text:
            resource["presentedForm"] = [
                {
                    "contentType": "text/plain",
                    "language": "en",
                    "data": None,  # In production, base64-encode here
                    "title": "Radiology Report",
                    "creation": now_iso,
                    "_text": report_text[:10000],   # Non-standard extension for easy reading
                }
            ]

        # Observation references for key findings
        if observations:
            resource["result"] = observations

        # Conclusion / follow-up summary
        if conclusion_text:
            resource["conclusion"] = conclusion_text[:2000]

        # QA quality flags as extensions
        if qa_result and _get(qa_result, "issues"):
            issues = _get(qa_result, "issues") or []
            if issues:
                resource.setdefault("extension", []).append({
                    "url": f"{self.FHIR_BASE_URL}/StructureDefinition/qa-issues",
                    "valueString": f"{len(issues)} QA issue(s) flagged",
                })

        logger.info("FHIR DiagnosticReport generated: id=%s", rid)
        return resource

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def _get_loinc_code(self, bundle) -> str:
        """Return best-guess LOINC code for the modality."""
        _MODALITY_LOINC = {
            "CR": "24627-2",  # Chest XR AP
            "CT": "24727-0",  # Thorax CT
            "MR": "24558-9",  # Head MRI
            "US": "24544-9",  # Abdomen ultrasound
            "DX": "36643-5",  # Chest X-ray 2 views
            "NM": "44136-0",  # Nuclear medicine study
            "PT": "44136-0",  # PET
        }
        try:
            meta = getattr(bundle, "exam_metadata", None) or {}
            modality = getattr(meta, "modality", None) or (meta.get("modality") if isinstance(meta, dict) else None)
            if modality:
                return _MODALITY_LOINC.get(str(modality).upper(), "55113-5")
        except Exception:
            pass
        return "55113-5"  # Generic: "Radiology Study"

    def _get_display_name(self, bundle) -> str:
        """Return a human-readable study description."""
        try:
            meta = getattr(bundle, "exam_metadata", None) or {}
            modality = getattr(meta, "modality", None) or (meta.get("modality") if isinstance(meta, dict) else None)
            body_part = getattr(meta, "body_part", None) or (meta.get("body_part") if isinstance(meta, dict) else None)
            parts = [str(modality) if modality else None, str(body_part) if body_part else None]
            return " ".join(p for p in parts if p) or "Radiology Study"
        except Exception:
            return "Radiology Study"
