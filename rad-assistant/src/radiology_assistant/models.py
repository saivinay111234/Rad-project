"""
Data models for the radiology assistant.

Defines the contract between components using Pydantic models.
"""

from enum import Enum
from typing import Optional, List, Tuple, Dict, Any
from pydantic import BaseModel, Field


class Finding(BaseModel):
    """Represents a single radiological finding."""
    
    location: str = Field(..., description="Anatomical location of the finding (e.g., 'right lower lobe')")
    type: str = Field(..., description="Type of finding (e.g., 'opacity', 'cardiomegaly', 'consolidation')")
    severity: str = Field(..., description="Severity level (e.g., 'mild', 'moderate', 'severe')")
    additional_details: Optional[str] = Field(None, description="Optional additional clinical details")

    class Config:
        json_schema_extra = {
            "example": {
                "location": "right lower lobe",
                "type": "opacity",
                "severity": "moderate",
                "additional_details": None
            }
        }


class ClinicalContext(BaseModel):
    """Clinical context for the report."""
    
    patient_info: str = Field(..., description="Brief patient demographics and history (e.g., '65-year-old male')")
    clinical_presentation: str = Field(..., description="Chief complaint and recent symptoms")
    relevant_history: Optional[str] = Field(None, description="Relevant medical history (e.g., 'history of COPD')")

    class Config:
        json_schema_extra = {
            "example": {
                "patient_info": "65-year-old male",
                "clinical_presentation": "Fever and cough for 3 days",
                "relevant_history": "History of COPD"
            }
        }


class KeyFinding(BaseModel):
    """A key finding extracted for the report."""
    label: str = Field(..., description="Short human-readable name of the finding")
    category: str = Field(..., description="Category e.g. pathology, normal_variant, device, artifact")
    severity: str = Field(..., description="Severity: critical, significant, minor, normal")


class UsedCVSignal(BaseModel):
    """Description of how a CV signal was used in the report."""
    cv_label: str = Field(..., description="Pathology name from the CV model")
    included_in_report: bool = Field(..., description="Whether this signal was mentioned in the report")
    reasoning: str = Field(..., description="Short explanation for inclusion/exclusion")


class CVHighlightMode(str, Enum):
    attention = "attention"       # assistive, saliency-style
    bounding_boxes = "boxes"      # optional later


class CVHighlightRequest(BaseModel):
    """Request for CV highlighting."""
    study_id: Optional[str] = None      # internal id, not required
    modality: str                       # e.g. "CR", "DX", "CT"
    body_part: Optional[str] = None     # "chest", "abdomen", etc.
    view: Optional[str] = None          # "PA", "AP", etc.
    assistive_mode: CVHighlightMode = CVHighlightMode.attention
    # NOTE: image file itself will come via FastAPI UploadFile, not in this model


class CVRegionHighlight(BaseModel):
    """A specific highlighted region."""
    label: str                         # e.g. "opacity", "nodule"
    score: float                       # 0.0 – 1.0 confidence
    bbox: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
    mask_present: bool = False


class CVHighlightResult(BaseModel):
    """Result from the CV agent."""
    study_id: Optional[str] = None
    modality: str
    summary: str                       # short text summary
    regions: List[CVRegionHighlight]
    heatmap_png_base64: Optional[str] = None  # for UI overlay


# --- Agent 3: Follow-Up & Incidental Findings Tracker ---

class FollowUpInterval(BaseModel):
    """Structured follow-up interval."""
    years: int = 0
    months: int = 0
    weeks: int = 0
    
    def is_empty(self) -> bool:
        return self.years == 0 and self.months == 0 and self.weeks == 0


class IncidentalFindingCategory(str, Enum):
    pulmonary_nodule = "pulmonary_nodule"
    liver_lesion = "liver_lesion"
    renal_cyst = "renal_cyst"
    adrenal_nodule = "adrenal_nodule"
    thyroid_nodule = "thyroid_nodule"
    other = "other"


class FollowUpType(str, Enum):
    imaging = "imaging"
    clinical = "clinical"
    none = "none"
    unknown = "unknown"


class RecommendationStrength(str, Enum):
    explicit = "explicit"
    conditional = "conditional"
    none = "none"


class IncidentalFinding(BaseModel):
    """An incidental finding extracted from the report."""
    id: str = Field(..., description="Unique identifier for this finding (e.g., IF1)")
    description: str = Field(..., description="Short human-readable summary")
    verbatim_snippet: str = Field(..., description="Excerpt from report")
    location: Optional[str] = Field(None, description="Anatomic location")
    size: Optional[str] = Field(None, description="Size description specific to the finding")
    category: IncidentalFindingCategory
    followup_required: bool
    followup_type: FollowUpType
    followup_modality: Optional[str] = None
    followup_interval: Optional[FollowUpInterval] = None
    followup_interval_text: Optional[str] = None
    followup_rationale: Optional[str] = None
    recommendation_strength: RecommendationStrength


class ExamMetadata(BaseModel):
    """Metadata for the exam being analyzed."""
    accession: Optional[str] = None
    exam_date: Optional[str] = None
    modality: Optional[str] = None
    body_region: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None


class FollowUpExtractionRequest(BaseModel):
    """Request to extract follow-ups from a report."""
    exam_metadata: ExamMetadata
    report_text: str


class FollowUpExtractionResponse(BaseModel):
    """Extracted follow-up information."""
    version: str = "v1"
    incidental_findings: List[IncidentalFinding]
    global_followup_comment: Optional[str] = None
    has_any_followup: bool


# --- Agent 4: Structured Reporting & QA Coach ---

class QASeverity(str, Enum):
    CRITICAL = "critical"          # clinically dangerous or clearly wrong
    MAJOR = "major"                # important but not immediately dangerous
    MINOR = "minor"                # stylistic or minor clarity issues
    INFO = "info"                  # optional improvements


class QAType(str, Enum):
    CONSISTENCY = "consistency"    # findings vs impression mismatches
    COMPLETENESS = "completeness"  # missing required fields/sections
    CLARITY = "clarity"            # vague wording, over-hedging
    STRUCTURE = "structure"        # section formatting/ordering
    REDUNDANCY = "redundancy"      # repeated content
    TERMINOLOGY = "terminology"    # non-standard wording
    OTHER = "other"


class QASection(str, Enum):
    TECHNIQUE = "TECHNIQUE"
    COMPARISON = "COMPARISON"
    FINDINGS = "FINDINGS"
    IMPRESSION = "IMPRESSION"
    OTHER = "OTHER"
    GLOBAL = "GLOBAL"


class QAChangeType(str, Enum):
    SUGGEST_EDIT = "suggest_edit"      # suggest replacement text
    ADD_SECTION = "add_section"        # suggest adding missing section
    REMOVE_TEXT = "remove_text"        # suggest deletion
    REORDER = "reorder"                # suggest reordering content
    NOTE_ONLY = "note_only"            # comment only, no concrete change


class QARequiredFields(BaseModel):
    """
    Body-region/modality specific expectations.
    """
    modality: Optional[str]
    body_region: Optional[str]
    required_sections: List[QASection] = []


class QAIssue(BaseModel):
    id: str
    severity: QASeverity
    type: QAType
    section: QASection
    description: str               # human-readable explanation
    location_hint: Optional[str]   # e.g. snippet or “paragraph index 2”
    suggested_change_type: QAChangeType
    suggested_text: Optional[str]  # replacement/added text if applicable


class QASummary(BaseModel):
    overall_quality: str           # e.g. "good", "acceptable", "needs_revision"
    num_critical: int
    num_major: int
    num_minor: int
    comments: Optional[str]


class ReportQARequest(BaseModel):
    exam_metadata: ExamMetadata    # reuse existing ExamMetadata
    report_text: str               # full report text as seen by radiologist
    qa_requirements: Optional[QARequiredFields] = None


class ReportQAResponse(BaseModel):
    version: str = "v1"
    original_report_text: str
    normalized_report_text: Optional[str] = None  # cleaned/normalized structure if generated
    issues: List[QAIssue]
    summary: QASummary


# --- Agent 5: Patient-Friendly Report Explainer ---

class PatientReadingLevel(str, Enum):
    VERY_SIMPLE = "very_simple"   # ~5th grade
    SIMPLE = "simple"             # ~8th grade
    STANDARD = "standard"         # ~10–12th grade


class PatientSummaryTone(str, Enum):
    NEUTRAL = "neutral"
    REASSURING = "reassuring"


class PatientNextStepUrgency(str, Enum):
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    UNKNOWN = "unknown"


class GlossaryItem(BaseModel):
    term: str
    explanation: str   # lay-language explanation


class PatientNextStep(BaseModel):
    description: str                     # e.g. "Your doctor may order a follow-up CT scan in 6 months."
    urgency: PatientNextStepUrgency
    followup_interval: Optional[FollowUpInterval] = None  # reuse from Agent 3 if available
    source_finding_id: Optional[str] = None               # link to IncidentalFinding.id if passed in


class PatientReportSummaryRequest(BaseModel):
    exam_metadata: ExamMetadata         # reuse from Agent 3
    report_text: str                    # final radiology report
    # followup_data is optional – to link Agent 3
    followup_data: Optional[FollowUpExtractionResponse] = None  
    reading_level: PatientReadingLevel = PatientReadingLevel.SIMPLE
    tone: PatientSummaryTone = PatientSummaryTone.NEUTRAL
    language_code: str = "en"           # ISO code, e.g. "en", "es"


class PatientReportSummaryResponse(BaseModel):
    version: str = "v1"
    patient_summary_text: str           # short paragraph summary
    key_points: List[str]               # bullet-style summary items
    next_steps: List[PatientNextStep]   # actions / follow-ups in lay language
    glossary: List[GlossaryItem]
    original_report_text: str


class ReportDraftRequest(BaseModel):
    """Input request for the report drafting agent."""
    
    findings: List[Finding] = Field(..., description="List of radiological findings")
    clinical_context: ClinicalContext = Field(..., description="Clinical context")
    modality: Optional[str] = Field(None, description="Imaging modality (e.g., 'Chest X-ray')")
    view: Optional[str] = Field(None, description="View type (e.g., 'PA & lateral')")
    prior_study_summary: Optional[str] = Field(None, description="Summary of prior imaging if available")
    cv_summary: Optional[CVHighlightResult] = Field(None, description="Summary from CV agent")

    class Config:
        json_schema_extra = {
            "example": {
                "findings": [
                    {
                        "location": "right lower lobe",
                        "type": "opacity",
                        "severity": "moderate"
                    }
                ],
                "clinical_context": {
                    "patient_info": "65-year-old male",
                    "clinical_presentation": "Fever",
                    "relevant_history": "COPD"
                },
                "modality": "Chest X-ray",
                "cv_summary": {
                     "modality": "DX",
                     "summary": "Suspicious opacity in RLL.",
                     "regions": [
                         {"label": "Opacity", "score": 0.85}
                     ]
                }
            }
        }


class ReportDraft(BaseModel):
    """Output report from the drafting agent."""
    
    report_text: str = Field(..., description="Full formatted report text (TECHNIQUE, COMPARISON, FINDINGS, IMPRESSION)")
    key_findings: List[KeyFinding] = Field(default_factory=list, description="Structured key findings")
    used_cv_signals: List[UsedCVSignal] = Field(default_factory=list, description="How CV signals were used")
    confidence_score: float = Field(0.75, ge=0.0, le=1.0, description="Confidence score (0-1)")

    class Config:
        json_schema_extra = {
            "example": {
                "report_text": "TECHNIQUE: ... FINDINGS: ... IMPRESSION: ...",
                "key_findings": [
                    {"label": "RLL Opacity", "category": "pathology", "severity": "significant"}
                ],
                "used_cv_signals": [
                    {"cv_label": "Opacity", "included_in_report": True, "reasoning": "Correlates with radiologist finding"}
                ],
                "confidence_score": 0.85
            }
        }


# --- Agent 6: Worklist Triage & Priority Recommender ---

class TriageLabel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"
    UNTRIAGED = "UNTRIAGED"


class TriageReasonType(str, Enum):
    MODEL_PREDICTION = "MODEL_PREDICTION"
    CLINICAL_INDICATION = "CLINICAL_INDICATION"
    TIME_IN_QUEUE = "TIME_IN_QUEUE"
    PROTOCOL_PRIORITY = "PROTOCOL_PRIORITY"
    FALLBACK = "FALLBACK"


class ModalityGroup(str, Enum):
    XR = "XR"
    CT = "CT"
    MR = "MR"
    US = "US"
    NM = "NM"
    OTHER = "OTHER"


class TriageThresholdConfig(BaseModel):
    """Configuration for triage thresholds for a specific modality/region."""
    modality_group: ModalityGroup
    body_region: Optional[str] = None
    critical_threshold: float = 0.9
    high_threshold: float = 0.7
    low_threshold: float = 0.3
    min_confidence: float = 0.5


class TriageConfig(BaseModel):
    """Global triage configuration."""
    thresholds: List[TriageThresholdConfig]
    max_batch_size: int = 10
    enable_llm_explanation: bool = True
    model_mapping: Dict[str, str] = Field(default_factory=dict, description="Map 'ModalityGroup/Region' to model name")


class WorklistItem(BaseModel):
    """An item in the worklist to be triaged."""
    study_id: str = Field(..., description="Unique identifier for the study")
    accession: Optional[str] = None
    patient_id: Optional[str] = None  # Treat as PHI
    exam_datetime: Optional[str] = None
    modality: str
    body_region: Optional[str] = None
    clinical_indication: Optional[str] = None
    priority_flag_from_order: Optional[str] = None  # e.g. "STAT"
    image_reference: Optional[Dict[str, Any]] = None  # e.g. {"thumbnail_path": "..."}


class WorklistTriageRequest(BaseModel):
    """Request to triage a list of studies."""
    worklist_items: List[WorklistItem]


class TriageReason(BaseModel):
    """Reason for a specific triage decision."""
    type: TriageReasonType
    description: str
    weight: float = 0.0


class WorklistTriageItem(BaseModel):
    """Triaged item with score and label."""
    study_id: str
    triage_score: float = Field(..., ge=0.0, le=1.0)
    triage_label: TriageLabel
    reasons: List[TriageReason]
    model_metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation_text: Optional[str] = None
    error: Optional[str] = None


class WorklistTriageResponse(BaseModel):
    """Batch response for triage request."""
    version: str = "v1"
    items: List[WorklistTriageItem]


# --- Agent 7: Learning & Feedback / Case Digest ---

class LearningEventType(str, Enum):
    ADDENDUM_CORRECTION = "ADDENDUM_CORRECTION"
    QA_ISSUE = "QA_ISSUE"
    PEER_REVIEW_DISCREPANCY = "PEER_REVIEW_DISCREPANCY"
    MISSED_FINDING = "MISSED_FINDING"
    OVER_CALL = "OVER_CALL"
    FOLLOWUP_MISMATCH = "FOLLOWUP_MISMATCH"
    INTERESTING_CASE = "INTERESTING_CASE"


class DiscrepancySeverity(str, Enum):
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    INFO = "INFO"


class DigestScope(str, Enum):
    INDIVIDUAL_RADIOLOGIST = "INDIVIDUAL_RADIOLOGIST"
    SERVICE_LINE = "SERVICE_LINE"


class DigestPeriod(str, Enum):
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    CUSTOM = "CUSTOM"


class LearningEvent(BaseModel):
    """Internal representation of a learning event."""
    event_id: str
    radiologist_id: str
    exam_metadata: ExamMetadata
    event_type: LearningEventType
    severity: DiscrepancySeverity
    source: str  # e.g., "qa_agent", "peer_review", "addendum"
    timestamp: str  # ISO datetime string
    report_text_before: Optional[str] = None
    report_text_after: Optional[str] = None
    qa_issues: Optional[List[QAIssue]] = None
    followup_data: Optional[FollowUpExtractionResponse] = None
    triage_info: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list)


class RadiologistLearningDigestRequest(BaseModel):
    """Request for a learning digest."""
    radiologist_id: str
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    scope: DigestScope = DigestScope.INDIVIDUAL_RADIOLOGIST
    digest_period: Optional[DigestPeriod] = None
    modality_filter: Optional[List[str]] = None
    body_region_filter: Optional[List[str]] = None
    max_cases: int = 10
    include_qc_only: bool = False
    language_code: str = "en"
    include_raw_snippets: bool = False


class LearningCaseSnippet(BaseModel):
    """A single case summary in the digest."""
    event_id: str
    exam_metadata: ExamMetadata
    event_type: LearningEventType
    severity: DiscrepancySeverity
    tags: List[str]
    short_description: str
    key_lesson: str
    report_snippet_before: Optional[str] = None
    report_snippet_after: Optional[str] = None


class LearningTheme(BaseModel):
    """A theme or pattern identified in the events."""
    theme_id: str
    name: str
    description: str
    event_ids: List[str]
    suggested_actions: List[str]


class LearningStats(BaseModel):
    """Statistics for the digest period."""
    num_total_events: int
    num_critical: int
    num_major: int
    num_minor: int
    num_addenda: int
    num_peer_review_discrepancies: int
    num_cases_in_digest: int


class RadiologistLearningDigestResponse(BaseModel):
    """The final structured learning digest."""
    version: str = "v1"
    radiologist_id: str
    start_date: str
    end_date: str
    language_code: str
    summary_text: str
    key_themes: List[LearningTheme]
    cases: List[LearningCaseSnippet]
    stats: LearningStats
    generation_metadata: Dict[str, Any]


# --- Agent 8: Study Finalization Orchestrator ---

class StudyPipelineStage(str, Enum):
    CV_ANALYSIS = "CV_ANALYSIS"
    REPORT_DRAFT = "REPORT_DRAFT"
    QA_REVIEW = "QA_REVIEW"
    FOLLOWUP_EXTRACTION = "FOLLOWUP_EXTRACTION"
    PATIENT_SUMMARY = "PATIENT_SUMMARY"


class StageStatus(str, Enum):
    NOT_RUN = "NOT_RUN"
    SUCCESS = "SUCCESS"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class PipelineStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"


class PipelineOptions(BaseModel):
    """Configuration for the orchestration pipeline."""
    run_cv_analysis: bool = True
    run_qa_review: bool = True
    run_followup_extraction: bool = True
    run_patient_summary: bool = True
    max_stage_timeout_seconds: Optional[int] = None


class StudyOrchestrationRequest(BaseModel):
    """Request to orchestrate the entire study workflow."""
    study_id: str
    exam_metadata: ExamMetadata
    clinical_context: ClinicalContext
    image_references: Optional[List[Dict[str, Any]]] = None
    prior_report_text: Optional[str] = None
    radiologist_id: Optional[str] = None
    pipeline_options: PipelineOptions = Field(default_factory=PipelineOptions)
    language_code: str = "en"
    dry_run: bool = False


class StageResult(BaseModel):
    """Result of a single pipeline stage."""
    stage: StudyPipelineStage
    status: StageStatus
    error: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudyBundle(BaseModel):
    """Aggregated results from all agents."""
    study_id: str
    exam_metadata: ExamMetadata
    cv_analysis: Optional[CVHighlightResult] = None
    report_draft: Optional[ReportDraft] = None
    qa_result: Optional[ReportQAResponse] = None
    final_report_text: Optional[str] = None
    followup_data: Optional[FollowUpExtractionResponse] = None
    patient_summary: Optional[PatientReportSummaryResponse] = None


class StudyOrchestrationResponse(BaseModel):
    """Final response from the orchestrator."""
    version: str = "v1"
    pipeline_status: PipelineStatus
    study_id: str
    bundle: StudyBundle
    stages: List[StageResult]
    generation_metadata: Dict[str, Any]


