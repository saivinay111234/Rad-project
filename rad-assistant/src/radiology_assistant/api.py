"""FastAPI wrapper for the Radiology Assistant API.

Exposes all 8 agents as REST endpoints with JWT authentication,
role-based access control, and production-grade error handling.
"""

import logging
from typing import Any, Optional, List, Dict
from pydantic import BaseModel
from functools import lru_cache
from contextlib import asynccontextmanager
from datetime import timedelta

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from radiology_assistant.models import (
    ReportDraftRequest, ReportDraft,
    CVHighlightRequest, CVHighlightResult,
    FollowUpExtractionRequest, FollowUpExtractionResponse,
    ReportQARequest, ReportQAResponse,
    PatientReportSummaryRequest, PatientReportSummaryResponse,
    WorklistTriageRequest, WorklistTriageResponse,
    StudyOrchestrationRequest, StudyOrchestrationResponse,
    RadiologistLearningDigestRequest, RadiologistLearningDigestResponse, LearningEvent,
)
from radiology_assistant.config import Config
from radiology_assistant.llm_client import LLMClient
from radiology_assistant.auth import (
    TokenResponse, authenticate_user, create_access_token,
    get_current_user, require_role, UserRole, TokenData,
)
from radiology_assistant.agents import ReportDraftingAgent
from radiology_assistant.agents.visual_highlighter import VisualHighlightingAgent
from radiology_assistant.agents.worklist_triage import WorklistTriageAgent
from radiology_assistant.agents.study_orchestrator import StudyOrchestratorAgent
from radiology_assistant.cv.models import ChestXrayAnomalyModel
from radiology_assistant.cv.io import InvalidDICOMError, load_dicom_with_metadata, load_image_from_bytes

from sqlalchemy.orm import Session
from radiology_assistant.database import get_db, init_db
from radiology_assistant.repositories import SQLLearningEventRepository
from radiology_assistant.middleware import AuditLoggingMiddleware
from radiology_assistant.observability import setup_logging, setup_metrics
from radiology_assistant.tasks import orchestrate_study_task
from celery.result import AsyncResult

logger = logging.getLogger(__name__)
# setup_logging() will configure the root logger later in lifespan


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler: initialize DB tables and pre-warm agents on startup."""
    # 1. Setup structured logging
    setup_logging()
    logger.info("API startup: initializing database")
    
    # 2. Init DB
    init_db()  # Create tables if they don't exist (idempotent)
    
    # 3. Setup Prometheus metrics
    setup_metrics(app)
    
    logger.info("API startup: pre-initializing ReportDraftingAgent")
    try:
        get_agent()
        logger.info("ReportDraftingAgent pre-initialized successfully")
    except Exception:
        logger.exception("Failed to pre-initialize agent on startup")
        raise
    yield
    logger.info("API shutdown")
    logger.info("API shutdown")


# Rate limiting setup
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Radiology Assistant API",
    description="AI-powered radiology assistant with 8 specialized agents.",
    version="2.0.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Phase 4: Production Audit Logging
app.add_middleware(AuditLoggingMiddleware)

# ---------------------------------------------------------------------------
# Shared LLM client getter
# ---------------------------------------------------------------------------

@lru_cache()
def get_llm_client() -> LLMClient:
    """Return a cached singleton LLMClient."""
    Config.validate()
    return LLMClient()


# Module-level singleton for Agent 1. Tests can monkeypatch this value.
_agent: ReportDraftingAgent | None = None


# ---------------------------------------------------------------------------
# Auth endpoints (public — no token required)
# ---------------------------------------------------------------------------

@app.post("/v1/auth/token", response_model=TokenResponse, tags=["Auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange username + password for a JWT access token.

    **Demo credentials** (change in production):
    - `admin` / `admin123` (role: admin)
    - `dr_smith` / `radiologist123` (role: radiologist)
    - `patient_viewer` / `viewer123` (role: patient_viewer)
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    expires = timedelta(minutes=Config.JWT_EXPIRE_MINUTES)
    token = create_access_token(sub=user["username"], role=user["role"], expires_delta=expires)
    logger.info("Token issued for user=%s role=%s", user["username"], user["role"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=Config.JWT_EXPIRE_MINUTES * 60,
    )



def get_agent() -> ReportDraftingAgent:
    """Return singleton ReportDraftingAgent, creating it on first use.

    Creation is lazy to avoid forcing configuration validation at import time
    (which makes tests brittle). Tests should set `api._agent` to a Mock
    before calling endpoints to avoid needing a real API key.
    """
    global _agent
    if _agent is None:
        # Validate config but don't crash at import time — raise if missing when used.
        try:
            Config.validate()
        except Exception as e:
            logger.error("Configuration error while initializing agent: %s", e)
            raise

        llm_client = LLMClient()
        _agent = ReportDraftingAgent(llm_client)
        logger.info("ReportDraftingAgent initialized")
    return _agent



# ---------------------------------------------------------------------------
# Public endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
def health() -> Any:
    """Simple health check endpoint (no auth required)."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "radiology-assistant",
        "version": "2.0.0",
    }


# ---------------------------------------------------------------------------
# Agent 1: Report Drafting
# ---------------------------------------------------------------------------

@app.post("/v1/reports/draft", response_model=ReportDraft, tags=["Agent 1: Report Drafting"])
@app.post("/draft_report", response_model=ReportDraft, include_in_schema=False)  # legacy alias
def draft_report(
    request: ReportDraftRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Draft a radiology report from structured findings and clinical context."""
    logger.info("draft_report: user=%s modality=%s", current_user.sub, request.modality)
    try:
        agent = get_agent()
    except Exception:
        logger.exception("Failed to initialize agent")
        raise HTTPException(status_code=500, detail="Internal error while initializing agent.")
    try:
        report = agent.draft_report(request)
        logger.info("Report drafted successfully")
        return report
    except Exception:
        logger.exception("Error drafting report")
        raise HTTPException(status_code=500, detail="Internal error while drafting report.")


# ---------------------------------------------------------------------------
# Agent 2: Visual Highlighting
# ---------------------------------------------------------------------------

_cv_agent: VisualHighlightingAgent | None = None

def get_cv_agent() -> VisualHighlightingAgent:
    """Return singleton VisualHighlightingAgent."""
    global _cv_agent
    if _cv_agent is None:
        # In a real app, we might load weights from Config
        model = ChestXrayAnomalyModel(device="cpu")
        _cv_agent = VisualHighlightingAgent(model=model, logger=logger)
        logger.info("VisualHighlightingAgent initialized")
    return _cv_agent

@app.post("/v1/cv/highlight", response_model=CVHighlightResult, tags=["Agent 2: Visual Highlighting"])
@app.post("/cv/highlight", response_model=CVHighlightResult, include_in_schema=False)  # legacy alias
async def cv_highlight(
    request: CVHighlightRequest = Depends(),
    file: UploadFile = File(...),
    current_user: TokenData = Depends(get_current_user),
):
    """Highlight anomalies in an uploaded chest X-ray DICOM or image file."""
    try:
        image_bytes = await file.read()
    except Exception:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    agent = get_cv_agent()
    
    # Try to extract DICOM metadata to auto-fill modality if missing
    modality = request.modality
    try:
        _, meta = load_dicom_with_metadata(image_bytes)
        if not modality and meta.get("modality"):
            modality = meta["modality"]
            logger.info("Auto-populated modality from DICOM: %s", modality)
    except Exception:
        # Fallback to request modality or generic image loading in agent
        pass

    logger.info("cv_highlight: user=%s modality=%s", current_user.sub, modality)
    
    # Update request if modality was found
    if not request.modality and modality:
        request.modality = modality

    try:
        result = agent.highlight(request, image_bytes)
        return result
    except ValueError as e:
        logger.warning("cv_highlight value error: %s", e)
        raise HTTPException(status_code=422, detail=str(e))
    except InvalidDICOMError:
        logger.warning("Invalid DICOM/image uploaded")
        raise HTTPException(status_code=400, detail="Invalid DICOM or image format.")
    except Exception:
        logger.exception("Error in cv_highlight")
        raise HTTPException(status_code=500, detail="Internal error while highlighting image.")


# ---------------------------------------------------------------------------
# Phase 10: DICOM Integration - Metadata Retrieval
# ---------------------------------------------------------------------------

@app.get("/v1/dicom/{accession}/metadata", tags=["DICOM Integration"])
def get_dicom_metadata(
    accession: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Simulated DICOMweb WADO-RS metadata endpoint.
    In a real system, this would query a PACS or DICOM archive.
    Here we search our local Report database for a matching study/accession.
    """
    from radiology_assistant.db_models import ReportDB
    
    report = db.query(ReportDB).filter(ReportDB.study_id == accession).first()
    if not report:
         raise HTTPException(status_code=404, detail="Study/Accession not found in local cache.")
    
    return {
        "accession": report.study_id,
        "radiologist_id": report.radiologist_id,
        "created_at": report.created_at.isoformat(),
        "summary": "DICOM metadata retrieved from local study record."
    }


# ---------------------------------------------------------------------------
# Agent 3: Follow-Up & Incidental Findings
# ---------------------------------------------------------------------------

_followup_agent: "FollowUpExtractorAgent | None" = None

def get_followup_agent() -> "FollowUpExtractorAgent":
    """Return singleton FollowUpExtractorAgent."""
    global _followup_agent
    if _followup_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        
        llm_client = LLMClient()
        # Local import to avoid circular dependencies if any
        from radiology_assistant.agents.followup_extractor import FollowUpExtractorAgent
        _followup_agent = FollowUpExtractorAgent(llm_client, logger=logger)
        logger.info("FollowUpExtractorAgent initialized")
    return _followup_agent


@app.post("/v1/reports/extract_followups", response_model=FollowUpExtractionResponse, tags=["Agent 3: Follow-Up Tracker"])
def extract_followups(
    request: FollowUpExtractionRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Extract follow-up recommendations and incidental findings from a report."""
    logger.info("extract_followups: user=%s", current_user.sub)
    try:
        agent = get_followup_agent()
        response = agent.extract_followups(request)
        return response
    except Exception:
        logger.exception("Error identifying follow-ups")
        raise HTTPException(status_code=500, detail="Internal error while extracting follow-ups.")



# --- Agent 4: Structured Reporting & QA Coach ---

_qa_agent: "ReportQAAgent | None" = None

def get_report_qa_agent() -> "ReportQAAgent":
    """Return singleton ReportQAAgent."""
    global _qa_agent
    if _qa_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        
        llm_client = LLMClient()
        from radiology_assistant.agents.report_qa_agent import ReportQAAgent
        _qa_agent = ReportQAAgent(llm_client, logger=logger)
        logger.info("ReportQAAgent initialized")
    return _qa_agent


@app.post("/v1/reports/qa", response_model=ReportQAResponse, tags=["Agent 4: QA Coach"])
def review_report_qa(
    request: ReportQARequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Review a report for quality, consistency, and completeness."""
    logger.info("report_qa: user=%s", current_user.sub)
    try:
        agent = get_report_qa_agent()
        response = agent.review_report(request)
        return response
    except Exception as e:
        if type(e).__name__ == "LLMJsonParseError":
            logger.exception("report_qa_json_parse_error")
            raise HTTPException(status_code=502, detail="Failed to parse QA model response.")
        logger.exception("report_qa_unexpected_error")
        raise HTTPException(status_code=500, detail="Unexpected error during report QA.")


# --- Agent 5: Patient-Friendly Report Explainer ---

_patient_explainer_agent: "PatientReportExplainerAgent | None" = None

def get_patient_explainer_agent() -> "PatientReportExplainerAgent":
    """Return singleton PatientReportExplainerAgent."""
    global _patient_explainer_agent
    if _patient_explainer_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        
        llm_client = LLMClient()
        from radiology_assistant.agents.patient_report_explainer import PatientReportExplainerAgent
        _patient_explainer_agent = PatientReportExplainerAgent(llm_client, logger=logger)
        logger.info("PatientReportExplainerAgent initialized")
    return _patient_explainer_agent


@app.post("/v1/reports/patient_summary", response_model=PatientReportSummaryResponse, tags=["Agent 5: Patient Explainer"])
def generate_patient_summary(
    request: PatientReportSummaryRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Generate a patient-friendly report summary and next steps."""
    logger.info("patient_summary: user=%s", current_user.sub)
    try:
        agent = get_patient_explainer_agent()
        response = agent.explain(request)
        return response
    except Exception as e:
        if type(e).__name__ == "LLMJsonParseError":
            logger.exception("patient_explainer_json_parse_error")
            raise HTTPException(status_code=502, detail="Failed to parse patient explainer model response.")
        logger.exception("patient_explainer_unexpected_error")
        raise HTTPException(status_code=500, detail="Unexpected error during patient explanation.")


# ---------------------------------------------------------------------------
# Agent 6: Worklist Triage
# ---------------------------------------------------------------------------

_triage_agent: WorklistTriageAgent | None = None

def get_triage_agent() -> WorklistTriageAgent:
    """Return singleton WorklistTriageAgent."""
    global _triage_agent
    if _triage_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        
        llm_client = LLMClient()
        triage_config = Config.get_triage_config()
        _triage_agent = WorklistTriageAgent(config=triage_config, llm_client=llm_client)
        logger.info("WorklistTriageAgent initialized")
    return _triage_agent


@app.post("/v1/worklist/triage", response_model=WorklistTriageResponse, tags=["Agent 6: Worklist Triage"])
def triage_worklist(
    request: WorklistTriageRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Agent 6: Triage a worklist of studies and recommend priority order."""
    logger.info("worklist_triage: user=%s items=%d", current_user.sub, len(request.worklist_items))
    try:
        agent = get_triage_agent()
        return agent.triage(request)
    except Exception:
        logger.exception("Error during worklist triage")
        raise HTTPException(status_code=500, detail="Internal error during triage.")


# ---------------------------------------------------------------------------
# Agent 7: Learning & Feedback
# ---------------------------------------------------------------------------

from radiology_assistant.agents.learning_feedback import LearningFeedbackAgent, LearningEventRepository


class MockLearningEventRepository:
    """Kept for test compatibility only — use SQLLearningEventRepository in production."""
    def get_events(self, radiologist_id: str, start_date: str, end_date: str) -> list[LearningEvent]:
        return []
    def save(self, event: LearningEvent) -> None:
        pass

_learning_agent: LearningFeedbackAgent | None = None

def get_learning_agent(db: Session = None) -> LearningFeedbackAgent:
    """Return a LearningFeedbackAgent backed by the database."""
    global _learning_agent
    if _learning_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        llm_client = LLMClient()
        # Use mock repository as fallback if no DB session provided (e.g. tests)
        repository = SQLLearningEventRepository(db) if db is not None else MockLearningEventRepository()
        _learning_agent = LearningFeedbackAgent(repository, llm_client, logger=logger)
        logger.info("LearningFeedbackAgent initialized")
    return _learning_agent

@app.post("/v1/learning/radiologist_digest", response_model=RadiologistLearningDigestResponse, tags=["Agent 7: Learning"])
def generate_radiologist_digest(
    request: RadiologistLearningDigestRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Generate a learning & feedback digest for a radiologist."""
    logger.info("radiologist_digest: user=%s target_id=%s", current_user.sub, request.radiologist_id)
    try:
        agent = get_learning_agent()
        return agent.generate_radiologist_digest(request)
    except Exception:
        logger.exception("Error generating learning digest")
        raise HTTPException(status_code=500, detail="Internal error generating digest.")



# --- Agent 8: Orchestrator ---

@lru_cache()
def get_orchestrator_agent() -> StudyOrchestratorAgent:
    """Dependency injection for Agent 8 Orchestrator."""
    llm = get_llm_client()
    
    # Reuse existing getters where possible or instantiate fresh
    # Agent 1
    drafter = get_agent()
    
    # Agent 2
    cv = get_cv_agent()
    
    # Agent 3
    followup = get_followup_agent()
    
    # Agent 4
    qa = get_report_qa_agent()
    
    # Agent 5
    explainer = get_patient_explainer_agent()
    
    # Agent 7 (Repo needs to be extracted or we use the Mock from get_learning_agent if accessible)
    # Since get_learning_agent creates the repo inside, we might need to access it or duplicate logic.
    # For now, we reuse the pattern:
    learning = get_learning_agent()
    learning_repo = learning.repository
    
    return StudyOrchestratorAgent(
        report_drafter=drafter,
        qa_agent=qa,
        followup_agent=followup,
        patient_explainer=explainer,
        cv_agent=cv,
        learning_repo=learning_repo,
        logger=logger
    )

@app.post("/v1/studies/orchestrate", tags=["Agent 8: Orchestrator"])
@limiter.limit("10/minute")
async def orchestrate_study(
    request: StudyOrchestrationRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Agent 8: Orchestrate the entire radiology study workflow.
    Dispatches a Celery task for background processing.
    Returns a task_id for polling.
    """
    logger.info("orchestrate_study: user=%s study=%s", current_user.sub, request.study_id)
    # Convert Pydantic model to dict for Celery serialization
    task = orchestrate_study_task.delay(request.model_dump())
    return {"task_id": task.id, "status": "PENDING"}


@app.get("/v1/studies/tasks/{task_id}/status", tags=["Agent 8: Orchestrator"])
def get_orchestration_status(
    task_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Poll the status of a background study orchestration task.
    Returns status (PENDING, STARTED, SUCCESS, FAILURE) and result if ready.
    """
    res = AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": res.status,
    }
    if res.ready():
        if res.successful():
            response["result"] = res.result
        else:
            response["error"] = str(res.result)
    return response


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feedback Loop
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    radiologist_id: str
    correction_type: str = "general"
    correction_text: str

class FeedbackResponse(BaseModel):
    feedback_id: str
    report_id: str
    learning_event_id: Optional[str] = None
    message: str

@app.post("/v1/reports/{report_id}/feedback", response_model=FeedbackResponse, tags=["Feedback Loop"])
def submit_feedback(
    report_id: str,
    payload: FeedbackRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit a radiologist correction on a report. Auto-creates a LearningEvent."""
    import uuid as _uuid
    from datetime import datetime, timezone
    from radiology_assistant.repositories import FeedbackRepository, SQLLearningEventRepository
    from radiology_assistant.models import LearningEvent, LearningEventType, DiscrepancySeverity

    logger.info("submit_feedback: user=%s report=%s", current_user.sub, report_id)
    feedback_repo = FeedbackRepository(db)
    feedback_id = feedback_repo.save_feedback(
        report_id=report_id,
        radiologist_id=payload.radiologist_id,
        correction_text=payload.correction_text,
        correction_type=payload.correction_type,
    )

    learning_event_id = str(_uuid.uuid4())
    try:
        event = LearningEvent(
            event_id=learning_event_id,
            radiologist_id=payload.radiologist_id,
            event_type=LearningEventType.ADDENDUM_CORRECTION,
            severity=DiscrepancySeverity.MINOR,
            report_text_after=payload.correction_text,
            tags=["feedback", payload.correction_type],
            timestamp=datetime.now(timezone.utc),
        )
        SQLLearningEventRepository(db).save(event)
    except Exception:
        logger.exception("Failed to create LearningEvent from feedback (non-fatal)")
        learning_event_id = None

    return FeedbackResponse(
        feedback_id=feedback_id,
        report_id=report_id,
        learning_event_id=learning_event_id,
        message="Feedback recorded and learning event created.",
    )


# ---------------------------------------------------------------------------
# Fatigue Detector
# ---------------------------------------------------------------------------

@app.get("/v1/radiologist/{radiologist_id}/fatigue_report", tags=["Fatigue Detector"])
def get_fatigue_report(
    radiologist_id: str,
    analysis_days: int = 30,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Analyze error patterns by time-of-day/day-of-week for a radiologist.
    Returns high-risk time slots and peer review recommendation.
    """
    if current_user.role != UserRole.ADMIN.value and current_user.sub != radiologist_id:
        raise HTTPException(status_code=403, detail="Access denied.")
    from radiology_assistant.agents.fatigue_detector import RadiologistFatigueDetector
    try:
        report = RadiologistFatigueDetector().analyze(
            radiologist_id=radiologist_id, db_session=db, analysis_period_days=analysis_days
        )
        return report.model_dump()
    except Exception:
        logger.exception("Fatigue analysis failed for radiologist=%s", radiologist_id)
        raise HTTPException(status_code=500, detail="Failed to generate fatigue report.")


# ---------------------------------------------------------------------------
# FHIR Export
# ---------------------------------------------------------------------------

@app.get("/v1/studies/{study_id}/fhir", tags=["FHIR Export"])
def export_study_fhir(
    study_id: str,
    patient_id: Optional[str] = None,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export a study report as a FHIR R4 DiagnosticReport (HL7 JSON)."""
    from radiology_assistant.agents.fhir_exporter import FHIRExporter
    from radiology_assistant.db_models import ReportDB

    logger.info("fhir_export: user=%s study=%s", current_user.sub, study_id)
    report = db.query(ReportDB).filter_by(study_id=study_id).order_by(ReportDB.created_at.desc()).first()

    class _Stub:
        pass
    bundle = _Stub()
    if report:
        stub_draft = _Stub()
        stub_draft.report_text = report.report_text
        stub_draft.key_findings = report.get_key_findings()
        bundle.report_draft = stub_draft
        bundle.followup_data = None
        bundle.qa_result = None
        bundle.exam_metadata = None

    return FHIRExporter().to_diagnostic_report(
        bundle=bundle, patient_id=patient_id, study_id=study_id,
        report_id=getattr(report, "id", None),
    )


# ---------------------------------------------------------------------------
# Follow-up Reminders
# ---------------------------------------------------------------------------

class ScheduleRemindersRequest(BaseModel):
    study_id: str
    patient_token: Optional[str] = None
    followup_response_dict: dict

@app.post("/v1/followup/schedule", tags=["Follow-up Prevention"])
def schedule_followup_reminders(
    payload: ScheduleRemindersRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Schedule follow-up reminders from a FollowUpExtractionResponse dict."""
    from radiology_assistant.agents.followup_reminder import FollowupReminderEngine
    from radiology_assistant.models import FollowUpExtractionResponse
    logger.info("schedule_reminders: user=%s study=%s", current_user.sub, payload.study_id)
    try:
        followup = FollowUpExtractionResponse(**payload.followup_response_dict)
        engine = FollowupReminderEngine()
        result = engine.schedule_reminders(
            study_id=payload.study_id,
            findings=followup.findings or [],
            patient_token=payload.patient_token,
            db_session=db,
        )
        return result.model_dump()
    except Exception:
        logger.exception("Failed to schedule follow-up reminders")
        raise HTTPException(status_code=500, detail="Failed to schedule reminders.")


@app.get("/v1/followup/due", tags=["Follow-up Prevention"])
def get_due_reminders(
    within_days: int = 7,
    current_user: TokenData = Depends(require_role(UserRole.ADMIN, UserRole.RADIOLOGIST)),
    db: Session = Depends(get_db),
):
    """Return pending follow-up reminders due within N days."""
    from radiology_assistant.agents.followup_reminder import FollowupReminderEngine
    reminders = FollowupReminderEngine().get_due_reminders(db_session=db, within_days=within_days)
    return {"within_days": within_days, "count": len(reminders),
            "reminders": [r.model_dump() for r in reminders]}


# ---------------------------------------------------------------------------
# CME Platform
# ---------------------------------------------------------------------------

class CMEGenerateRequest(BaseModel):
    radiologist_id: str
    digest_dict: dict

class CMEGradeRequest(BaseModel):
    radiologist_id: str
    submitted_answers: dict

@app.post("/v1/cme/generate_case", tags=["CME Platform"])
def generate_cme_case(
    payload: CMEGenerateRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a CME case (3 MCQs + explanations) from a radiologist's learning digest."""
    from radiology_assistant.agents.cme_platform import CMEPlatformAgent
    from radiology_assistant.models import RadiologistLearningDigestResponse
    logger.info("generate_cme_case: user=%s rad=%s", current_user.sub, payload.radiologist_id)
    try:
        digest = RadiologistLearningDigestResponse(**payload.digest_dict)
        case = CMEPlatformAgent(llm_client=get_llm_client()).generate_cme_case(
            radiologist_id=payload.radiologist_id, digest=digest, db_session=db
        )
        return case.model_dump()
    except Exception:
        logger.exception("CME case generation failed")
        raise HTTPException(status_code=500, detail="Failed to generate CME case.")


@app.post("/v1/cme/{case_id}/grade", tags=["CME Platform"])
def grade_cme_case(
    case_id: str,
    payload: CMEGradeRequest,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Grade submitted CME answers. Returns score, pass/fail, and credits earned."""
    import json as _json
    from radiology_assistant.agents.cme_platform import CMEPlatformAgent, CMECase
    from radiology_assistant.db_models import CMECaseDB

    row = db.query(CMECaseDB).filter(CMECaseDB.id == case_id).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"CME case {case_id} not found.")
    try:
        case = CMECase(**_json.loads(row.case_json))
        result = CMEPlatformAgent(llm_client=get_llm_client()).grade_answers(
            case=case, submitted_answers=payload.submitted_answers,
            radiologist_id=payload.radiologist_id, db_session=db,
        )
        return result.model_dump()
    except Exception:
        logger.exception("CME grading failed for case=%s", case_id)
        raise HTTPException(status_code=500, detail="Failed to grade CME case.")


# ---------------------------------------------------------------------------
# Phase 8: Admin - Dynamic Triage Config
# ---------------------------------------------------------------------------

@app.put("/v1/admin/triage_config", tags=["Admin"])
def update_triage_config(
    new_config: dict,
    current_user: TokenData = Depends(require_role(UserRole.ADMIN)),
):
    """
    Update the triage configuration YAML file. Admin only.
    Expects full TriageConfig JSON/Dict.
    """
    import yaml
    import os
    from radiology_assistant.models import TriageConfig
    
    # 1. Validate the new config
    try:
        parsed = TriageConfig(**new_config)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid triage config: {e}")

    # 2. Save to file
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "triage_config.yaml")
    try:
        with open(config_path, 'w') as f:
            yaml.dump(parsed.model_dump(), f)
    except Exception as e:
        logger.exception("Failed to save triage config")
        raise HTTPException(status_code=500, detail="Failed to persist configuration.")

    # 3. Clear lru_cache for triage agent getter to force reload
    get_triage_agent.cache_clear()
    
    return {"message": "Triage configuration updated and agent reloaded."}


if __name__ == "__main__":
    # Allow running via: python -m radiology_assistant.api
    import uvicorn

    uvicorn.run("radiology_assistant.api:app", host="0.0.0.0", port=8000, reload=True)
