"""FastAPI wrapper for the Report Drafting Agent.

This module exposes a thin HTTP API around the existing Agent 1
implementation. It intentionally keeps the agent instantiation
lightweight and allows tests to patch `_agent` to avoid real LLM calls.
"""

import logging
from typing import Any
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends

from radiology_assistant.models import (
    ReportDraftRequest, ReportDraft, 
    CVHighlightRequest, CVHighlightResult,
    FollowUpExtractionRequest, FollowUpExtractionResponse,
    ReportQARequest, ReportQAResponse,
    PatientReportSummaryRequest, PatientReportSummaryResponse,
    WorklistTriageRequest, WorklistTriageResponse,
    StudyOrchestrationRequest, StudyOrchestrationResponse
)
from radiology_assistant.config import Config
from radiology_assistant.llm_client import LLMClient
from radiology_assistant.agents import ReportDraftingAgent
from radiology_assistant.agents.visual_highlighter import VisualHighlightingAgent
from radiology_assistant.agents.worklist_triage import WorklistTriageAgent
from radiology_assistant.agents.study_orchestrator import StudyOrchestratorAgent
from radiology_assistant.cv.models import ChestXrayAnomalyModel
from radiology_assistant.cv.io import InvalidDICOMError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler to pre-initialize the agent on startup.

    Uses the same fail-fast behavior as the previous startup event but
    avoids the deprecated `on_event` API.
    """
    logger.info("API startup: pre-initializing ReportDraftingAgent")
    try:
        # Create the singleton agent (may raise if config is invalid)
        get_agent()
        logger.info("ReportDraftingAgent pre-initialized successfully")
    except Exception:
        logger.exception("Failed to pre-initialize agent on startup")
        raise
    yield
    logger.info("API shutdown")


app = FastAPI(title="Radiology Report Drafting Agent API", lifespan=lifespan)

# Module-level singleton for the agent. Tests can monkeypatch this value.
_agent: ReportDraftingAgent | None = None


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



# lifespan handler replaces the previous startup event (see above)


@app.get("/health")
def health() -> Any:
    """Simple health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "radiology-report-agent",
        "version": "1.0.0",
    }


@app.post("/draft_report", response_model=ReportDraft)
def draft_report(request: ReportDraftRequest):
    """Draft a radiology report from a `ReportDraftRequest`.

    This endpoint uses the existing `ReportDraftingAgent` to produce a
    `ReportDraft` model. Errors are logged and returned as HTTP 500.
    """
    logger.info("Received draft_report request: modality=%s view=%s", request.modality, request.view)
    try:
        agent = get_agent()
    except Exception:
        # Log internal exception details (no PHI expected here) but return generic error to client
        logger.exception("Failed to initialize agent (internal error)")
        raise HTTPException(status_code=500, detail="Internal error while initializing agent.")

    try:
        report = agent.draft_report(request)
        logger.info("Report drafted successfully")
        return report
    except Exception:
        # Internal logs include stack trace but must not include PHI or sensitive payloads
        logger.exception("Error drafting report (internal error)")
        # Return a generic error message to the client
        raise HTTPException(status_code=500, detail="Internal error while drafting report.")


# --- Agent 2: Visual Highlighting ---

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

@app.post("/cv/highlight", response_model=CVHighlightResult)
async def cv_highlight(
    request: CVHighlightRequest = Depends(),
    file: UploadFile = File(...)
):
    """
    Highlight anomalies in an uploaded DICOM or image.
    """
    logger.info("Received cv_highlight request: modality=%s", request.modality)
    
    # Read bytes from upload
    try:
        image_bytes = await file.read()
    except Exception:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    agent = get_cv_agent()
    try:
        result = agent.highlight(request, image_bytes)
        return result
    except InvalidDICOMError:
        logger.warning("Invalid DICOM/image uploaded")
        raise HTTPException(status_code=400, detail="Invalid DICOM or image format.")
    except Exception:
        logger.exception("Error in cv_highlight")
        raise HTTPException(status_code=500, detail="Internal error while highlighting image.")


    except Exception:
        logger.exception("Error in cv_highlight")
        raise HTTPException(status_code=500, detail="Internal error while highlighting image.")


# --- Agent 3: Follow-Up & Incidental Findings ---

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


@app.post("/v1/reports/extract_followups", response_model=FollowUpExtractionResponse)
def extract_followups(request: FollowUpExtractionRequest):
    """
    Extract follow-up recommendations and incidental findings.
    """
    logger.info("Received extract_followups request")
    try:
        agent = get_followup_agent()
        response = agent.extract_followups(request)
        return response
    except Exception:
        logger.exception("Error identifying follow-ups")
        raise HTTPException(status_code=500, detail="Internal error while extracting follow-ups.")


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


@app.post("/v1/reports/qa", response_model=ReportQAResponse)
def review_report_qa(request: ReportQARequest):
    """
    Review report for quality, consistency, and completeness.
    """
    logger.info("Received report_qa request")
    try:
        agent = get_report_qa_agent()
        response = agent.review_report(request)
        return response
    except Exception as e:
        # Check if it is our custom JSON error by name (avoids circular import)
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


@app.post("/v1/reports/patient_summary", response_model=PatientReportSummaryResponse)
def generate_patient_summary(request: PatientReportSummaryRequest):
    """
    Generate patient-friendly summary and next steps.
    """
    logger.info("Received patient_summary request")
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



# --- Agent 6: Worklist Triage ---

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


@app.post("/v1/worklist/triage", response_model=WorklistTriageResponse)
def triage_worklist(request: WorklistTriageRequest):
    """
    Agent 6: Triage a worklist of studies to recommend priority.
    """
    logger.info("Received worklist triage request")
    try:
        agent = get_triage_agent()
        return agent.triage(request)
    except Exception:
        logger.exception("Error during worklist triage")
        raise HTTPException(status_code=500, detail="Internal error during triage.")


# --- Agent 7: Learning & Feedback ---

from radiology_assistant.agents.learning_feedback import LearningFeedbackAgent, LearningEventRepository
from radiology_assistant.models import (
    RadiologistLearningDigestRequest, RadiologistLearningDigestResponse, LearningEvent
)

class MockLearningEventRepository:
    """Mock repository for demonstration/testing purposes."""
    def get_events(self, radiologist_id: str, start_date: str, end_date: str) -> list[LearningEvent]:
        # Return empty list or could be seeded with sample data for specific IDs
        return []

_learning_agent: LearningFeedbackAgent | None = None

def get_learning_agent() -> LearningFeedbackAgent:
    global _learning_agent
    if _learning_agent is None:
        try:
            Config.validate()
        except Exception:
            raise
        
        llm_client = LLMClient()
        # In production, this would be a real DB repository
        repository = MockLearningEventRepository()
        _learning_agent = LearningFeedbackAgent(repository, llm_client, logger=logger)
        logger.info("LearningFeedbackAgent initialized")
    return _learning_agent

@app.post("/v1/learning/radiologist_digest", response_model=RadiologistLearningDigestResponse)
def generate_radiologist_digest(request: RadiologistLearningDigestRequest):
    """
    Generate a learning digest for a radiologist.
    """
    logger.info("Received radiologist_digest request")
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

@app.post("/v1/studies/orchestrate", response_model=StudyOrchestrationResponse)
async def orchestrate_study(
    request: StudyOrchestrationRequest,
    agent: StudyOrchestratorAgent = Depends(get_orchestrator_agent)
):
    """
    Agent 8: Orchestrate the entire radiology study workflow.
    Chains CV, Drafting, QA, Follow-up, and Patient Summary.
    """
    try:
        return agent.orchestrate_study(request)
    except Exception as e:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    # Allow running via: python -m radiology_assistant.api
    import uvicorn

    uvicorn.run("radiology_assistant.api:app", host="0.0.0.0", port=8000, reload=True)

