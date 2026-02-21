"""
Celery tasks for the Radiology Assistant.
"""

import logging
from .celery_app import app as celery_app
from .models import StudyOrchestrationRequest, StudyOrchestrationResponse
from .database import SessionLocal
from .repositories import ReportRepository, SQLLearningEventRepository

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="radiology_assistant.tasks.orchestrate_study_task")
def orchestrate_study_task(self, request_dict: dict):
    """
    Background task to run the full study orchestration pipeline.
    """
    from .agents.study_orchestrator import StudyOrchestratorAgent
    from .api import get_orchestrator_agent
    
    task_id = self.request.id
    logger.info("Starting background orchestration task_id=%s study_id=%s", 
                task_id, request_dict.get("study_id"))

    # 1. Deserialize request
    try:
        request = StudyOrchestrationRequest(**request_dict)
    except Exception as e:
        logger.error("Failed to parse orchestration request in task: %s", e)
        return {"error": f"Invalid request dictionary: {e}"}

    # 2. Get the agent (singleton)
    # NOTE: In a worker process, we might want to ensure a fresh session or agent
    try:
        agent = get_orchestrator_agent()
    except Exception as e:
        logger.error("Failed to initialize orchestrator agent in worker: %s", e)
        return {"error": "Internal agent initialization failure."}

    # 3. Process the study
    try:
        # The orchestrator should handle its own error reporting in the response
        response: StudyOrchestrationResponse = agent.orchestrate_study(request)
        
        # 4. Persistence (Optional: if the agent doesn't do it)
        # We can also save the final bundle to a 'Results' table here if needed
        # row = ResultsDB(task_id=task_id, result_json=response.model_dump_json())
        # db.add(row)...
        
        return response.model_dump()
    except Exception as e:
        logger.exception("Unexpected error during background orchestration")
        return {"error": f"Orchestration failed: {str(e)}"}
