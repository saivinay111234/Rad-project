import logging
import sys
import json
from datetime import datetime

# Setup path to import src
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from radiology_assistant.agents.learning_feedback import LearningFeedbackAgent, LearningEventRepository
from radiology_assistant.models import (
    RadiologistLearningDigestRequest, LearningEvent, LearningEventType, 
    DiscrepancySeverity, ExamMetadata, QAIssue, QAType, QASeverity, QASection, QAChangeType
)
from radiology_assistant.llm_client import LLMClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ManualTest")

class ManualMockRepo:
    def get_events(self, radiologist_id, start_date, end_date):
        logger.info(f"Fetching events for {radiologist_id}...")
        
        # Create some interesting mock events
        evt1 = LearningEvent(
            event_id="evt_missed_pna",
            radiologist_id=radiologist_id,
            exam_metadata=ExamMetadata(modality="DX", body_region="Chest", exam_date="2025-01-02"),
            event_type=LearningEventType.QA_ISSUE,
            severity=DiscrepancySeverity.MAJOR,
            source="qa_agent",
            timestamp="2025-01-02T14:00:00",
            report_text_before="Lungs are clear.",
            tags=["pneumonia", "missed_finding"],
            qa_issues=[
                QAIssue(id="qa1", severity=QASeverity.MAJOR, type=QAType.CLARITY, section=QASection.FINDINGS, 
                        description="Potential opacity in LLL missed.", suggested_change_type=QAChangeType.NOTE_ONLY,
                        location_hint=None, suggested_text=None)
            ]
        )
        
        evt2 = LearningEvent(
            event_id="evt_addendum_nodule",
            radiologist_id=radiologist_id,
            exam_metadata=ExamMetadata(modality="CT", body_region="Chest", exam_date="2025-01-03"),
            event_type=LearningEventType.ADDENDUM_CORRECTION,
            severity=DiscrepancySeverity.MINOR,
            source="system",
            timestamp="2025-01-03T09:00:00",
            report_text_before="No nodules.",
            report_text_after="Addendum: small 3mm nodule RUL.",
            tags=["pulmonary_nodule", "addendum"]
        )
        
        evt3 = LearningEvent(
            event_id="evt_peer_review_ok",
            radiologist_id=radiologist_id,
            exam_metadata=ExamMetadata(modality="MR", body_region="Brain", exam_date="2025-01-05"),
            event_type=LearningEventType.INTERESTING_CASE,
            severity=DiscrepancySeverity.INFO,
            source="peer_review",
            timestamp="2025-01-05T11:00:00",
            tags=["rare_variant", "interesting"],
            report_text_before="Complex case of ..."
        )
        
        return [evt1, evt2, evt3]

def main():
    print("--- Starting Manual Learning Digest Test ---")
    
    # Init
    repo = ManualMockRepo()
    llm_client = LLMClient() # Will use real or mock depending on env, but let's assume it works or fails soft
    agent = LearningFeedbackAgent(repo, llm_client, logger=logger)
    
    req = RadiologistLearningDigestRequest(
        radiologist_id="dr_smith",
        start_date="2025-01-01",
        end_date="2025-01-07",
        include_raw_snippets=True
    )
    
    print(f"Generating digest for {req.radiologist_id}...")
    try:
        response = agent.generate_radiologist_digest(req)
        
        print("\n=== DIGEST GENERATED ===")
        print(f"Version: {response.version}")
        print(f"Summary: {response.summary_text}")
        print(f"Stats: Total Events={response.stats.num_total_events}, Critical={response.stats.num_critical}")
        
        print("\n--- Key Themes ---")
        for t in response.key_themes:
            print(f"* {t.name}: {t.description}")
            
        print("\n--- Cases ---")
        for c in response.cases:
            print(f"- [{c.severity}] {c.short_description}")
            print(f"  Lesson: {c.key_lesson}")
            
        print("\nTest Completed Successfully.")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
