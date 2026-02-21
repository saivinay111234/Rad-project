"""
Repository implementations for the Radiology Assistant.

Provides database-backed implementations of the repository protocols
defined in agents/learning_feedback.py. Replaces the MockLearningEventRepository.
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from .db_models import LearningEventDB, FeedbackEventDB, ReportDB, FollowupReminderDB
from .models import LearningEvent, LearningEventType, DiscrepancySeverity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Learning Event Repository
# ---------------------------------------------------------------------------

class SQLLearningEventRepository:
    """
    SQLAlchemy-backed implementation of the LearningEventRepository protocol.
    Replaces MockLearningEventRepository for production use.
    """

    def __init__(self, db: Session):
        self.db = db

    def get_events(
        self,
        radiologist_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[LearningEvent]:
        """
        Fetch learning events for a radiologist within an optional date range.

        Args:
            radiologist_id: The radiologist's user ID.
            start_date: ISO 8601 date string (inclusive). If None, no lower bound.
            end_date: ISO 8601 date string (inclusive). If None, no upper bound.

        Returns:
            List of LearningEvent Pydantic models.
        """
        query = self.db.query(LearningEventDB).filter(
            LearningEventDB.radiologist_id == radiologist_id
        )
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
                query = query.filter(LearningEventDB.timestamp >= start_dt)
            except ValueError:
                logger.warning("Invalid start_date format: %s", start_date)
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
                query = query.filter(LearningEventDB.timestamp <= end_dt)
            except ValueError:
                logger.warning("Invalid end_date format: %s", end_date)

        rows = query.order_by(LearningEventDB.timestamp.desc()).all()
        return [_db_to_learning_event(row) for row in rows]

    def save(self, event: LearningEvent) -> None:
        """
        Persist a LearningEvent to the database.

        Args:
            event: The LearningEvent Pydantic model to save.
        """
        now = datetime.now(timezone.utc)
        row = LearningEventDB(
            id=event.event_id or str(uuid.uuid4()),
            radiologist_id=event.radiologist_id,
            event_type=event.event_type.value if hasattr(event.event_type, 'value') else str(event.event_type),
            severity=event.severity.value if hasattr(event.severity, 'value') else str(event.severity),
            source=getattr(event, 'source', 'system'),
            modality=event.modality,
            body_region=event.body_region,
            report_text_before=event.report_text_before,
            report_text_after=event.report_text_after,
            timestamp=event.timestamp if isinstance(event.timestamp, datetime) else now,
            hour_of_day=(event.timestamp if isinstance(event.timestamp, datetime) else now).hour,
            day_of_week=(event.timestamp if isinstance(event.timestamp, datetime) else now).weekday(),
        )
        row.set_tags(event.tags or [])
        if event.qa_issues:
            row.set_qa_issues(event.qa_issues)

        self.db.add(row)
        try:
            self.db.commit()
            logger.debug("Saved LearningEvent id=%s for radiologist=%s", row.id, row.radiologist_id)
        except Exception:
            self.db.rollback()
            logger.exception("Failed to save LearningEvent")
            raise


def _db_to_learning_event(row: LearningEventDB) -> LearningEvent:
    """Convert a DB ORM row back to the Pydantic LearningEvent model."""
    return LearningEvent(
        event_id=row.id,
        radiologist_id=row.radiologist_id,
        event_type=LearningEventType(row.event_type) if row.event_type else LearningEventType.QA_ISSUE,
        severity=DiscrepancySeverity(row.severity) if row.severity else DiscrepancySeverity.MINOR,
        modality=row.modality,
        body_region=row.body_region,
        tags=row.get_tags(),
        report_text_before=row.report_text_before,
        report_text_after=row.report_text_after,
        qa_issues=[],  # Complex nested objects — deserialized on demand if needed
        timestamp=row.timestamp,
    )


# ---------------------------------------------------------------------------
# Report Repository
# ---------------------------------------------------------------------------

class ReportRepository:
    """CRUD operations for persisted radiology reports."""

    def __init__(self, db: Session):
        self.db = db

    def save_report(
        self,
        report_text: str,
        study_id: Optional[str] = None,
        radiologist_id: Optional[str] = None,
        modality: Optional[str] = None,
        confidence_score: float = 0.75,
    ) -> str:
        """Persist a report and return its generated ID."""
        report_id = str(uuid.uuid4())
        row = ReportDB(
            id=report_id,
            study_id=study_id,
            radiologist_id=radiologist_id,
            modality=modality,
            report_text=report_text,
            confidence_score=confidence_score,
        )
        self.db.add(row)
        self.db.commit()
        logger.debug("Saved report id=%s study=%s", report_id, study_id)
        return report_id

    def get_report_by_id(self, report_id: str) -> Optional[ReportDB]:
        """Fetch a single report by its ID, or None if not found."""
        return self.db.query(ReportDB).filter(ReportDB.id == report_id).first()


# ---------------------------------------------------------------------------
# Feedback Repository
# ---------------------------------------------------------------------------

class FeedbackRepository:
    """CRUD operations for radiologist feedback events."""

    def __init__(self, db: Session):
        self.db = db

    def save_feedback(
        self,
        report_id: str,
        radiologist_id: str,
        correction_text: str,
        correction_type: str = "general",
        learning_event_id: Optional[str] = None,
    ) -> str:
        """Save feedback and return its generated ID."""
        feedback_id = str(uuid.uuid4())
        row = FeedbackEventDB(
            id=feedback_id,
            report_id=report_id,
            radiologist_id=radiologist_id,
            correction_type=correction_type,
            correction_text=correction_text,
            learning_event_id=learning_event_id,
        )
        self.db.add(row)
        self.db.commit()
        logger.debug("Saved feedback id=%s for report=%s", feedback_id, report_id)
        return feedback_id

    def get_feedback_by_report(self, report_id: str) -> List[FeedbackEventDB]:
        """Get all feedback entries for a given report."""
        return (
            self.db.query(FeedbackEventDB)
            .filter(FeedbackEventDB.report_id == report_id)
            .order_by(FeedbackEventDB.created_at.desc())
            .all()
        )


# ---------------------------------------------------------------------------
# Followup Reminder Repository
# ---------------------------------------------------------------------------

class FollowupReminderRepository:
    """CRUD operations for follow-up reminders."""

    def __init__(self, db: Session):
        self.db = db

    def save_reminder(self, reminder: FollowupReminderDB) -> str:
        """Persist a FollowupReminderDB row and return its ID."""
        self.db.add(reminder)
        self.db.commit()
        return reminder.id

    def get_due_reminders(self, within_days: int = 7) -> List[FollowupReminderDB]:
        """Return reminders due within the next `within_days` days."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) + timedelta(days=within_days)
        return (
            self.db.query(FollowupReminderDB)
            .filter(
                FollowupReminderDB.status == "pending",
                FollowupReminderDB.due_date <= cutoff,
            )
            .order_by(FollowupReminderDB.due_date.asc())
            .all()
        )

    def get_by_study(self, study_id: str) -> List[FollowupReminderDB]:
        """Get all reminders for a study."""
        return (
            self.db.query(FollowupReminderDB)
            .filter(FollowupReminderDB.study_id == study_id)
            .all()
        )
