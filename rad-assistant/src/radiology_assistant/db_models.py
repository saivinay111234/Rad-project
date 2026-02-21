"""
SQLAlchemy ORM models for the Radiology Assistant.

These map Python classes to database tables. All models use the shared
Base from database.py. Run `init_db()` or Alembic migrations to create tables.
"""

import json
from datetime import datetime, timezone
from sqlalchemy import Column, String, Text, Float, Boolean, Integer, DateTime, Enum as SAEnum
from sqlalchemy.orm import mapped_column, Mapped
from typing import Optional

from .database import Base
from .auth import UserRole


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

class UserDB(Base):
    """Persisted user account."""
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    username: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default=UserRole.RADIOLOGIST.value)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

class ReportDB(Base):
    """Persisted radiology report draft."""
    __tablename__ = "reports"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    radiologist_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    modality: Mapped[Optional[str]] = mapped_column(String(16))
    report_text: Mapped[str] = mapped_column(Text, nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.75)
    key_findings_json: Mapped[Optional[str]] = mapped_column(Text)  # JSON array
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

    def set_key_findings(self, findings: list) -> None:
        self.key_findings_json = json.dumps([f.model_dump() if hasattr(f, 'model_dump') else f for f in findings])

    def get_key_findings(self) -> list:
        return json.loads(self.key_findings_json) if self.key_findings_json else []


# ---------------------------------------------------------------------------
# Learning Events
# ---------------------------------------------------------------------------

class LearningEventDB(Base):
    """Persisted learning event (QA issue, peer review discrepancy, etc.)."""
    __tablename__ = "learning_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)           # event_id
    radiologist_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    severity: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), default="system")
    modality: Mapped[Optional[str]] = mapped_column(String(16))
    body_region: Mapped[Optional[str]] = mapped_column(String(64))
    tags_json: Mapped[Optional[str]] = mapped_column(Text)                   # JSON array of strings
    report_text_before: Mapped[Optional[str]] = mapped_column(Text)
    report_text_after: Mapped[Optional[str]] = mapped_column(Text)
    qa_issues_json: Mapped[Optional[str]] = mapped_column(Text)              # JSON array of QAIssue dicts
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    hour_of_day: Mapped[int] = mapped_column(Integer, default=0)            # for fatigue analysis
    day_of_week: Mapped[int] = mapped_column(Integer, default=0)            # 0=Mon, 6=Sun

    def set_tags(self, tags: list) -> None:
        self.tags_json = json.dumps(tags)

    def get_tags(self) -> list:
        return json.loads(self.tags_json) if self.tags_json else []

    def set_qa_issues(self, issues: list) -> None:
        self.qa_issues_json = json.dumps([i.model_dump() if hasattr(i, 'model_dump') else i for i in issues])


# ---------------------------------------------------------------------------
# Feedback Events
# ---------------------------------------------------------------------------

class FeedbackEventDB(Base):
    """Radiologist correction/feedback on a specific report."""
    __tablename__ = "feedback_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    report_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    radiologist_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    correction_type: Mapped[str] = mapped_column(String(64), default="general")
    correction_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Auto-created learning_event_id after feedback is processed
    learning_event_id: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

class AuditLogDB(Base):
    """Immutable audit trail for HIPAA compliance — who accessed what."""
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    username: Mapped[Optional[str]] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(256), nullable=False)  # e.g. "POST /v1/reports/draft"
    resource: Mapped[Optional[str]] = mapped_column(String(256))      # e.g. study_id or report_id
    ip_address: Mapped[Optional[str]] = mapped_column(String(64))
    status_code: Mapped[Optional[int]] = mapped_column(Integer)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)


# ---------------------------------------------------------------------------
# Follow-up Reminders
# ---------------------------------------------------------------------------

class FollowupReminderDB(Base):
    """Scheduled follow-up reminder for a patient/study."""
    __tablename__ = "followup_reminders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    study_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    patient_token: Mapped[Optional[str]] = mapped_column(String(128))     # de-identified patient reference
    finding_id: Mapped[Optional[str]] = mapped_column(String(64))         # IncidentalFinding.id
    followup_type: Mapped[str] = mapped_column(String(32), default="imaging")
    followup_modality: Mapped[Optional[str]] = mapped_column(String(32))
    followup_text: Mapped[str] = mapped_column(Text, nullable=False)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")    # pending | sent | overdue | done
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


# ---------------------------------------------------------------------------
# CME Cases
# ---------------------------------------------------------------------------

class CMECaseDB(Base):
    """Generated CME (Continuing Medical Education) case."""
    __tablename__ = "cme_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    radiologist_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    digest_period: Mapped[Optional[str]] = mapped_column(String(32))
    case_json: Mapped[str] = mapped_column(Text, nullable=False)          # full CMECase JSON
    credit_points: Mapped[float] = mapped_column(Float, default=0.5)
    status: Mapped[str] = mapped_column(String(32), default="pending")    # pending | graded | expired
    score: Mapped[Optional[float]] = mapped_column(Float)
    graded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
