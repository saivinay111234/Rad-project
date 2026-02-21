"""
Follow-up Reminder Engine — Follow-up Failure Prevention.

Schedules, tracks, and surfaces upcoming follow-up reminders based on
incidental findings extracted from radiology reports. Prevents patient
follow-up failures by maintaining a persistent reminder table.

Clinical motivation: Studies show ~30% of recommended follow-up imaging
is never performed, leading to delayed diagnosis of early-stage cancers.
"""

import uuid
import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Followup interval parsing
# ---------------------------------------------------------------------------

# Maps common Fleischner/guideline interval descriptions to approximate days
_INTERVAL_MAP = {
    "3 months": 90,
    "6 months": 180,
    "3-6 months": 90,  # Conservative: use shorter interval
    "6-12 months": 180,
    "12 months": 365,
    "annual": 365,
    "annually": 365,
    "1 year": 365,
    "2 years": 730,
    "18-24 months": 540,
    "immediately": 1,
    "urgent": 3,
    "routine": 180,
}


def _parse_interval_days(interval_text: Optional[str]) -> Optional[int]:
    """
    Parse a natural-language follow-up interval to a number of days.

    Args:
        interval_text: e.g., "6 months", "3-6 months", "annual"

    Returns:
        Number of days until follow-up due, or None if unparseable.
    """
    if not interval_text:
        return None

    text_lower = interval_text.strip().lower()
    for pattern, days in _INTERVAL_MAP.items():
        if pattern in text_lower:
            return days

    # Try to parse "N months" or "N weeks" patterns directly
    import re
    m = re.search(r"(\d+)\s*months?", text_lower)
    if m:
        return int(m.group(1)) * 30

    m = re.search(r"(\d+)\s*weeks?", text_lower)
    if m:
        return int(m.group(1)) * 7

    m = re.search(r"(\d+)\s*years?", text_lower)
    if m:
        return int(m.group(1)) * 365

    return None


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class FollowupReminderOut(BaseModel):
    """API response shape for a scheduled reminder."""
    reminder_id: str
    study_id: str
    finding_id: Optional[str] = None
    followup_type: str
    followup_modality: Optional[str] = None
    followup_text: str
    due_date: Optional[datetime] = None
    status: str = "pending"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ScheduleRemindersResponse(BaseModel):
    """Response returned from POST /v1/followup/schedule."""
    study_id: str
    scheduled_count: int
    skipped_count: int
    reminders: List[FollowupReminderOut]
    message: str


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FollowupReminderEngine:
    """
    Converts incidental findings into scheduled follow-up reminders.

    Works with the FollowupReminderDB table via FollowupReminderRepository.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def schedule_reminders(
        self,
        study_id: str,
        findings: list,  # List[IncidentalFinding]
        patient_token: Optional[str],
        db_session,
    ) -> ScheduleRemindersResponse:
        """
        Create follow-up reminders from a list of IncidentalFinding objects.

        Args:
            study_id: The study being analyzed.
            findings: List of IncidentalFinding Pydantic models.
            patient_token: De-identified patient identifier (no MRN/PHI).
            db_session: SQLAlchemy Session for persistence.

        Returns:
            ScheduleRemindersResponse with created reminders.
        """
        from .db_models import FollowupReminderDB

        scheduled: List[FollowupReminderOut] = []
        skipped = 0

        for finding in findings:
            # Only schedule for findings that require follow-up
            if not getattr(finding, "followup_required", False):
                skipped += 1
                continue

            interval_text = getattr(finding, "followup_interval", None)
            days = _parse_interval_days(interval_text)
            due_date = (
                datetime.now(timezone.utc) + timedelta(days=days)
                if days is not None
                else None
            )

            reminder_id = str(uuid.uuid4())
            followup_type = getattr(finding, "followup_type", "imaging") or "imaging"
            followup_modality = getattr(finding, "followup_modality", None)
            followup_text = (
                getattr(finding, "followup_details", None)
                or getattr(finding, "description", "Follow-up required.")
            )

            row = FollowupReminderDB(
                id=reminder_id,
                study_id=study_id,
                patient_token=patient_token,
                finding_id=getattr(finding, "id", None),
                followup_type=followup_type,
                followup_modality=followup_modality,
                followup_text=followup_text,
                due_date=due_date,
                status="pending",
            )
            db_session.add(row)

            scheduled.append(FollowupReminderOut(
                reminder_id=reminder_id,
                study_id=study_id,
                finding_id=getattr(finding, "id", None),
                followup_type=followup_type,
                followup_modality=followup_modality,
                followup_text=followup_text,
                due_date=due_date,
            ))

        if scheduled:
            try:
                db_session.commit()
                self._logger.info(
                    "Scheduled %d follow-up reminders for study=%s",
                    len(scheduled), study_id
                )
            except Exception:
                db_session.rollback()
                self._logger.exception("Failed to commit follow-up reminders")
                raise

        return ScheduleRemindersResponse(
            study_id=study_id,
            scheduled_count=len(scheduled),
            skipped_count=skipped,
            reminders=scheduled,
            message=(
                f"Scheduled {len(scheduled)} reminder(s). "
                f"{skipped} finding(s) did not require follow-up."
            ),
        )

    def get_due_reminders(
        self,
        db_session,
        within_days: int = 7,
    ) -> List[FollowupReminderOut]:
        """
        Fetch all pending reminders due within the next `within_days` days.

        Args:
            db_session: SQLAlchemy Session.
            within_days: Look-ahead window (default 7 days).

        Returns:
            List of FollowupReminderOut models ordered by due date ascending.
        """
        from .repositories import FollowupReminderRepository

        repo = FollowupReminderRepository(db_session)
        rows = repo.get_due_reminders(within_days=within_days)

        return [
            FollowupReminderOut(
                reminder_id=row.id,
                study_id=row.study_id,
                finding_id=row.finding_id,
                followup_type=row.followup_type,
                followup_modality=row.followup_modality,
                followup_text=row.followup_text,
                due_date=row.due_date,
                status=row.status,
                created_at=row.created_at,
            )
            for row in rows
        ]
