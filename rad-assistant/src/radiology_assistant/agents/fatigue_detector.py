"""
Radiologist Fatigue Detection Agent.

Analyzes historical learning events (QA flags, peer review discrepancies,
addendum corrections) to identify time-of-day and day-of-week patterns
where a radiologist's error rate is elevated — a proxy for fatigue.

Clinical motivation: Studies show diagnostic error rates increase after
3+ hours of continuous reading and during overnight shifts (0-6 AM).
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from collections import defaultdict

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output Models
# ---------------------------------------------------------------------------

class FatigueTimeSlot(BaseModel):
    """Error concentration in a specific time window."""
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour (0-23 UTC)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day (0=Mon, 6=Sun)")
    error_count: int
    total_events: int
    error_rate: float = Field(..., description="Error fraction (0.0-1.0)")
    fatigue_risk_score: float = Field(..., description="Composite risk score (0.0-1.0)")
    label: str = Field(..., description="Human-readable time label")

    @property
    def is_high_risk(self) -> bool:
        return self.fatigue_risk_score >= 0.6


class FatigueReport(BaseModel):
    """Fatigue analysis result for a single radiologist."""
    radiologist_id: str
    analysis_period_days: int
    total_events_analyzed: int
    total_errors: int
    overall_error_rate: float
    high_risk_slots: List[FatigueTimeSlot]
    peer_review_recommended: bool
    summary: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

# Error event types — learning events of these types count as "errors"
_ERROR_EVENT_TYPES = {
    "qa_flag",
    "peer_review_discrepancy",
    "addendum_correction",
    "critical_miss",
    "near_miss",
}

# Hours with inherently higher fatigue risk (overnight, end of long shift)
_HIGH_FATIGUE_HOURS = {0, 1, 2, 3, 4, 5, 23}  # midnight to 5am + 11pm

_DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_SHIFT_LABELS = {
    range(0, 6): "Night Shift",
    range(6, 12): "Morning Shift",
    range(12, 18): "Afternoon Shift",
    range(18, 24): "Evening Shift",
}


def _get_shift_label(hour: int) -> str:
    for r, label in _SHIFT_LABELS.items():
        if hour in r:
            return label
    return "Unknown"


class RadiologistFatigueDetector:
    """
    Analyzes QA learning events to surface fatigue risk patterns.

    Works by:
    1. Loading all LearningEvents for a radiologist in the past N days
    2. Bucketing them by (hour_of_day, day_of_week)
    3. Computing error_rate = error_events / total_events per bucket
    4. Computing fatigue_risk_score = error_rate × volume_weight × time_of_day_weight
    5. Returning high-risk time slots and a recommendation

    Requires the database session to query LearningEventDB rows directly
    (bypasses the repository to access hour_of_day / day_of_week columns).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self._logger = logger or logging.getLogger(__name__)

    def analyze(
        self,
        radiologist_id: str,
        db_session,
        analysis_period_days: int = 30,
        min_events_per_slot: int = 3,
    ) -> FatigueReport:
        """
        Run fatigue analysis for a radiologist.

        Args:
            radiologist_id: The radiologist's user ID.
            db_session: SQLAlchemy Session with access to LearningEventDB.
            analysis_period_days: How many days back to analyze (default 30).
            min_events_per_slot: Minimum events in a time slot to include in analysis.

        Returns:
            FatigueReport with high-risk slots and recommendations.
        """
        from .db_models import LearningEventDB

        cutoff = datetime.now(timezone.utc) - timedelta(days=analysis_period_days)

        rows = (
            db_session.query(LearningEventDB)
            .filter(
                LearningEventDB.radiologist_id == radiologist_id,
                LearningEventDB.timestamp >= cutoff,
            )
            .all()
        )

        self._logger.info(
            "Fatigue analysis for %s: %d events in past %d days",
            radiologist_id, len(rows), analysis_period_days
        )

        if not rows:
            return FatigueReport(
                radiologist_id=radiologist_id,
                analysis_period_days=analysis_period_days,
                total_events_analyzed=0,
                total_errors=0,
                overall_error_rate=0.0,
                high_risk_slots=[],
                peer_review_recommended=False,
                summary=(
                    f"No learning events found for radiologist {radiologist_id} "
                    f"in the past {analysis_period_days} days."
                ),
            )

        # ------------------------------------------------------------------
        # Bucket events by (hour, day_of_week)
        # ------------------------------------------------------------------
        bucket_total: Dict[tuple, int] = defaultdict(int)
        bucket_errors: Dict[tuple, int] = defaultdict(int)
        total_errors = 0

        for row in rows:
            key = (row.hour_of_day, row.day_of_week)
            bucket_total[key] += 1
            if row.event_type in _ERROR_EVENT_TYPES:
                bucket_errors[key] += 1
                total_errors += 1

        # ------------------------------------------------------------------
        # Score each bucket
        # ------------------------------------------------------------------
        total_events = len(rows)
        overall_error_rate = total_errors / total_events if total_events > 0 else 0.0
        max_volume = max(bucket_total.values()) if bucket_total else 1

        high_risk_slots: List[FatigueTimeSlot] = []

        for (hour, dow), total in bucket_total.items():
            if total < min_events_per_slot:
                continue

            errors = bucket_errors.get((hour, dow), 0)
            error_rate = errors / total

            # Fatigue risk score is a weighted composite:
            # - error_rate: direct error signal
            # - volume_weight: high-volume slots matter more
            # - time_weight: overnight hours carry extra risk
            volume_weight = total / max_volume
            time_weight = 1.3 if hour in _HIGH_FATIGUE_HOURS else 1.0
            fatigue_risk = min(1.0, error_rate * volume_weight * time_weight)

            day_name = _DAY_NAMES[dow]
            shift_label = _get_shift_label(hour)
            label = f"{day_name} {hour:02d}:00 ({shift_label})"

            slot = FatigueTimeSlot(
                hour_of_day=hour,
                day_of_week=dow,
                error_count=errors,
                total_events=total,
                error_rate=round(error_rate, 3),
                fatigue_risk_score=round(fatigue_risk, 3),
                label=label,
            )
            if slot.is_high_risk:
                high_risk_slots.append(slot)

        high_risk_slots.sort(key=lambda s: s.fatigue_risk_score, reverse=True)
        peer_review_recommended = len(high_risk_slots) >= 2 or overall_error_rate >= 0.15

        # Build summary
        if high_risk_slots:
            top = high_risk_slots[0]
            summary = (
                f"Analysis of {total_events} events over {analysis_period_days} days. "
                f"Overall error rate: {overall_error_rate:.1%}. "
                f"Highest-risk time slot: {top.label} "
                f"(score={top.fatigue_risk_score:.2f}, error_rate={top.error_rate:.1%}). "
            )
            if peer_review_recommended:
                summary += " ⚠️ Peer review recommended during high-risk windows."
        else:
            summary = (
                f"Analysis of {total_events} events over {analysis_period_days} days. "
                f"Overall error rate: {overall_error_rate:.1%}. No high-risk time slots detected."
            )

        return FatigueReport(
            radiologist_id=radiologist_id,
            analysis_period_days=analysis_period_days,
            total_events_analyzed=total_events,
            total_errors=total_errors,
            overall_error_rate=round(overall_error_rate, 4),
            high_risk_slots=high_risk_slots[:5],  # Top 5 worst slots
            peer_review_recommended=peer_review_recommended,
            summary=summary,
        )
