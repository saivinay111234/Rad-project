"""
CME (Continuing Medical Education) Platform Agent.

Generates case-based learning material and MCQ questions from a radiologist's
learning digest, then grades submitted answers and awards CME credits.

Supports AMA PRA Category 1 Credit™ format (0.5-1.0 credits per case).
"""

import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class CMEQuestion(BaseModel):
    """Multiple-choice question for a CME case."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_text: str
    options: Dict[str, str]          # e.g. {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: str              # Key into options (e.g. "B")
    explanation: str
    learning_objective: str


class CMECase(BaseModel):
    """Generated CME case from a radiologist learning digest."""
    case_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    radiologist_id: str
    title: str
    case_description: str
    clinical_context: str
    questions: List[CMEQuestion]
    learning_objectives: List[str]
    credit_points: float = Field(default=0.5, ge=0.25, le=1.0,
                                  description="AMA PRA Category 1 credits awarded on passing")
    passing_score: float = Field(default=0.7, description="Minimum fraction correct to earn credits")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source_digest_period: Optional[str] = None


class CMEGradeResult(BaseModel):
    """Result of grading a radiologist's CME case attempt."""
    case_id: str
    radiologist_id: str
    submitted_answers: Dict[str, str]     # question_id -> chosen_option key
    correct_answers: Dict[str, str]       # question_id -> correct_answer key
    num_questions: int
    num_correct: int
    score: float                          # Fraction (0.0-1.0)
    passed: bool
    credits_earned: float                 # 0.0 if failed, case.credit_points if passed
    feedback: List[str]                   # Per-question feedback strings
    graded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# CME Platform Agent
# ---------------------------------------------------------------------------

_CME_PROMPT_TEMPLATE = """\
You are a radiology CME (Continuing Medical Education) content creator.

Based on the following learning digest themes and error patterns from a radiologist's recent work,
generate ONE clinically realistic CME case with exactly 3 multiple-choice questions.

Learning digest summary:
{digest_summary}

Key error themes:
{error_themes}

Generate a JSON object with this exact structure:
{{
  "title": "Brief descriptive case title",
  "case_description": "2-3 sentence clinical scenario with imaging findings (do NOT reveal diagnosis yet)",
  "clinical_context": "Patient demographics, clinical history, referral reason (100-150 words)",
  "learning_objectives": ["Objective 1", "Objective 2", "Objective 3"],
  "questions": [
    {{
      "question_text": "Clear clinical question",
      "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
      "correct_answer": "B",
      "explanation": "Why B is correct and why the others are wrong (2-4 sentences)",
      "learning_objective": "Which objective this tests"
    }}
  ],
  "credit_points": 0.5
}}

Rules:
- Questions must test clinically important decision points (not trivia)
- Options must be plausible (avoid silly distractors)
- Explanations must reference evidence-based guidelines (Fleischner, BI-RADS, etc.)
- Do not include PHI or reference specific patients
- Output valid JSON only, no extra text
"""


class CMEPlatformAgent:
    """
    Generates case-based CME material from a radiologist's learning digest
    and grades submitted answers.

    Requires an LLMClient for case generation.
    """

    def __init__(self, llm_client, logger: Optional[logging.Logger] = None):
        self.llm = llm_client
        self._logger = logger or logging.getLogger(__name__)

    def generate_cme_case(
        self,
        radiologist_id: str,
        digest,  # RadiologistLearningDigestResponse or compatible
        db_session=None,
    ) -> CMECase:
        """
        Generate a CME case from a radiologist's learning digest.

        Args:
            radiologist_id: The target radiologist ID.
            digest: RadiologistLearningDigestResponse with themes and error patterns.
            db_session: Optional SQLAlchemy session to persist the generated case.

        Returns:
            CMECase with 3 MCQ questions.
        """
        # Extract digest content (handles both Pydantic objects and dicts)
        def _get(obj, key, default=""):
            try:
                val = getattr(obj, key, None)
                return val if val is not None else (obj.get(key, default) if isinstance(obj, dict) else default)
            except Exception:
                return default

        digest_summary = _get(digest, "digest_text") or _get(digest, "summary") or "No digest available."
        themes = _get(digest, "themes") or []
        error_themes = "\n".join(f"- {t}" for t in (themes[:5] if themes else ["General QA improvement"]))
        digest_period = _get(digest, "period") or _get(digest, "date_range") or "recent"

        prompt = _CME_PROMPT_TEMPLATE.format(
            digest_summary=str(digest_summary)[:1000],
            error_themes=error_themes,
        )

        self._logger.info("Generating CME case for radiologist=%s", radiologist_id)

        try:
            raw = self.llm.generate(prompt, temperature=0.4)
            data = self._parse_json(raw)
        except Exception as e:
            self._logger.exception("LLM failed to generate CME case")
            # Return a fallback structured case instead of crashing
            data = self._fallback_case()

        # Build CMEQuestion objects
        questions = []
        for q_data in data.get("questions", [])[:5]:
            questions.append(CMEQuestion(
                question_text=q_data.get("question_text", "Question unavailable."),
                options=q_data.get("options", {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}),
                correct_answer=q_data.get("correct_answer", "A"),
                explanation=q_data.get("explanation", "See relevant guidelines."),
                learning_objective=q_data.get("learning_objective", "General radiology knowledge"),
            ))

        case = CMECase(
            radiologist_id=radiologist_id,
            title=data.get("title", "Radiology CME Case"),
            case_description=data.get("case_description", "Case description unavailable."),
            clinical_context=data.get("clinical_context", ""),
            questions=questions,
            learning_objectives=data.get("learning_objectives", []),
            credit_points=float(data.get("credit_points", 0.5)),
            source_digest_period=digest_period,
        )

        # Optionally persist to DB
        if db_session is not None:
            self._save_to_db(case, db_session)

        return case

    def grade_answers(
        self,
        case: CMECase,
        submitted_answers: Dict[str, str],  # question_id -> option key
        radiologist_id: str,
        db_session=None,
    ) -> CMEGradeResult:
        """
        Grade a radiologist's submitted CME answers.

        Args:
            case: The CMECase being attempted.
            submitted_answers: Map of question_id -> selected option key (e.g. "B").
            radiologist_id: Who is submitting answers.
            db_session: Optional session to update case status in DB.

        Returns:
            CMEGradeResult with score, pass/fail, and per-question feedback.
        """
        correct_map = {q.question_id: q.correct_answer for q in case.questions}
        num_correct = 0
        feedback = []

        for question in case.questions:
            qid = question.question_id
            chosen = submitted_answers.get(qid, "")
            correct = correct_map[qid]
            if chosen.upper() == correct.upper():
                num_correct += 1
                feedback.append(f"Q{qid[:6]}: ✓ Correct. {question.explanation[:100]}")
            else:
                chosen_text = question.options.get(chosen.upper(), "No answer selected")
                correct_text = question.options.get(correct.upper(), "")
                feedback.append(
                    f"Q{qid[:6]}: ✗ You chose {chosen} ({chosen_text[:60]}). "
                    f"Correct: {correct} ({correct_text[:60]}). {question.explanation[:100]}"
                )

        n = len(case.questions)
        score = num_correct / n if n > 0 else 0.0
        passed = score >= case.passing_score
        credits_earned = case.credit_points if passed else 0.0

        self._logger.info(
            "CME graded: radiologist=%s case=%s score=%.1f%% passed=%s credits=%.1f",
            radiologist_id, case.case_id, score * 100, passed, credits_earned
        )

        result = CMEGradeResult(
            case_id=case.case_id,
            radiologist_id=radiologist_id,
            submitted_answers=submitted_answers,
            correct_answers=correct_map,
            num_questions=n,
            num_correct=num_correct,
            score=round(score, 3),
            passed=passed,
            credits_earned=credits_earned,
            feedback=feedback,
        )

        # Update DB status if session provided
        if db_session is not None:
            try:
                from .db_models import CMECaseDB
                row = db_session.query(CMECaseDB).filter(CMECaseDB.id == case.case_id).first()
                if row:
                    row.status = "graded"
                    row.score = score
                    row.graded_at = datetime.now(timezone.utc)
                    db_session.commit()
            except Exception:
                self._logger.exception("Failed to update CME case status in DB")

        return result

    def _parse_json(self, raw: str) -> dict:
        """Extract JSON from LLM response (handles markdown code blocks)."""
        import re
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip().strip("`")
        # Try full parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError("No valid JSON found in LLM response")

    def _fallback_case(self) -> dict:
        """Return a structured fallback CME case if LLM fails."""
        return {
            "title": "Pulmonary Nodule Management",
            "case_description": "A 58-year-old ex-smoker undergoes low-dose chest CT screening. "
                                 "A 7mm solid nodule is identified in the right upper lobe.",
            "clinical_context": "Female patient, 58 years old, 30 pack-year smoking history, "
                                 "quit 3 years ago. No prior CT for comparison.",
            "learning_objectives": [
                "Apply Fleischner 2017 guidelines to solid pulmonary nodule management",
                "Recognize high-risk features that modify follow-up recommendations",
                "Distinguish Lung-RADS categories for screening CTs",
            ],
            "questions": [
                {
                    "question_text": "What is the most appropriate next step for this 7mm solid nodule?",
                    "options": {
                        "A": "No follow-up required",
                        "B": "CT at 6-12 months",
                        "C": "Immediate biopsy",
                        "D": "PET-CT within 1 week",
                    },
                    "correct_answer": "B",
                    "explanation": "Per Fleischner 2017, a 6-8mm solid nodule in a high-risk patient "
                                   "(≥1 risk factor, e.g. smoking) requires CT at 6-12 months then "
                                   "consider CT at 18-24 months if stable.",
                    "learning_objective": "Apply Fleischner 2017 guidelines",
                }
            ],
            "credit_points": 0.5,
        }

    def _save_to_db(self, case: CMECase, db_session) -> None:
        """Persist a CMECase to the database."""
        try:
            from .db_models import CMECaseDB
            row = CMECaseDB(
                id=case.case_id,
                radiologist_id=case.radiologist_id,
                digest_period=case.source_digest_period,
                case_json=case.model_dump_json(),
                credit_points=case.credit_points,
                status="pending",
            )
            db_session.add(row)
            db_session.commit()
            self._logger.debug("CME case saved to DB: id=%s", case.case_id)
        except Exception:
            db_session.rollback()
            self._logger.exception("Failed to save CME case to DB")
