import logging
import json
from typing import List, Optional, Dict, Any, Protocol
from datetime import datetime
from collections import defaultdict

from radiology_assistant.models import (
    LearningEvent, RadiologistLearningDigestRequest, RadiologistLearningDigestResponse,
    LearningCaseSnippet, LearningTheme, LearningStats,
    LearningEventType, DiscrepancySeverity, QAIssue
)
from radiology_assistant.llm_client import LLMClient

# Interface for data access (to be implemented by a real DB adapter)
class LearningEventRepository(Protocol):
    def get_events(self, radiologist_id: str, start_date: str, end_date: str) -> List[LearningEvent]:
        ...

class LearningFeedbackAgent:
    """
    Agent 7: Learning & Feedback / Case Digest Agent.
    Aggregates historical learning events and produces a constructive, non-punitive digest.
    """
    
    def __init__(self, repository: LearningEventRepository, llm_client: LLMClient, logger: Optional[logging.Logger] = None):
        self.repository = repository
        self.llm_client = llm_client
        self.logger = logger or logging.getLogger(__name__)

    def generate_radiologist_digest(self, request: RadiologistLearningDigestRequest) -> RadiologistLearningDigestResponse:
        """
        Main entry point to generate a learning digest for a radiologist.
        """
        self.logger.info(f"Generating learning digest for radiologist {request.radiologist_id} "
                         f"from {request.start_date} to {request.end_date}")

        # 1. Fetch Events
        events = self.repository.get_events(request.radiologist_id, request.start_date, request.end_date)
        
        # Filter by modality/body_region if requested
        if request.modality_filter:
            events = [e for e in events if e.exam_metadata.modality in request.modality_filter]
        if request.body_region_filter:
            events = [e for e in events if e.exam_metadata.body_region in request.body_region_filter]
        
        # Filter qc_only if requested
        if request.include_qc_only:
            qc_types = {
                LearningEventType.ADDENDUM_CORRECTION, 
                LearningEventType.QA_ISSUE, 
                LearningEventType.PEER_REVIEW_DISCREPANCY,
                LearningEventType.MISSED_FINDING,
                LearningEventType.OVER_CALL
            }
            events = [e for e in events if e.event_type in qc_types]

        total_events_fetched = len(events)
        self.logger.info(f"Fetched {total_events_fetched} events after filtering.")

        # 2. Handle No Events
        if not events:
            return self._create_empty_digest(request)

        # 3. Score & Select Cases
        selected_events = self._score_and_select_cases(events, request.max_cases)
        
        # 4. Group into Themes (Deterministic Baseline)
        themes_map = self._group_themes_deterministically(selected_events)
        
        # 5. Calculate Stats
        stats = self._calculate_stats(events, len(selected_events))

        # 6. Generate Digest Content (LLM or Fallback)
        try:
            digest_content = self._generate_llm_digest(request, selected_events, themes_map, stats)
            generation_method = "llm"
        except Exception as e:
            self.logger.error(f"LLM generation failed: {e}. Falling back to deterministic digest.")
            digest_content = self._generate_fallback_digest(request, selected_events, themes_map, stats)
            generation_method = "fallback"

        # 7. Build Final Response
        response = RadiologistLearningDigestResponse(
            version="v1",
            radiologist_id=request.radiologist_id,
            start_date=request.start_date,
            end_date=request.end_date,
            language_code=request.language_code,
            summary_text=digest_content['summary_text'],
            key_themes=digest_content['key_themes'],
            cases=digest_content['cases'],
            stats=stats,
            generation_metadata={
                "generated_at": datetime.now().isoformat(),
                "events_considered": total_events_fetched,
                "generation_method": generation_method
            }
        )
        
        return response

    def _create_empty_digest(self, request: RadiologistLearningDigestRequest) -> RadiologistLearningDigestResponse:
        return RadiologistLearningDigestResponse(
            version="v1",
            radiologist_id=request.radiologist_id,
            start_date=request.start_date,
            end_date=request.end_date,
            language_code=request.language_code,
            summary_text="No significant learning events found for this period.",
            key_themes=[],
            cases=[],
            stats=LearningStats(
                num_total_events=0, num_critical=0, num_major=0, num_minor=0,
                num_addenda=0, num_peer_review_discrepancies=0, num_cases_in_digest=0
            ),
            generation_metadata={
                "generated_at": datetime.now().isoformat(),
                "events_considered": 0,
                "generation_method": "empty"
            }
        )

    def _score_and_select_cases(self, events: List[LearningEvent], max_cases: int) -> List[LearningEvent]:
        """Scores events by severity/importance and selects top N."""
        def get_score(e: LearningEvent) -> int:
            score = 0
            # Base severity
            if e.severity == DiscrepancySeverity.CRITICAL: score += 10
            elif e.severity == DiscrepancySeverity.MAJOR: score += 5
            elif e.severity == DiscrepancySeverity.MINOR: score += 2
            
            # Event type boosters
            if e.event_type == LearningEventType.PEER_REVIEW_DISCREPANCY: score += 3
            if e.event_type == LearningEventType.MISSED_FINDING: score += 3
            if e.event_type == LearningEventType.INTERESTING_CASE: score += 1
            
            return score

        # Sort by score desc, then by date desc
        scored_events = sorted(events, key=lambda e: (get_score(e), e.timestamp), reverse=True)
        return scored_events[:max_cases]

    def _group_themes_deterministically(self, events: List[LearningEvent]) -> Dict[str, List[LearningEvent]]:
        """Groups events by simple tags or body region as a baseline."""
        themes = defaultdict(list)
        for e in events:
            # Use first tag if available, else body region, else "General"
            key = "General"
            if e.tags:
                key = e.tags[0]
            elif e.exam_metadata.body_region:
                key = e.exam_metadata.body_region
            themes[key].append(e)
        return themes

    def _calculate_stats(self, all_events: List[LearningEvent], num_in_digest: int) -> LearningStats:
        stats = LearningStats(
            num_total_events=len(all_events),
            num_critical=0, num_major=0, num_minor=0,
            num_addenda=0, num_peer_review_discrepancies=0,
            num_cases_in_digest=num_in_digest
        )
        for e in all_events:
            if e.severity == DiscrepancySeverity.CRITICAL: stats.num_critical += 1
            elif e.severity == DiscrepancySeverity.MAJOR: stats.num_major += 1
            elif e.severity == DiscrepancySeverity.MINOR: stats.num_minor += 1
            
            if e.event_type == LearningEventType.ADDENDUM_CORRECTION: stats.num_addenda += 1
            if e.event_type == LearningEventType.PEER_REVIEW_DISCREPANCY: stats.num_peer_review_discrepancies += 1
        return stats

    def _generate_llm_digest(self, request: RadiologistLearningDigestRequest, 
                             selected_events: List[LearningEvent], 
                             themes_map: Dict[str, List[LearningEvent]],
                             stats: LearningStats) -> Dict[str, Any]:
        
        # Construct prompt data (No PHI)
        cases_data = []
        for e in selected_events:
            case_info = {
                "id": e.event_id,
                "type": e.event_type,
                "severity": e.severity,
                "modality": e.exam_metadata.modality,
                "body_region": e.exam_metadata.body_region,
                "tags": e.tags,
                "qa_issues": [i.description for i in e.qa_issues] if e.qa_issues else [],
                "snippet": e.report_text_before[:200] if request.include_raw_snippets and e.report_text_before else "N/A"
            }
            cases_data.append(case_info)

        system_prompt = (
            "You are a helpful, constructive senior radiologist mentor. "
            "Your goal is to create a learning digest for a colleague based on recent cases. "
            "TONE: Constructive, educational, non-punitive. Do not assign blame. Focus on improvement. "
            "CONSTRAINT: Do not include any PHI (names, dates, MRNs). "
            "OUTPUT: Valid JSON matching the schema for summary_text, key_themes, and cases."
        )
        
        user_prompt = f"""
        Here are the key cases for review (Total events considered: {stats.num_total_events}):
        {json.dumps(cases_data, indent=2)}

        Task:
        1. Write a short, encouraging summary text ({request.language_code}).
        2. Identify key themes based on the cases.
        3. For each case, provide a short 'key_lesson' and a 'short_description'.

        Return JSON format:
        {{
            "summary_text": "...",
            "key_themes": [
                {{"theme_id": "...", "name": "...", "description": "...", "event_ids": ["..."], "suggested_actions": ["..."]}}
            ],
            "cases": [
                {{"event_id": "...", "short_description": "...", "key_lesson": "..."}}
            ]
        }}
        """

        response_text = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        # Clean markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.strip("```json").strip("```")
        
        data = json.loads(response_text)
        
        # Hydrate objects
        themes = []
        for t in data.get("key_themes", []):
            themes.append(LearningTheme(**t))
            
        final_cases = []
        case_lookup = {c["event_id"]: c for c in data.get("cases", [])}
        
        for e in selected_events:
            llm_case = case_lookup.get(e.event_id, {})
            final_cases.append(LearningCaseSnippet(
                event_id=e.event_id,
                exam_metadata=e.exam_metadata,
                event_type=e.event_type,
                severity=e.severity,
                tags=e.tags,
                short_description=llm_case.get("short_description", f"{e.event_type} in {e.exam_metadata.body_region}"),
                key_lesson=llm_case.get("key_lesson", "Review case for details."),
                report_snippet_before=e.report_text_before if request.include_raw_snippets else None
            ))
            
        return {
            "summary_text": data.get("summary_text", "Digest generated."),
            "key_themes": themes,
            "cases": final_cases
        }

    def _generate_fallback_digest(self, request: RadiologistLearningDigestRequest,
                                  selected_events: List[LearningEvent],
                                  themes_map: Dict[str, List[LearningEvent]],
                                  stats: LearningStats) -> Dict[str, Any]:
        """Deterministic fallback if LLM fails."""
        
        summary = f"Review of {stats.num_total_events} events. Focused on {len(selected_events)} key cases."
        
        themes = []
        for name, events in themes_map.items():
            themes.append(LearningTheme(
                theme_id=name.lower().replace(" ", "_"),
                name=name,
                description=f"Group of {len(events)} cases related to {name}.",
                event_ids=[e.event_id for e in events],
                suggested_actions=["Review these cases for common patterns."]
            ))
            
        cases = []
        for e in selected_events:
            cases.append(LearningCaseSnippet(
                event_id=e.event_id,
                exam_metadata=e.exam_metadata,
                event_type=e.event_type,
                severity=e.severity,
                tags=e.tags,
                short_description=f"{e.severity} {e.event_type} - {e.exam_metadata.body_region}",
                key_lesson="Check original report for details.",
                report_snippet_before=e.report_text_before if request.include_raw_snippets else None
            ))
            
        return {
            "summary_text": summary,
            "key_themes": themes,
            "cases": cases
        }
