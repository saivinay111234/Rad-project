#!/usr/bin/env python
"""
Manual test script for Agent 5: Patient-Friendly Report Explainer.
"""

import sys
import os
import io
import json
import logging

# Ensure src is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from radiology_assistant.models import (
    PatientReportSummaryRequest, 
    ExamMetadata, 
    PatientReadingLevel,
    PatientSummaryTone
)
from radiology_assistant.api import get_patient_explainer_agent

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_patient_explainer_agent.py <path_to_report.txt> [reading_level] [tone]")
        sys.exit(1)

    report_path = sys.argv[1]
    reading_level = sys.argv[2] if len(sys.argv) > 2 else "simple"
    tone = sys.argv[3] if len(sys.argv) > 3 else "neutral"

    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    print(f"--- Explaining Report: {report_path} ---")
    print(f"Level: {reading_level}, Tone: {tone}")
    
    # 1. Prepare Request
    request = PatientReportSummaryRequest(
        exam_metadata=ExamMetadata(
            modality="XR",
            body_region="Chest",
            age=45,
            sex="M"
        ),
        report_text=report_text,
        reading_level=PatientReadingLevel(reading_level),
        tone=PatientSummaryTone(tone),
        language_code="en",
        followup_data=None # We'll start without linked follow-up for this simple test
    )
    
    # 2. Call Agent
    try:
        agent = get_patient_explainer_agent()
        print("Agent initialized.")
        response = agent.explain(request)
        
        print("\n--- Patient Summary ---")
        print(response.patient_summary_text)
        
        print("\n--- Key Points ---")
        for kp in response.key_points:
            print(f"- {kp}")
            
        print("\n--- Next Steps ---")
        if not response.next_steps:
            print("(None)")
        else:
            for step in response.next_steps:
                print(f"- [{step.urgency.upper()}] {step.description}")
                if step.followup_interval:
                     print(f"  (suggested interval: {step.followup_interval})")

        print("\n--- Glossary ---")
        if not response.glossary:
             print("(None)")
        else:
            for item in response.glossary:
                print(f"- {item.term}: {item.explanation}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
