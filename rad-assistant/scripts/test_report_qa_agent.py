#!/usr/bin/env python
"""
Manual test script for Agent 4: Structured Reporting & QA Coach.
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

from radiology_assistant.models import ReportQARequest, ExamMetadata
from radiology_assistant.api import get_report_qa_agent

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_report_qa_agent.py <path_to_report.txt>")
        sys.exit(1)

    report_path = sys.argv[1]
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    print(f"--- Analyzing Report: {report_path} ---")
    
    # 1. Prepare Request
    request = ReportQARequest(
        exam_metadata=ExamMetadata(
            modality="XR",
            body_region="Chest",
            age=50,
            sex="F"
        ),
        report_text=report_text
    )
    
    # 2. Call Agent
    try:
        agent = get_report_qa_agent()
        print("Agent initialized.")
        response = agent.review_report(request)
        
        print("\n--- QA Results ---")
        print(f"Overall Quality: {response.summary.overall_quality}")
        print(f"Critical Issues: {response.summary.num_critical}")
        print(f"Major Issues: {response.summary.num_major}")
        
        if response.issues:
            print("\nIssues Found:")
            for issue in response.issues:
                print(f"[{issue.severity.upper()}] {issue.type}: {issue.description}")
                if issue.suggested_text:
                    print(f"   Suggestion: {issue.suggested_text}")

        # Dump full JSON for inspection
        # print("\nFull JSON:")
        # print(response.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
