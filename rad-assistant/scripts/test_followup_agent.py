#!/usr/bin/env python
"""
Manual test script for Agent 3: Follow-Up & Incidental Findings Tracker.
"""

import sys
import os
import json
import logging

# Ensure src is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from radiology_assistant.models import FollowUpExtractionRequest, ExamMetadata
from radiology_assistant.api import get_followup_agent

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_followup_agent.py <path_to_request.json>")
        print("Creating a dummy request for demo...")
        
        request = FollowUpExtractionRequest(
            exam_metadata=ExamMetadata(
                modality="CT",
                body_region="Chest",
                exam_date="2023-11-01",
                age=65,
                sex="M"
            ),
            report_text="""
            FINDINGS: 
            Lungs: 6 mm solid nodule in the right lower lobe.
            Liver: 2 cm incidental cyst.
            
            IMPRESSION:
            1. RLL pulmonary nodule. Recommend CT chest follow-up in 6 months.
            2. Liver cyst, likely benign. No specific follow-up required.
            """
        )
    else:
        with open(sys.argv[1], 'r') as f:
            data = json.load(f)
            request = FollowUpExtractionRequest.model_validate(data)

    print(f"--- Sending Request ---\n{request.model_dump_json(indent=2)}\n")

    try:
        agent = get_followup_agent()
        response = agent.extract_followups(request)
        print("\n--- Agent Response ---")
        print(response.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
