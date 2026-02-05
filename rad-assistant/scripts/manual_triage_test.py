"""
Manual test script for Agent 6: Worklist Triage.

Usage:
    python -m scripts.manual_triage_test
"""

import sys
import os
import json
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from radiology_assistant.models import (
    WorklistTriageRequest,
    WorklistItem,
    TriageConfig,
    TriageThresholdConfig,
    ModalityGroup
)
from radiology_assistant.agents.worklist_triage import WorklistTriageAgent
from radiology_assistant.config import Config

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def main():
    print("Initializing WorklistTriageAgent...")
    
    # Use default config
    config = TriageConfig(
        thresholds=[
            TriageThresholdConfig(
                modality_group=ModalityGroup.XR,
                body_region="Chest",
                critical_threshold=0.9,
                high_threshold=0.7,
                low_threshold=0.3
            )
        ],
        model_mapping={"XR/Chest": "mock"},
        enable_llm_explanation=False # Disable LLM for manual test to avoid API costs/keys
    )
    
    agent = WorklistTriageAgent(config=config, llm_client=None)
    
    print("Creating dummy worklist items...")
    items = [
        # Item 1: STAT priority (should be HIGH or CRITICAL score boosted)
        WorklistItem(
            study_id="STUDY_001",
            modality="XR",
            body_region="Chest",
            clinical_indication="Shortness of breath, rule out pneumothorax",
            priority_flag_from_order="STAT"
        ),
        # Item 2: Routine, no image (should be Low/Routine)
        WorklistItem(
            study_id="STUDY_002",
            modality="XR",
            body_region="Chest",
            clinical_indication="Cough",
            priority_flag_from_order="Routine"
        ),
        # Item 3: Missing config modality (should be UNTRIAGED)
        WorklistItem(
            study_id="STUDY_003",
            modality="CT", # No config for CT
            body_region="Head",
            priority_flag_from_order="Routine"
        ),
        # Item 4: With dummy image path (will fail CV load but should fail-soft)
        WorklistItem(
            study_id="STUDY_004",
            modality="XR",
            body_region="Chest",
            image_reference={"thumbnail_path": "non_existent_file.png"}
        )
    ]
    
    request = WorklistTriageRequest(worklist_items=items)
    
    print("Running triage...")
    response = agent.triage(request)
    
    print("\n" + "="*60)
    print("TRIAGE RESULTS")
    print("="*60)
    
    for item in response.items:
        print(f"\nStudy ID: {item.study_id}")
        print(f"Label:    {item.triage_label.value}")
        print(f"Score:    {item.triage_score:.2f}")
        if item.reasons:
            print("Reasons:")
            for r in item.reasons:
                print(f"  - [{r.type.value}] {r.description} (w={r.weight})")
        if item.error:
            print(f"Error:    {item.error}")
            
    print("\n" + "="*60)
    print("Done.")

if __name__ == "__main__":
    main()
