import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    print("Attempting to import radiology_assistant.models...")
    import radiology_assistant.models
    print("Successfully imported radiology_assistant.models")

    print("Attempting to import radiology_assistant.agents.patient_report_explainer...")
    import radiology_assistant.agents.patient_report_explainer
    print("Successfully imported radiology_assistant.agents.patient_report_explainer")

except Exception:
    traceback.print_exc()
