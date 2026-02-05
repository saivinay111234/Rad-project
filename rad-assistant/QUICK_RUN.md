# 🚀 HOW TO RUN THE AGENT

## Quick Start

The agent is ready to use! Here are the easiest ways to run it:

### Option 1: Using Python Wrapper (Recommended)
```powershell
cd "D:\Data Science\Projects\Rad project\rad-assistant"
python run_agent.py
```

To run the HTTP API server (FastAPI):
```powershell
python run_api.py
```

### Option 2: Using Batch File (Windows)
```cmd
cd "D:\Data Science\Projects\Rad project\rad-assistant"
run_agent.bat
```

### Option 3: Using PowerShell with PYTHONPATH
```powershell
cd "D:\Data Science\Projects\Rad project\rad-assistant"
$env:PYTHONPATH = "$(pwd)\src"
python -m radiology_assistant.run_report_agent
```

### Option 4: With Custom Request File
```powershell
python run_agent.py examples/request_ct.json
python run_agent.py examples/request_cardiac.json
```

---

## What Happens When You Run It

1. ✅ Loads example request (or your custom file)
2. ✅ Initializes Gemini LLM client
3. ✅ Creates report agent
4. ✅ Drafts professional radiology report
5. ✅ Displays report with:
   - TECHNIQUE section
   - FINDINGS section
   - IMPRESSION section
   - Confidence score
   - JSON output

---

## Example Output

```
================================================================================
RADIOLOGY REPORT
================================================================================

TECHNIQUE:
PA and lateral views of the chest were obtained.

FINDINGS:
The cardiac silhouette is mildly enlarged. There is a moderate right lower lobe 
opacity consistent with consolidation.

IMPRESSION:
Moderate right lower lobe consolidation consistent with pneumonia. Mild 
cardiomegaly noted. Clinical correlation recommended.

Confidence Score: 0.85

================================================================================
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'radiology_assistant'"
**Solution**: Use `python run_agent.py` instead of `-m` syntax. The wrapper handles paths automatically.

### "GEMINI_API_KEY not set"
**Solution**: Check `.env` file has your API key:
```env
GEMINI_API_KEY=your-actual-key-here
```

### "No request file found"
**Solution**: Use one of these:
- Run without arguments: `python run_agent.py` (uses default)
- Specify a file: `python run_agent.py examples/request.json`

### API returns 404 error
**Solution**: The API key might not have access to the model or the model name needs updating.
- Check your API key is valid
- Try different models: `gemini-pro`, `gemini-1.5-pro`, `gemini-1.5-flash`

---

## Available Example Requests

```powershell
python run_agent.py examples/request.json          # Chest X-ray (pneumonia)
python run_agent.py examples/request_ct.json       # CT scan (nodule)
python run_agent.py examples/request_cardiac.json  # Cardiac (CHF)
```

---

## Using Programmatically

```python
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from radiology_assistant.config import Config
from radiology_assistant.llm_client import LLMClient
from radiology_assistant.agents import ReportDraftingAgent
from radiology_assistant.models import Finding, ClinicalContext, ReportDraftRequest

# Initialize
Config.validate()
client = LLMClient()
agent = ReportDraftingAgent(client)

# Create request
request = ReportDraftRequest(
    findings=[
        Finding(location="RLL", type="opacity", severity="moderate")
    ],
    clinical_context=ClinicalContext(
        patient_info="65-year-old male",
        clinical_presentation="Fever and cough",
        relevant_history="COPD"
    ),
    modality="Chest X-ray",
    view="PA & lateral"
)

# Generate report
report = agent.draft_report(request)
print(report.impression)
```

---

## Next Steps

1. ✅ Run: `python run_agent.py`
2. ✅ Verify it works with example data
3. ✅ Check API key is working
4. ✅ Integrate into your application
5. ✅ Customize for your needs

---

**Ready to go!** 🎉

For more details, see:
- `README.md` - Full feature overview
- `SETUP_GUIDE.md` - Detailed setup
- `PHASES.md` - Technical architecture
