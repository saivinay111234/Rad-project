# Radiology Assistant - Agent 1: Report Drafting

An AI-powered system for generating professional radiology reports from structured clinical findings and patient context using Google's Gemini LLM.

## Features

✅ **Structured Input/Output**: Pydantic models ensure consistent data contracts  
✅ **LLM Integration**: Clean wrapper around Gemini API for easy provider switching  
✅ **Error Handling**: Robust retry logic and fallback report generation  
✅ **Professional Reports**: Generates reports with TECHNIQUE, FINDINGS, and IMPRESSION sections  
✅ **Fully Testable**: Unit tests and CLI for easy validation  
✅ **Configurable**: Environment-based settings for flexible deployment  

## Project Structure

```
rad-assistant/
├── .venv/                          # Python virtual environment
├── src/
│   └── radiology_assistant/
│       ├── __init__.py
│       ├── config.py               # Configuration management
│       ├── models.py               # Pydantic data models
│       ├── llm_client.py           # Gemini API wrapper
│       ├── run_report_agent.py     # CLI entry point
│       └── agents/
│           ├── __init__.py
│           └── report_drafter.py   # Main agent logic
├── tests/
│   ├── __init__.py
│   └── test_report_drafter.py      # Unit tests
├── examples/
│   ├── request.json                # Sample CXR request
│   ├── request_ct.json             # Sample CT request
│   └── request_cardiac.json        # Sample cardiac request
├── .env                            # Environment variables
├── .env.example                    # Example env file (no secrets)
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── pyproject.toml                  # Project configuration
└── README.md                       # This file
```

## Setup

### Prerequisites
- Python 3.8+
- Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

### Installation

1. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   ```bash
   # Copy and edit .env file
   cp .env .env.local  # or just edit .env directly
   # Add your Gemini API key to .env
   ```

4. **Verify installation:**
   ```bash
   python -c "from radiology_assistant.models import ReportDraftRequest; print('✓ Imports working')"
   ```

## Quick Start

### Running the CLI

```bash
# Run with default example request
python -m radiology_assistant.run_report_agent

# Run with specific request file
python -m radiology_assistant.run_report_agent examples/request_ct.json

# Or use --request flag
python -m radiology_assistant.run_report_agent --request examples/request_cardiac.json
```

### Using Programmatically

```python
from radiology_assistant.config import Config
from radiology_assistant.llm_client import LLMClient
from radiology_assistant.models import (
    Finding, ClinicalContext, ReportDraftRequest
)
from radiology_assistant.agents import ReportDraftingAgent

# Initialize components
Config.validate()
llm_client = LLMClient()
agent = ReportDraftingAgent(llm_client)

# Create request
findings = [
    Finding(
        location="right lower lobe",
        type="opacity",
        severity="moderate"
    )
]
context = ClinicalContext(
    patient_info="65-year-old male",
    clinical_presentation="Fever and cough for 3 days",
    relevant_history="History of COPD"
)
request = ReportDraftRequest(
    findings=findings,
    clinical_context=context,
    modality="Chest X-ray",
    view="PA & lateral"
)

# Generate report
report = agent.draft_report(request)

# Use the report
print(report.technique)
print(report.findings)
print(report.impression)
```


## API Contract

### Request Format (ReportDraftRequest)

Standard input for the report generation agent:

```json
{
  "modality": "Chest X-ray",
  "view": "PA and lateral",
  "clinical_context": {
    "patient_info": "65-year-old male",
    "clinical_presentation": "Fever and cough for 3 days",
    "relevant_history": "History of COPD, former smoker"
  },
  "findings": [
    {
      "location": "right lower lobe",
      "type": "opacity",
      "severity": "moderate",
      "additional_details": "consolidation with air bronchograms"
    },
    {
      "location": "cardiac silhouette",
      "type": "cardiomegaly",
      "severity": "mild"
    }
  ]
}
```

### Response Format (ReportDraft)

Standard output from the report generation agent:

```json
{
  "technique": "PA and lateral views of the chest were obtained.",
  "findings": "There is a moderate right lower lobe opacity consistent with consolidation. The cardiac silhouette is mildly enlarged.",
  "impression": "Moderate right lower lobe consolidation consistent with pneumonia. Mild cardiomegaly. Clinical correlation recommended.",
  "confidence_score": 0.85
}
```

**Field Descriptions:**
- `technique` (string): Standard technical description of the imaging procedure performed
- `findings` (string): Detailed paragraph describing all radiological findings organized by anatomical region
- `impression` (string): Clinical summary and interpretation of findings with differential considerations
- `confidence_score` (float 0.0-1.0): Model's confidence in the accuracy of this report (0.0=low confidence, 1.0=high confidence)

## Data Models

### Finding
Represents a single radiological finding.
```python
Finding(
    location: str,                    # Anatomical location
    type: str,                        # Finding type
    severity: str,                    # mild/moderate/severe
    additional_details: Optional[str] # Extra context
)
```

### ClinicalContext
Clinical information about the patient.
```python
ClinicalContext(
    patient_info: str,                      # Demographics
    clinical_presentation: str,             # Chief complaint
    relevant_history: Optional[str]        # Medical history
)
```

### ReportDraftRequest
Input to the agent.
```python
ReportDraftRequest(
    findings: List[Finding],
    clinical_context: ClinicalContext,
    modality: Optional[str],  # e.g., "Chest X-ray"
    view: Optional[str]       # e.g., "PA & lateral"
)
```

### ReportDraft
Output from the agent.
```python
ReportDraft(
    technique: str,              # Technical description
    findings: str,               # Detailed findings
    impression: str,             # Clinical impression
    confidence_score: Optional[float]  # Quality score (0-1)
)
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=radiology_assistant --cov-report=html

# Run specific test file
python -m pytest tests/test_report_drafter.py -v
```

## Configuration

Edit `.env` file to customize:

```env
# LLM Settings
GEMINI_API_KEY=your-key
LLM_TEMPERATURE=0.3          # 0=deterministic, 1=creative
LLM_MAX_TOKENS=1500          # Max response length
LLM_PROVIDER=gemini

# Application
DEBUG=False
LOG_LEVEL=INFO

# Retry behavior
MAX_RETRIES=3
RETRY_DELAY=1.0              # Seconds between retries
```

## Development Roadmap

### ✅ Completed (Phase 0-5)
- Project structure and setup
- Data models (Finding, ClinicalContext, ReportDraft)
- LLM client with Gemini integration
- Report Drafting Agent with error handling
- Unit tests and CLI tool
- Example requests and configuration

### 🔄 Next Steps (Phase 6+)
- **Phase 6**: Vision model for automatic finding extraction from images
- **Phase 7**: Web API (FastAPI) for programmatic access
- **Phase 8**: Database integration for report history
- **Phase 9**: Fine-tuning LLM for radiology-specific language
- **Phase 10**: UI dashboard for clinician interaction

## API Reference

### LLMClient

```python
llm_client = LLMClient(api_key=None, temperature=0.3)

# Generate text
response = llm_client.generate(
    prompt="Your prompt",
    temperature=0.3,
    max_tokens=1500,
    system_prompt="Optional system instructions"
)

# Generate JSON
json_response = llm_client.generate_json(prompt="...")
```

### ReportDraftingAgent

```python
agent = ReportDraftingAgent(llm_client)

# Draft a report
report = agent.draft_report(request: ReportDraftRequest) -> ReportDraft
```

## Error Handling

The agent includes robust error handling:

- **JSON Parsing Errors**: Attempts to extract JSON from response
- **API Errors**: Automatic retry with exponential backoff
- **Invalid Input**: Fallback report with low confidence score
- **Configuration Errors**: Clear error messages on startup

## Logging

Enable debug logging by setting `LOG_LEVEL=DEBUG` in `.env`:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

### "GEMINI_API_KEY environment variable is not set"
- Check that `.env` file exists in the project root
- Verify `GEMINI_API_KEY` is set in `.env`
- Reload the terminal/IDE after changing `.env`

### "Import could not be resolved"
- Make sure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check that you're running from the project root

### LLM returns invalid JSON
- The agent will fall back to a basic report
- Check logs for details: `LOG_LEVEL=DEBUG`
- Try increasing `LLM_TEMPERATURE=0.2` for more deterministic output

## Performance Notes

- **Temperature**: Lower values (0.1-0.3) produce more consistent medical reports
- **Max Tokens**: 1500 is usually sufficient for a complete report
- **Retries**: The agent automatically retries failed requests with exponential backoff

## Security

- **Never commit `.env` with real API keys** - use `.env.example` as template
- **Use environment variables** in production (not .env files)
- **Validate all user input** before passing to the agent

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review example requests in `examples/`
3. Check application logs with `LOG_LEVEL=DEBUG`
4. Open an issue with detailed error messages

---

**Current Version**: 0.1.0  
**Last Updated**: 2025-11-15  
**Status**: Active Development - Phase 0-5 Complete
