"""
CLI entrypoint for the Report Drafting Agent.

Usage:
    python -m radiology_assistant.run_report_agent [--request path/to/request.json]
"""

import json
import sys
import logging
from pathlib import Path
from typing import Optional

from radiology_assistant.config import Config
from radiology_assistant.llm_client import LLMClient
from radiology_assistant.models import ReportDraftRequest
from radiology_assistant.agents import ReportDraftingAgent

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_request_from_file(filepath: str) -> ReportDraftRequest:
    """Load a ReportDraftRequest from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ReportDraftRequest(**data)


def main(request_file: Optional[str] = None):
    """
    Main CLI entry point.
    
    Args:
        request_file: Path to JSON file containing ReportDraftRequest
    """
    try:
        # Load or use default request
        if request_file:
            logger.info(f"Loading request from {request_file}")
            request = load_request_from_file(request_file)
        else:
            # Use example request from examples/request.json
            # From src/radiology_assistant/run_report_agent.py: up 2 levels = project root
            example_path = Path(__file__).parent.parent.parent / "examples" / "request.json"
            if example_path.exists():
                logger.info(f"Loading default request from {example_path}")
                request = load_request_from_file(str(example_path))
            else:
                logger.error(f"No request file found at {example_path}")
                return 1
        
        # Initialize LLM client and agent
        logger.info("Initializing LLM client and agent...")
        try:
            Config.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            return 1
        
        llm_client = LLMClient()
        agent = ReportDraftingAgent(llm_client)
        
        # Draft the report
        logger.info("Drafting report...")
        report = agent.draft_report(request)
        
        # Output the report
        print("\n" + "="*80)
        print("RADIOLOGY REPORT")
        print("="*80)
        print(f"\n{report.report_text}")
        
        if report.key_findings:
            print("\nKey Findings:")
            for kf in report.key_findings:
                print(f"- {kf.label} ({kf.category}, {kf.severity})")

        if report.used_cv_signals:
            print("\nCV Signals Used:")
            for cv in report.used_cv_signals:
                print(f"- {cv.cv_label}: {cv.reasoning}")

        if report.confidence_score:
            print(f"\nConfidence Score: {report.confidence_score:.2f}")
        print("\n" + "="*80)
        
        # Also output as JSON for programmatic use
        print("\nJSON Output:")
        report_json = report.model_dump()
        print(json.dumps(report_json, indent=2))
        
        logger.info("Report generation completed successfully")
        return 0
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in request file: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    # Parse command line arguments
    request_file = None
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(__doc__)
            sys.exit(0)
        elif sys.argv[1] == "--request" and len(sys.argv) > 2:
            request_file = sys.argv[2]
        else:
            request_file = sys.argv[1]
    
    exit_code = main(request_file)
    sys.exit(exit_code)
