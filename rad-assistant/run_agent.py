#!/usr/bin/env python
"""
Wrapper script to run the radiology assistant agent.
This handles the Python path automatically.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import and run
from radiology_assistant.run_report_agent import main

if __name__ == "__main__":
    # Pass any command-line arguments
    exit_code = main(*sys.argv[1:] if len(sys.argv) > 1 else [None])
    sys.exit(exit_code)
