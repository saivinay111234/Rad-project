#!/usr/bin/env python
"""Run the FastAPI server for the Radiology Assistant.

This wrapper ensures the `src` directory is on `sys.path` so the
package imports work the same way as other project scripts.

Usage:
    python run_api.py
"""

import os
import sys
from pathlib import Path

project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    import uvicorn

    # Host/port can be configured via environment variables
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload_flag = os.getenv("API_RELOAD", "True").lower() in ("1", "true", "yes")

    # Run the app from the package import path
    uvicorn.run("radiology_assistant.api:app", host=host, port=port, reload=reload_flag)
