#!/usr/bin/env python
"""
Manual test script for Agent 2: Visual Highlighting Agent.
"""

import sys
import os
import io
import logging
from PIL import Image

# Ensure src is in path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
sys.path.insert(0, src_path)

from radiology_assistant.models import CVHighlightRequest
from radiology_assistant.api import get_cv_agent

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    print("--- Testing Visual Highlighting Agent ---")
    
    # 1. Create Dummy Image
    print("Generating dummy 512x512 image...")
    img = Image.new('L', (512, 512), color=128)
    # Add a "nodule" (fake)
    for x in range(200, 250):
        for y in range(200, 250):
            img.putpixel((x, y), 255)
            
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()
    
    # 2. Prepare Request
    request = CVHighlightRequest(
        modality="DX",
        body_part="chest",
        view="PA"
    )
    
    # 3. Call Agent
    try:
        agent = get_cv_agent()
        print("Agent initialized.")
        result = agent.highlight(request, image_bytes)
        
        print("\n--- Agent Response ---")
        print(f"Study ID: {result.study_id}")
        print(f"Summary: {result.summary}")
        print(f"Regions Found: {len(result.regions)}")
        if result.regions:
            print(f"Top Region: {result.regions[0].label} ({result.regions[0].score:.2f})")
        print(f"Heatmap Generated: {'Yes' if result.heatmap_png_base64 else 'No'}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Print full stack if needed
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
