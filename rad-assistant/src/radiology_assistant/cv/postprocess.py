from typing import List, Dict, Any
from ..models import CVRegionHighlight

def regions_to_models(regions_raw: List[Dict[str, Any]]) -> List[CVRegionHighlight]:
    """
    Convert raw model output regions to CVRegionHighlight objects.
    Validates bbox shape and score range.
    """
    highlights = []
    for r in regions_raw:
        score = float(r.get("score", 0.0))
        # Clip score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        bbox = r.get("bbox")
        if bbox and len(bbox) == 4:
            # Ensure bbox is int
            bbox = tuple(map(int, bbox))
        else:
            bbox = None
            
        highlights.append(CVRegionHighlight(
            label=str(r.get("label", "unknown")),
            score=score,
            bbox=bbox,
            mask_present=r.get("mask_present", False)
        ))
    return highlights
