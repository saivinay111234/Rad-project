import pytest
from unittest.mock import MagicMock
import numpy as np
from radiology_assistant.agents.visual_highlighter import VisualHighlightingAgent
from radiology_assistant.models import CVHighlightRequest, CVHighlightResult
from radiology_assistant.cv.models import ChestXrayAnomalyModel

@pytest.fixture
def mock_model():
    model = MagicMock(spec=ChestXrayAnomalyModel)
    # Mock predict return
    model.predict.return_value = {
        "heatmap": np.zeros((224, 224), dtype=np.float32),
        "regions": [
            {"label": "opacity", "score": 0.9, "bbox": [10, 10, 50, 50]}
        ]
    }
    return model

def test_highlight_success(mock_model):
    agent = VisualHighlightingAgent(model=mock_model)
    
    # Create dummy image bytes (PNG)
    import io
    from PIL import Image
    img = Image.new('L', (224, 224), color=100)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    request = CVHighlightRequest(modality="DX")
    result = agent.highlight(request, image_bytes)

    assert isinstance(result, CVHighlightResult)
    assert result.modality == "DX"
    assert len(result.regions) == 1
    assert result.regions[0].label == "opacity"
    assert result.regions[0].score == 0.9
    assert result.heatmap_png_base64 is not None
    assert len(result.heatmap_png_base64) > 0
    assert "opacity" in result.summary

def test_highlight_no_anomalies(mock_model):
    mock_model.predict.return_value = {
        "heatmap": None,
        "regions": []
    }
    agent = VisualHighlightingAgent(model=mock_model)
    
    # Create dummy image bytes
    import io
    from PIL import Image
    img = Image.new('L', (224, 224), color=100)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    image_bytes = buf.getvalue()

    request = CVHighlightRequest(modality="DX")
    result = agent.highlight(request, image_bytes)

    assert len(result.regions) == 0
    assert "No significant anomalies" in result.summary
