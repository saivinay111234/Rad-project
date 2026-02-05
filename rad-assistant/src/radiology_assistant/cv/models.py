from typing import Protocol, Dict, Any, Optional
import torch
import numpy as np

import torchxrayvision as xrv

class AnomalyModel(Protocol):
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        ...

class ChestXrayAnomalyModel:
    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        # Load TorchXRayVision DenseNet
        # weights="densenet121-res224-all" is a good general purpose model
        self.model = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.model.to(device)
        self.model.eval()

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """
        Returns structure like:
        {
          "heatmap": np.ndarray (H, W),
          "regions": [
             {"label": "opacity", "score": 0.7, "bbox": [x, y, w, h]}
          ]
        }
        """
        x = x.to(self.device)
        x.requires_grad = True
        
        # Forward pass
        outputs = self.model(x)
        
        # Get top prediction
        # outputs is [1, N_pathologies]
        probs = torch.sigmoid(outputs)[0] # Sigmoid for multi-label
        top_score, top_idx = torch.max(probs, dim=0)
        top_label = self.model.pathologies[top_idx]
        
        # Generate Saliency Map (Input Gradient)
        # We want to see what part of the image contributed to this top score
        # Simple gradient: d(score)/d(image)
        self.model.zero_grad()
        outputs[0, top_idx].backward()
        
        # Gradients are [1, 1, H, W]
        grads = x.grad.data[0, 0].cpu().numpy()
        
        # Process gradients into a heatmap
        # Take absolute value and smooth/normalize
        heatmap = np.abs(grads)
        
        # Normalize heatmap to 0-1 for visualization
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
            
        # Create region result
        # For now, we don't have bounding boxes from DenseNet (it's a classifier).
        # We can return the top label and score, and maybe a dummy bbox or just the heatmap.
        # The contract allows bbox to be None.
        
        regions = [
            {
                "label": top_label,
                "score": float(top_score.item()),
                "bbox": None, # DenseNet doesn't give boxes
                "mask_present": False
            }
        ]
        
        # If score is very low, maybe don't return any regions?
        # Let's keep it for now so we always see what it thinks.

        return {
            "heatmap": heatmap,
            "regions": regions
        }
