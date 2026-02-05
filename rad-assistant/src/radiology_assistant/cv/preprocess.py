from typing import Tuple
import numpy as np
import torch

def preprocess_for_model(img: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> torch.Tensor:
    """
    Preprocess image for TorchXRayVision model:
    1. Resize to target_size (224x224).
    2. Normalize to [-1024, 1024] range (XRV expectation).
    3. Convert to torch.Tensor with shape [1, 1, H, W].
    """
    from PIL import Image
    
    # Resize
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
    resized_img = np.array(pil_img).astype(np.float32)

    # XRV Normalization:
    # XRV expects values roughly in range -1024 to 1024 (Hounsfield-like).
    # Our input is likely 0-255 (if from PNG) or arbitrary float (if DICOM).
    # We'll map [min, max] -> [-1024, 1024] to be safe and consistent.
    
    min_val = resized_img.min()
    max_val = resized_img.max()
    
    if max_val > min_val:
        # Normalize to 0-1 first
        img_01 = (resized_img - min_val) / (max_val - min_val)
        # Map to -1024, 1024
        img_xrv = img_01 * 2048 - 1024
    else:
        img_xrv = np.zeros_like(resized_img)

    # Convert to tensor [1, 1, H, W]
    tensor = torch.from_numpy(img_xrv).unsqueeze(0).unsqueeze(0)
    
    return tensor
