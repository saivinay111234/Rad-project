import cv2
import numpy as np
import base64
from PIL import Image
import io

def make_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    Create a heatmap overlay on the original image.
    1. Normalize heatmap to 0-255.
    2. Apply colormap (JET).
    3. Blend with grayscale image.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        # Normalize image to 0-255 if it's float
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            # If it's already > 1 but float, just cast? Or normalize?
            # Let's assume it's in a reasonable range or normalize it.
            # Safe bet: min-max norm
            im_min, im_max = image.min(), image.max()
            if im_max > im_min:
                image = ((image - im_min) / (im_max - im_min) * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

    # Resize heatmap to match image size
    h, w = image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    # Normalize heatmap to 0-255
    hm_min, hm_max = heatmap_resized.min(), heatmap_resized.max()
    if hm_max > hm_min:
        heatmap_norm = ((heatmap_resized - hm_min) / (hm_max - hm_min) * 255).astype(np.uint8)
    else:
        heatmap_norm = np.zeros_like(heatmap_resized, dtype=np.uint8)

    # Apply colormap
    heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
    
    # Convert grayscale image to BGR for blending
    if len(image.shape) == 2:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_bgr = image

    # Blend
    overlay = cv2.addWeighted(image_bgr, 0.7, heatmap_color, 0.3, 0)
    
    return overlay

def encode_png_base64(img: np.ndarray) -> str:
    """
    Convert image (numpy array) to base64 encoded PNG string.
    """
    # Convert BGR to RGB for PIL if needed (OpenCV uses BGR)
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
        
    pil_img = Image.fromarray(img_rgb)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
