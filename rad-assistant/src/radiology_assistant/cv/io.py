import io
from typing import BinaryIO
import numpy as np
import pydicom
from PIL import Image

class InvalidDICOMError(Exception):
    """Raised when DICOM loading fails."""
    pass

def load_dicom_from_bytes(data: bytes) -> np.ndarray:
    """
    Load DICOM from bytes and return a 2D float32 numpy array.
    Handles rescale slope/intercept if present.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        pixel_array = ds.pixel_array.astype(np.float32)

        # Handle rescale slope/intercept
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
            
        return pixel_array
    except Exception as e:
        raise InvalidDICOMError(f"Failed to load DICOM: {e}") from e

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Load generic image (PNG/JPEG) from bytes and return a 2D grayscale float32 numpy array.
    """
    try:
        image = Image.open(io.BytesIO(data)).convert('L') # Convert to grayscale
        return np.array(image).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}") from e
