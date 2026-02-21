import io
from typing import BinaryIO, Dict, Any, Tuple, Optional
import numpy as np
import pydicom
from PIL import Image

class InvalidDICOMError(Exception):
    """Raised when DICOM loading fails."""
    pass

def apply_windowing(image: np.ndarray, center: float, width: float) -> np.ndarray:
    """
    Apply DICOM windowing (perceived brightness/contrast adjustment).
    Maps pixels to [0, 255] range based on WindowCenter/WindowWidth.
    """
    min_val = center - (width / 2.0)
    max_val = center + (width / 2.0)
    
    # Clip and normalize
    windowed = np.clip(image, min_val, max_val)
    windowed = (windowed - min_val) / width
    return (windowed * 255.0).astype(np.uint8)

def extract_dicom_metadata(ds: pydicom.dataset.FileDataset) -> Dict[str, Any]:
    """Extract clinical metadata from a DICOM dataset."""
    return {
        "accession": getattr(ds, 'AccessionNumber', None),
        "modality": getattr(ds, 'Modality', None),
        "body_part": getattr(ds, 'BodyPartExamined', None),
        "patient_sex": getattr(ds, 'PatientSex', None),
        "patient_age": getattr(ds, 'PatientAge', None),
        "exam_date": getattr(ds, 'ContentDate', None),
        "window_center": getattr(ds, 'WindowCenter', None),
        "window_width": getattr(ds, 'WindowWidth', None),
        "photometric_interpretation": getattr(ds, 'PhotometricInterpretation', "MONOCHROME2")
    }

def load_dicom_from_bytes(data: bytes, apply_dicom_windowing: bool = True) -> np.ndarray:
    """
    Load DICOM from bytes and return a 2D float32 numpy array.
    Automatically handles rescale slope/intercept and optional windowing.
    """
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        pixel_array = ds.pixel_array.astype(np.float32)

        # 1. Rescale Slope/Intercept
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            pixel_array = pixel_array * slope + intercept
        
        # 2. Apply Windowing if tags exist
        if apply_dicom_windowing:
            wc = getattr(ds, 'WindowCenter', None)
            ww = getattr(ds, 'WindowWidth', None)
            
            # Handle list-type window tags
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]

            if wc is not None and ww is not None:
                pixel_array = apply_windowing(pixel_array, float(wc), float(ww)).astype(np.float32)
            
        return pixel_array
    except Exception as e:
        raise InvalidDICOMError(f"Failed to load DICOM: {e}") from e

def load_dicom_with_metadata(data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load DICOM and return (pixel_array, metadata_dict)."""
    try:
        ds = pydicom.dcmread(io.BytesIO(data))
        metadata = extract_dicom_metadata(ds)
        image = load_dicom_from_bytes(data)
        return image, metadata
    except Exception as e:
        raise InvalidDICOMError(f"Failed to load DICOM with metadata: {e}") from e

def load_image_from_bytes(data: bytes) -> np.ndarray:
    """
    Load generic image (PNG/JPEG) from bytes and return a 2D grayscale float32 numpy array.
    """
    try:
        image = Image.open(io.BytesIO(data)).convert('L') # Convert to grayscale
        return np.array(image).astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}") from e
