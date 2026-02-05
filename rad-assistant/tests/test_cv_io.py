import io
import pytest
import numpy as np
from PIL import Image
from radiology_assistant.cv.io import load_image_from_bytes, InvalidDICOMError, load_dicom_from_bytes

def test_load_image_from_bytes_png():
    # Create a dummy PNG
    img = Image.new('L', (100, 100), color=128)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    data = buf.getvalue()

    loaded = load_image_from_bytes(data)
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100)
    assert loaded.dtype == np.float32
    assert np.allclose(loaded, 128)

def test_load_image_from_bytes_invalid():
    with pytest.raises(ValueError):
        load_image_from_bytes(b"not an image")

def test_load_dicom_from_bytes_invalid():
    # Just test that it raises InvalidDICOMError on garbage
    with pytest.raises(InvalidDICOMError):
        load_dicom_from_bytes(b"not a dicom")
