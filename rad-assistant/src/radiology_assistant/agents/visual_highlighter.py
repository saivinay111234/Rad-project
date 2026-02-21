import logging
from typing import Optional, FrozenSet
import numpy as np

from ..models import CVHighlightRequest, CVHighlightResult, CVRegionHighlight
from ..cv.io import load_dicom_from_bytes, load_image_from_bytes, InvalidDICOMError
from ..cv.preprocess import preprocess_for_model
from ..cv.models import ChestXrayAnomalyModel
from ..cv.postprocess import regions_to_models
from ..cv.visualize import make_heatmap_overlay, encode_png_base64

# Only chest X-ray modalities are supported by the TorchXRayVision DenseNet121 model.
# CT, MR, US, NM require different models and preprocessing pipelines.
SUPPORTED_CV_MODALITIES: FrozenSet[str] = frozenset({"CR", "DX", "XR"})

class VisualHighlightingAgent:
    def __init__(self, model: ChestXrayAnomalyModel, logger: Optional[logging.Logger] = None):
        self.model = model
        self.logger = logger or logging.getLogger(__name__)

    def highlight(self, request: CVHighlightRequest, image_bytes: bytes) -> CVHighlightResult:
        """
        Run the visual highlighting pipeline.

        Raises:
            ValueError: If the imaging modality is not supported by the CV model.
            InvalidDICOMError: If the image cannot be parsed as DICOM or a standard image format.
        """
        # --- Modality Guard ---
        # Only chest X-ray modalities work with TorchXRayVision DenseNet121.
        # Return a clean error instead of running garbage inference on unsupported modalities.
        modality_upper = (request.modality or "").upper()
        if modality_upper not in SUPPORTED_CV_MODALITIES:
            raise ValueError(
                f"Modality '{request.modality}' is not supported for CV analysis. "
                f"Supported modalities: {sorted(SUPPORTED_CV_MODALITIES)}. "
                "CT, MR, US, and NM require a different model — contact the integration team."
            )

        self.logger.info(f"Starting visual highlighting for modality={request.modality}")

        # 1. Load Image
        try:
            # Try DICOM first
            image = load_dicom_from_bytes(image_bytes)
            self.logger.info("Loaded DICOM image.")
        except InvalidDICOMError:
            # Fallback to generic image
            try:
                image = load_image_from_bytes(image_bytes)
                self.logger.info("Loaded generic image (PNG/JPEG).")
            except Exception as e:
                self.logger.error(f"Failed to load image: {e}")
                raise InvalidDICOMError("Could not load image as DICOM or supported image format.") from e

        # 2. Preprocess
        try:
            tensor = preprocess_for_model(image)
            self.logger.info(f"Preprocessed image to tensor shape: {tensor.shape}")
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

        # 3. Run Model
        try:
            output = self.model.predict(tensor)
            self.logger.info("Model prediction successful.")
        except Exception as e:
            self.logger.error(f"Model prediction failed: {e}")
            raise

        # 4. Postprocess Regions
        regions_raw = output.get("regions", [])
        regions = regions_to_models(regions_raw)
        self.logger.info(f"Found {len(regions)} regions.")

        # 5. Generate Visualization
        heatmap = output.get("heatmap")
        heatmap_b64 = None
        if heatmap is not None:
            try:
                overlay = make_heatmap_overlay(image, heatmap)
                heatmap_b64 = encode_png_base64(overlay)
                self.logger.info("Generated heatmap overlay.")
            except Exception as e:
                self.logger.error(f"Visualization failed: {e}")
                # Don't fail the whole request if visualization fails, just log it
                pass

        # 6. Generate Summary
        # Simple rule-based summary for now
        if regions:
            # Find highest scoring region
            best_region = max(regions, key=lambda r: r.score)
            summary = f"Increased attention in {best_region.label} area (score: {best_region.score:.2f})."
        else:
            summary = "No significant anomalies detected."

        # 7. Return Result
        return CVHighlightResult(
            study_id=request.study_id,
            modality=request.modality,
            summary=summary,
            regions=regions,
            heatmap_png_base64=heatmap_b64
        )
