"""Pluggable OCR engine interface supporting EasyOCR and optional Tesseract."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaseOCREngine(ABC):
    """Abstract base class for OCR engines."""

    @abstractmethod
    def extract_text(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Extract text from an image.

        Returns:
            full_text: concatenated string of all recognized lines
            avg_confidence: average confidence score between 0.0 and 1.0
            details: list of dicts with keys "text", "confidence", "bbox"
        """
        pass


class EasyOCREngine(BaseOCREngine):
    """EasyOCR implementation with lazy loading or external reader injection."""

    def __init__(
        self,
        reader: Optional[Any] = None,
        languages: Optional[List[str]] = None,
        gpu: bool = False,
    ):
        self.languages = languages or ["en"]
        self.gpu = gpu
        self._reader = reader

    @property
    def reader(self) -> Any:
        if self._reader is None:
            try:
                import os
                import easyocr

                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_dir = os.path.join(base_dir, ".easyocr", "model")
                user_network_dir = os.path.join(base_dir, ".easyocr", "user_network")
                os.makedirs(model_dir, exist_ok=True)
                os.makedirs(user_network_dir, exist_ok=True)
                kwargs = {
                    "gpu": self.gpu,
                    "model_storage_directory": model_dir,
                    "user_network_directory": user_network_dir,
                }

                logger.info("Initializing EasyOCR reader for %s (gpu=%s, model_dir=%s)...", self.languages, self.gpu, model_dir)
                self._reader = easyocr.Reader(self.languages, **kwargs)
            except Exception as e:
                logger.error("Failed to initialize EasyOCR reader: %s", e)
                raise
        return self._reader

    def extract_text(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        if image is None or image.size == 0:
            return "", 0.0, []

        try:
            results = self.reader.readtext(image)
        except Exception as e:
            logger.error("EasyOCR text extraction failed: %s", e)
            return "", 0.0, []

        if not results:
            return "", 0.0, []

        lines: List[str] = []
        confidences: List[float] = []
        details: List[Dict[str, Any]] = []

        for bbox, text, conf in results:
            text_str = str(text).strip()
            conf_val = float(conf)
            if text_str:
                lines.append(text_str)
                confidences.append(conf_val)
                # bbox is a 4-point polygon [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                details.append(
                    {
                        "text": text_str,
                        "confidence": conf_val,
                        "bbox": [list(pt) for pt in bbox],
                    }
                )

        full_text = "\n".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, avg_confidence, details


class TesseractOCREngine(BaseOCREngine):
    """Optional Tesseract OCR engine via pytesseract."""

    def __init__(self) -> None:
        self.available = False
        try:
            import pytesseract

            self._pytesseract = pytesseract
            self.available = True
        except ImportError:
            self._pytesseract = None
            logger.info("pytesseract is not installed; Tesseract engine disabled.")

    def extract_text(
        self, image: np.ndarray
    ) -> Tuple[str, float, List[Dict[str, Any]]]:
        if not self.available or self._pytesseract is None:
            logger.warning(
                "pytesseract is not installed or available on this system; returning empty result."
            )
            return "", 0.0, []

        if image is None or image.size == 0:
            return "", 0.0, []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        try:
            data = self._pytesseract.image_to_data(
                gray, output_type=self._pytesseract.Output.DICT
            )
        except Exception as e:
            logger.error("Tesseract execution failed: %s", e)
            return "", 0.0, []

        lines: List[str] = []
        confidences: List[float] = []
        details: List[Dict[str, Any]] = []

        n_boxes = len(data["text"])
        for i in range(n_boxes):
            text_val = data["text"][i].strip()
            conf_val = float(data["conf"][i])
            if text_val and conf_val > 0:
                normalized_conf = conf_val / 100.0
                lines.append(text_val)
                confidences.append(normalized_conf)
                details.append(
                    {
                        "text": text_val,
                        "confidence": normalized_conf,
                        "bbox": [
                            data["left"][i],
                            data["top"][i],
                            data["width"][i],
                            data["height"][i],
                        ],
                    }
                )

        full_text = " ".join(lines)
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return full_text, avg_confidence, details


def get_ocr_engine(engine_name: str = "easyocr", reader: Optional[Any] = None) -> BaseOCREngine:
    """Factory function for OCR engines. Automatically falls back to EasyOCR if Tesseract is not installed."""
    engine_name_lower = engine_name.lower().strip()
    if engine_name_lower == "easyocr":
        return EasyOCREngine(reader=reader)
    if engine_name_lower == "tesseract":
        tess = TesseractOCREngine()
        if not tess.available:
            logger.warning("Tesseract is not installed on this system; gracefully falling back to EasyOCR.")
            return EasyOCREngine(reader=reader)
        return tess
    raise ValueError(f"Unknown OCR engine: {engine_name}. Supported: 'easyocr', 'tesseract'")


def find_best_orientation(
    image: np.ndarray, engine: BaseOCREngine
) -> Tuple[np.ndarray, int, float]:
    """Test rotations (0, 90, 180, 270) to find the orientation with highest OCR confidence."""
    if image is None or image.size == 0:
        return image, 0, 0.0

    rotations: Dict[int, np.ndarray] = {
        0: image,
        90: cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        180: cv2.rotate(image, cv2.ROTATE_180),
        270: cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
    }

    best_image = image
    best_angle = 0
    best_confidence = -1.0

    for angle, rotated in rotations.items():
        try:
            _, conf, details = engine.extract_text(rotated)
            # Give weight to both confidence and number of detected text boxes
            score = conf * (1.0 if len(details) > 3 else 0.5)
            if score > best_confidence:
                best_confidence = score
                best_angle = angle
                best_image = rotated
        except Exception as e:
            logger.debug("Failed evaluating orientation %d: %s", angle, e)
            continue

    logger.info("Best orientation chosen: %d° with confidence score %.2f", best_angle, best_confidence)
    return best_image, best_angle, max(best_confidence, 0.0)

