"""Barcode detection and decoding using pyzbar, OpenCV, and OCR fallback."""

import logging
import re
from typing import List, Optional, Set

import cv2
import numpy as np

from modules.barcode_validator import BarcodeValidator
from modules.image_processor import ImageProcessor
from modules.models import BarcodeResult
from utils.validators import validate_barcode, validate_ean8, validate_ean13

logger = logging.getLogger(__name__)


class BarcodeScanner:
    """Multi-stage barcode detector supporting pyzbar, OpenCV, and OCR fallback."""

    @staticmethod
    def scan_pyzbar(image: np.ndarray) -> List[BarcodeResult]:
        """Decode barcodes using pyzbar library."""
        try:
            from pyzbar import pyzbar as pyzbar_lib
        except (ImportError, OSError) as e:
            logger.warning("pyzbar is not available: %s", e)
            return []

        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        try:
            decoded = pyzbar_lib.decode(gray)
        except Exception as e:
            logger.error("pyzbar decoding error: %s", e)
            return []

        results: List[BarcodeResult] = []
        type_map = {
            "EAN13": "EAN-13",
            "EAN8": "EAN-8",
            "UPCA": "UPC-A",
            "UPCE": "UPC-E",
            "CODE128": "CODE-128",
            "CODE39": "CODE-39",
            "ITF": "ITF",
        }

        for d in decoded:
            data = d.data.decode("utf-8", errors="replace").strip()
            if not data:
                continue
            btype = type_map.get(d.type, str(d.type))
            rect_val = (
                int(d.rect.left),
                int(d.rect.top),
                int(d.rect.width),
                int(d.rect.height),
            )
            r = BarcodeResult(
                data=data,
                type=btype,
                source="pyzbar",
                rect=rect_val,
                confidence=1.0,
            )
            r = BarcodeValidator.validate(r)
            results.append(r)

        return results

    @staticmethod
    def scan_cv2(image: np.ndarray) -> List[BarcodeResult]:
        """Decode barcodes using OpenCV BarcodeDetector fallback."""
        try:
            detector = cv2.barcode.BarcodeDetector()
        except (AttributeError, Exception) as e:
            logger.debug("cv2.barcode.BarcodeDetector not available: %s", e)
            return []

        if image is None or image.size == 0:
            return []

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        try:
            retval, decoded_info, decoded_type, points = detector.detectAndDecode(gray)
        except Exception as e:
            logger.debug("cv2 barcode detectAndDecode error: %s", e)
            return []

        results: List[BarcodeResult] = []
        if retval and decoded_info is not None:
            for info, dtype in zip(decoded_info, decoded_type):
                data = str(info).strip()
                if data:
                    r = BarcodeResult(
                        data=data,
                        type=str(dtype) if dtype else BarcodeValidator.detect_type(data),
                        source="cv2",
                        confidence=0.9,
                    )
                    r = BarcodeValidator.validate(r)
                    results.append(r)

        return results

    @staticmethod
    def scan_with_preprocessing(image: np.ndarray) -> List[BarcodeResult]:
        """Run pyzbar and OpenCV over all preprocessing variants until barcodes are found."""
        variants = ImageProcessor.get_preprocessing_variants(image)
        all_results: List[BarcodeResult] = []
        seen: Set[str] = set()

        # Phase 1: Try pyzbar on each variant
        for name, variant in variants:
            res = BarcodeScanner.scan_pyzbar(variant)
            for r in res:
                if r.data not in seen:
                    seen.add(r.data)
                    all_results.append(r)
            if all_results:
                logger.debug("Barcode detected with pyzbar on variant '%s'", name)
                return all_results

        # Phase 2: Try cv2 on each variant
        for name, variant in variants:
            res = BarcodeScanner.scan_cv2(variant)
            for r in res:
                if r.data not in seen:
                    seen.add(r.data)
                    all_results.append(r)
            if all_results:
                logger.debug("Barcode detected with cv2 on variant '%s'", name)
                return all_results

        return all_results

    @staticmethod
    def ocr_barcode_digits(image: np.ndarray, ocr_engine: Optional[object]) -> Optional[str]:
        """OCR the printed digits beneath the barcode lines and validate check digit."""
        if ocr_engine is None or image is None or image.size == 0:
            return None

        h, w = image.shape[:2]
        # Crop the bottom 35% where barcode digits are typically printed
        bottom_region = image[int(h * 0.65) :, :]

        try:
            extract_fn = getattr(ocr_engine, "extract_text", None)
            if not extract_fn:
                return None
            text, _, _ = extract_fn(bottom_region)
        except Exception as e:
            logger.debug("OCR barcode region extraction error: %s", e)
            return None

        # Clean spaces and find potential 13-digit EAN sequences
        # Often printed with spacing like "8 909329 191647"
        digits_clean = re.sub(r"\D", "", text)

        # Look for 13-digit candidate
        for i in range(len(digits_clean) - 12):
            candidate = digits_clean[i : i + 13]
            is_valid, _ = validate_ean13(candidate)
            if is_valid:
                logger.info("Found valid 13-digit EAN from OCR digits: %s", candidate)
                return candidate

        # Look for 8-digit candidate
        for i in range(len(digits_clean) - 7):
            candidate = digits_clean[i : i + 8]
            is_valid, _ = validate_ean8(candidate)
            if is_valid:
                logger.info("Found valid 8-digit EAN from OCR digits: %s", candidate)
                return candidate

        return None

    @classmethod
    def scan(cls, image: np.ndarray, ocr_engine: Optional[object] = None) -> List[BarcodeResult]:
        """Full pipeline: decode bars across variants, cross-check with OCR digits."""
        results = cls.scan_with_preprocessing(image)

        # Only fall back to OCR digits if no barcodes were decoded
        if not results and ocr_engine is not None:
            ocr_digits = cls.ocr_barcode_digits(image, ocr_engine)
            if ocr_digits:
                valid, msg = validate_barcode(ocr_digits)
                if valid:
                    results.append(
                        BarcodeResult(
                            data=ocr_digits,
                            type=BarcodeValidator.detect_type(ocr_digits),
                            source="OCR digits",
                            confidence=0.85,
                            is_valid=valid,
                            validation_message=msg,
                        )
                    )

        return results

