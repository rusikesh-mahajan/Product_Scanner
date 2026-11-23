"""Barcode validation with detailed reporting and correction suggestions."""

import logging
from typing import Optional

from modules.models import BarcodeResult
from utils.validators import (
    calculate_ean_check_digit,
    validate_barcode,
)

logger = logging.getLogger(__name__)

SUPPORTED_TYPES = ["EAN-13", "EAN-8", "UPC-A", "UPC-E", "CODE-128", "CODE-39"]


class BarcodeValidator:
    """Validates barcodes and offers correction hints."""

    @staticmethod
    def validate(barcode_result: BarcodeResult) -> BarcodeResult:
        """Validate *barcode_result* in-place and return it."""
        barcode = barcode_result.data.strip().replace(" ", "")
        barcode_result.data = barcode

        if not barcode:
            barcode_result.is_valid = False
            barcode_result.validation_message = "Empty barcode"
            return barcode_result

        is_valid, message = validate_barcode(barcode, barcode_result.type)
        barcode_result.is_valid = is_valid
        barcode_result.validation_message = message
        if not is_valid:
            logger.warning("Invalid barcode %s: %s", barcode, message)
        return barcode_result

    @staticmethod
    def detect_type(barcode: str) -> str:
        """Guess the barcode type from its length."""
        barcode = barcode.strip().replace(" ", "")
        if not barcode.isdigit():
            return "UNKNOWN"
        length_map = {13: "EAN-13", 8: "EAN-8", 12: "UPC-A"}
        if len(barcode) in (6, 7):
            return "UPC-E"
        return length_map.get(len(barcode), "UNKNOWN")

    @staticmethod
    def format_barcode(barcode: str) -> str:
        """Return a human-readable spaced barcode string."""
        barcode = barcode.strip().replace(" ", "")
        if len(barcode) == 13:
            return f"{barcode[0]} {barcode[1:7]} {barcode[7:]}"
        if len(barcode) == 8:
            return f"{barcode[:4]} {barcode[4:]}"
        if len(barcode) == 12:
            return f"{barcode[0]} {barcode[1:6]} {barcode[6:11]} {barcode[11]}"
        return barcode

    @staticmethod
    def suggest_correction(barcode: str) -> Optional[str]:
        """If check digit is wrong, return the corrected barcode string."""
        barcode = barcode.strip().replace(" ", "")
        if not barcode.isdigit():
            return None
        pairs = {13: 12, 8: 7, 12: 11}
        payload_len = pairs.get(len(barcode))
        if payload_len is None:
            return None
        correct = calculate_ean_check_digit(barcode[:payload_len])
        if str(correct) != barcode[-1]:
            return barcode[:payload_len] + str(correct)
        return None

