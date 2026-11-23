"""Validation utilities for barcodes and URLs."""

from typing import Tuple
from urllib.parse import urlparse


def calculate_ean_check_digit(barcode_without_check: str) -> int:
    """Calculate the EAN / UPC check digit.

    Works for EAN-13 (pass 12 digits), EAN-8 (7 digits), UPC-A (11 digits).
    Uses the standard odd/even weighting (1 and 3, from left, 0-indexed).
    """
    digits = [int(d) for d in barcode_without_check]
    total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits))
    return (10 - (total % 10)) % 10


def validate_ean13(barcode: str) -> Tuple[bool, str]:
    """Validate an EAN-13 barcode (13 digits with valid check digit)."""
    barcode = barcode.strip().replace(" ", "")
    if not barcode.isdigit():
        return False, "EAN-13 must contain only digits"
    if len(barcode) != 13:
        return False, f"EAN-13 must be 13 digits, got {len(barcode)}"
    expected = calculate_ean_check_digit(barcode[:12])
    actual = int(barcode[12])
    if expected != actual:
        return False, f"Invalid check digit: expected {expected}, got {actual}"
    return True, "Valid EAN-13"


def validate_ean8(barcode: str) -> Tuple[bool, str]:
    """Validate an EAN-8 barcode (8 digits with valid check digit)."""
    barcode = barcode.strip().replace(" ", "")
    if not barcode.isdigit():
        return False, "EAN-8 must contain only digits"
    if len(barcode) != 8:
        return False, f"EAN-8 must be 8 digits, got {len(barcode)}"
    expected = calculate_ean_check_digit(barcode[:7])
    actual = int(barcode[7])
    if expected != actual:
        return False, f"Invalid check digit: expected {expected}, got {actual}"
    return True, "Valid EAN-8"


def validate_upca(barcode: str) -> Tuple[bool, str]:
    """Validate a UPC-A barcode (12 digits with valid check digit)."""
    barcode = barcode.strip().replace(" ", "")
    if not barcode.isdigit():
        return False, "UPC-A must contain only digits"
    if len(barcode) != 12:
        return False, f"UPC-A must be 12 digits, got {len(barcode)}"
    expected = calculate_ean_check_digit(barcode[:11])
    actual = int(barcode[11])
    if expected != actual:
        return False, f"Invalid check digit: expected {expected}, got {actual}"
    return True, "Valid UPC-A"


def validate_barcode(barcode: str, barcode_type: str = "") -> Tuple[bool, str]:
    """Validate a barcode, auto-detecting type by length when *barcode_type* is empty."""
    barcode = barcode.strip().replace(" ", "")
    if not barcode:
        return False, "Empty barcode"

    if barcode_type:
        normalized = barcode_type.upper().replace("-", "").replace("_", "").replace(" ", "")
        if normalized == "EAN13":
            return validate_ean13(barcode)
        if normalized == "EAN8":
            return validate_ean8(barcode)
        if normalized in ("UPCA", "UPC"):
            return validate_upca(barcode)
        # Types we don't validate structurally
        return True, f"Barcode type {barcode_type} accepted (no check-digit validation)"

    # Auto-detect
    if barcode.isdigit():
        if len(barcode) == 13:
            return validate_ean13(barcode)
        if len(barcode) == 8:
            return validate_ean8(barcode)
        if len(barcode) == 12:
            return validate_upca(barcode)
    return True, "Barcode format not determined (no check-digit validation)"


def is_valid_url(url: str) -> bool:
    """Return True when *url* is a structurally valid HTTP(S) URL."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def extract_domain(url: str) -> str:
    """Extract the bare domain from a URL, stripping ``www.``."""
    try:
        domain = urlparse(url).netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

