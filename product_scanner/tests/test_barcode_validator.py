"""Unit tests for barcode check-digit validation and formatting."""

from modules.barcode_validator import BarcodeValidator
from modules.models import BarcodeResult
from utils.validators import (
    calculate_ean_check_digit,
    validate_barcode,
    validate_ean8,
    validate_ean13,
    validate_upca,
)


def test_valid_ean13_benetton_polo():
    """Verify barcode 8909329255677 from Benetton polo tag."""
    barcode = "8909329255677"
    is_valid, msg = validate_ean13(barcode)
    assert is_valid, msg
    assert calculate_ean_check_digit(barcode[:12]) == 7


def test_valid_ean13_benetton_jeans():
    """Verify barcode 8909329191647 from Benetton jeans tag."""
    barcode = "8909329191647"
    is_valid, msg = validate_ean13(barcode)
    assert is_valid, msg
    assert calculate_ean_check_digit(barcode[:12]) == 7


def test_valid_ean13_jackjones():
    """Verify barcode 8905944710232 from Jack & Jones shirt tag."""
    barcode = "8905944710232"
    is_valid, msg = validate_ean13(barcode)
    assert is_valid, msg
    assert calculate_ean_check_digit(barcode[:12]) == 2


def test_invalid_ean13_wrong_check():
    """Verify that a corrupted check digit fails validation."""
    barcode = "8909329255678"  # Should end in 7, not 8
    is_valid, msg = validate_ean13(barcode)
    assert not is_valid
    assert "Invalid check digit" in msg


def test_valid_ean8():
    """Verify standard valid EAN-8."""
    barcode = "96385074"
    is_valid, msg = validate_ean8(barcode)
    assert is_valid, msg


def test_invalid_ean8():
    """Verify invalid EAN-8 check digit."""
    barcode = "96385075"
    is_valid, msg = validate_ean8(barcode)
    assert not is_valid


def test_valid_upca():
    """Verify standard valid UPC-A."""
    barcode = "012345678905"
    is_valid, msg = validate_upca(barcode)
    assert is_valid, msg


def test_invalid_upca():
    """Verify invalid UPC-A."""
    barcode = "012345678900"
    is_valid, msg = validate_upca(barcode)
    assert not is_valid


def test_barcode_with_spaces():
    """Verify handling of printed spaces (e.g. '8 909329 255677')."""
    raw = "8 909329 255677"
    is_valid, msg = validate_barcode(raw)
    assert is_valid, msg


def test_empty_barcode():
    """Verify empty barcode rejection."""
    is_valid, msg = validate_barcode("")
    assert not is_valid
    assert "Empty" in msg


def test_detect_type():
    """Verify length-based barcode type detection."""
    assert BarcodeValidator.detect_type("8909329255677") == "EAN-13"
    assert BarcodeValidator.detect_type("96385074") == "EAN-8"
    assert BarcodeValidator.detect_type("012345678905") == "UPC-A"


def test_format_barcode():
    """Verify spaced formatting for human readability."""
    formatted = BarcodeValidator.format_barcode("8909329255677")
    assert formatted == "8 909329 255677"


def test_suggest_correction():
    """Verify check-digit suggestion for single-digit typo."""
    bad_barcode = "8909329255670"
    suggested = BarcodeValidator.suggest_correction(bad_barcode)
    assert suggested == "8909329255677"


def test_barcode_validator_model_integration():
    """Verify BarcodeValidator updates BarcodeResult instance."""
    result = BarcodeResult(
        data="8909329255677",
        type="EAN-13",
        source="pyzbar",
    )
    validated = BarcodeValidator.validate(result)
    assert validated.is_valid is True
    assert validated.validation_message == "Valid EAN-13"

