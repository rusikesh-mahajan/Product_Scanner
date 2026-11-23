"""Unit tests for the rule-based tag field parser against sample tags."""

import pytest

from modules.product_extractor import ProductExtractor


def test_benetton_jeans_parsing(benetton_jeans_ocr_text):
    """Test field extraction from Benetton jeans tag OCR text."""
    fields = ProductExtractor.parse_fields(benetton_jeans_ocr_text)

    # Brand inference
    assert fields.brand is not None
    assert "benetton" in fields.brand.lower()
    assert fields.brand_source == "Inferred from tag text"

    # Commodity
    assert fields.product_type is not None
    assert "JEANS" in fields.product_type.upper()

    # Price
    assert fields.mrp == pytest.approx(3299.0)

    # Style and size suffix
    assert fields.base_style is not None
    assert "26P4MB" in fields.base_style
    assert fields.size_suffix == "30"
    assert fields.size == "30"

    # Measurements
    assert "waist" in fields.measurements
    assert "80 cm" in fields.measurements["waist"]
    assert "inseam_length" in fields.measurements
    assert "82 cm" in fields.measurements["inseam_length"]

    # Colour and theme
    assert fields.colour_code == "902"
    assert fields.theme_code == "A03"

    # Gender & Season
    assert fields.gender == "MENS"
    assert fields.season == "SUMMER"

    # Marketer and contact
    assert fields.marketer is not None
    assert "Benetton" in fields.marketer
    assert fields.customer_care_email == "c.services@benetton.co.in"
    assert "+91-124-4590433" in (fields.customer_care_phone or "")


def test_jackjones_shirt_parsing(jackjones_shirt_ocr_text):
    """Test field extraction from Jack & Jones shirt tag OCR text."""
    fields = ProductExtractor.parse_fields(jackjones_shirt_ocr_text)

    # Brand (printed)
    assert fields.brand is not None
    assert "JACK" in fields.brand.upper()
    assert fields.brand_source == "OCR"

    # Product Name & Generic Name
    assert fields.product_name is not None
    assert "WAVORIA" in fields.product_name.upper()
    assert fields.product_type is not None
    assert "SHIRT" in fields.product_type.upper()

    # Style No
    assert fields.style_code == "9017901005"

    # Size & Fabric length
    assert fields.size == "M"

    # Colour Name
    assert fields.colour_name is not None
    assert "Cashmere Blue" in fields.colour_name

    # Price
    assert fields.mrp == pytest.approx(3299.0)

    # Manufacturing details
    assert fields.manufacturing_date is not None
    assert "Jul2025" in fields.manufacturing_date
    assert fields.manufacturer is not None
    assert "Carnation" in fields.manufacturer
    assert fields.marketer is not None
    assert "Best United" in fields.marketer
    assert fields.country_of_origin == "India"
    assert fields.customer_care_email == "help@jackjones.in"


def test_benetton_polo_parsing(benetton_polo_ocr_text):
    """Test field extraction from Benetton polo tag OCR text."""
    fields = ProductExtractor.parse_fields(benetton_polo_ocr_text)

    # Brand inference
    assert fields.brand is not None
    assert "benetton" in fields.brand.lower()

    # Commodity
    assert fields.product_type is not None
    assert "POLO" in fields.product_type.upper()

    # Price
    assert fields.mrp == pytest.approx(2999.0)

    # Style
    assert fields.base_style is not None
    assert "26P3P59J3029I" in fields.base_style
    assert fields.size_suffix == "M"
    assert fields.size == "M"

    # Measurements
    assert "length" in fields.measurements
    assert "71 cm" in fields.measurements["length"]
    assert "chest" in fields.measurements
    assert "98 cm" in fields.measurements["chest"]

    # Colour and theme
    assert fields.colour_code == "N22"
    assert fields.theme_code == "C69"

    # Gender & Season
    assert fields.gender == "MENS"
    assert fields.season == "SUMMER"


def test_generate_style_variants():
    """Verify OCR confusion normalisation generates O<->0 variants."""
    code = "26P4MBOO5043I"
    variants = ProductExtractor.generate_style_variants(code)
    assert "26P4MB005043I" in variants
    assert "26P4MBOO5043I" in variants

