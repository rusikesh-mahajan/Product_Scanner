"""Integration tests running the complete offline CV + OCR + Barcode pipeline on the 3 sample tag photos."""

import os
import cv2
import pytest

from modules.barcode_scanner import BarcodeScanner
from modules.image_processor import ImageProcessor
from modules.ocr_engine import EasyOCREngine, find_best_orientation
from modules.product_extractor import ProductExtractor

easyocr = pytest.importorskip("easyocr", reason="EasyOCR required for offline integration tests")
pyzbar = pytest.importorskip("pyzbar", reason="pyzbar required for offline integration tests")

SAMPLE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "sample_tags")


@pytest.fixture(scope="module")
def ocr_reader():
    """Load EasyOCR reader once for all integration tests."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(base_dir, ".easyocr", "model")
    user_network_dir = os.path.join(base_dir, ".easyocr", "user_network")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(user_network_dir, exist_ok=True)
    kwargs = {
        "gpu": False,
        "model_storage_directory": model_dir,
        "user_network_directory": user_network_dir,
    }
    return easyocr.Reader(["en"], **kwargs)


@pytest.mark.slow
def test_full_pipeline_benetton_jeans(ocr_reader):
    """Run full pipeline on benetton_jeans.jpeg."""
    path = os.path.join(SAMPLE_DIR, "benetton_jeans.jpeg")
    assert os.path.exists(path), f"Sample file missing: {path}"

    image = cv2.imread(path)
    assert image is not None

    # Step 1: Detect and crop
    cropped, _ = ImageProcessor.detect_and_crop_tag(image)

    # Step 2: OCR with orientation
    engine = EasyOCREngine(reader=ocr_reader)
    best_img, _, _ = find_best_orientation(cropped, engine)
    ocr_text, conf, _ = engine.extract_text(best_img)

    # Step 3: Barcode scanning
    barcodes = BarcodeScanner.scan(best_img, ocr_engine=engine)
    if not barcodes:
        barcodes = BarcodeScanner.scan(image, ocr_engine=engine)

    # Step 4: Parse fields
    fields = ProductExtractor.parse_fields(ocr_text, ocr_confidence=conf)

    # Verify barcode (expected 8909329191647)
    barcode_datas = [b.data for b in barcodes]
    assert "8909329191647" in barcode_datas, f"Decoded barcodes: {barcode_datas}"

    # Verify key parsed fields
    assert fields.product_type and "JEANS" in fields.product_type.upper()
    assert fields.mrp == pytest.approx(3299.0)


@pytest.mark.slow
def test_full_pipeline_jackjones_shirt(ocr_reader):
    """Run full pipeline on jackjones_shirt.jpeg."""
    path = os.path.join(SAMPLE_DIR, "jackjones_shirt.jpeg")
    assert os.path.exists(path), f"Sample file missing: {path}"

    image = cv2.imread(path)
    assert image is not None

    cropped, _ = ImageProcessor.detect_and_crop_tag(image)
    engine = EasyOCREngine(reader=ocr_reader)
    best_img, _, _ = find_best_orientation(cropped, engine)
    ocr_text, conf, _ = engine.extract_text(best_img)

    barcodes = BarcodeScanner.scan(best_img, ocr_engine=engine)
    if not barcodes:
        barcodes = BarcodeScanner.scan(image, ocr_engine=engine)

    fields = ProductExtractor.parse_fields(ocr_text, ocr_confidence=conf)

    # Verify barcode (expected 8905944710232)
    barcode_datas = [b.data for b in barcodes]
    assert "8905944710232" in barcode_datas, f"Decoded barcodes: {barcode_datas}"

    # Verify key parsed fields
    assert fields.brand and "JACK" in fields.brand.upper()
    assert fields.product_type and "SHIRT" in fields.product_type.upper()
    assert fields.mrp == pytest.approx(3299.0)


@pytest.mark.slow
def test_full_pipeline_benetton_polo(ocr_reader):
    """Run full pipeline on benetton_polo.jpeg."""
    path = os.path.join(SAMPLE_DIR, "benetton_polo.jpeg")
    assert os.path.exists(path), f"Sample file missing: {path}"

    image = cv2.imread(path)
    assert image is not None

    cropped, _ = ImageProcessor.detect_and_crop_tag(image)
    engine = EasyOCREngine(reader=ocr_reader)
    best_img, _, _ = find_best_orientation(cropped, engine)
    ocr_text, conf, _ = engine.extract_text(best_img)

    barcodes = BarcodeScanner.scan(best_img, ocr_engine=engine)
    if not barcodes:
        barcodes = BarcodeScanner.scan(image, ocr_engine=engine)

    fields = ProductExtractor.parse_fields(ocr_text, ocr_confidence=conf)

    # Verify barcode (expected 8909329255677)
    barcode_datas = [b.data for b in barcodes]
    assert "8909329255677" in barcode_datas, f"Decoded barcodes: {barcode_datas}"

    # Verify key parsed fields
    assert fields.product_type and "POLO" in fields.product_type.upper()
    assert fields.mrp == pytest.approx(2999.0)


@pytest.mark.slow
def test_accuracy_report(ocr_reader, capsys):
    """Run all 3 real images and print an evaluation accuracy table."""
    engine = EasyOCREngine(reader=ocr_reader)

    test_cases = [
        {
            "file": "benetton_jeans.jpeg",
            "expected_barcode": "8909329191647",
            "expected_brand": "Benetton",
            "expected_type": "JEANS",
            "expected_mrp": 3299.0,
            "expected_style": "26P4MB",
            "expected_color": "902",
        },
        {
            "file": "jackjones_shirt.jpeg",
            "expected_barcode": "8905944710232",
            "expected_brand": "JACK & JONES",
            "expected_type": "SHIRT",
            "expected_mrp": 3299.0,
            "expected_style": "9017901005",
            "expected_color": "Cashmere Blue",
        },
        {
            "file": "benetton_polo.jpeg",
            "expected_barcode": "8909329255677",
            "expected_brand": "Benetton",
            "expected_type": "POLO T-SHIRT",
            "expected_mrp": 2999.0,
            "expected_style": "26P3P59J3029I",
            "expected_color": "N22",
        },
    ]

    print("\n" + "=" * 80)
    print("REAL SAMPLE TAGS EVALUATION ACCURACY REPORT")
    print("=" * 80)

    for tc in test_cases:
        path = os.path.join(SAMPLE_DIR, tc["file"])
        img = cv2.imread(path)
        cropped, _ = ImageProcessor.detect_and_crop_tag(img)
        best_img, rot, _ = find_best_orientation(cropped, engine)
        ocr_text, conf, _ = engine.extract_text(best_img)

        barcodes = BarcodeScanner.scan(best_img, ocr_engine=engine)
        if not barcodes:
            barcodes = BarcodeScanner.scan(img, ocr_engine=engine)

        fields = ProductExtractor.parse_fields(ocr_text, ocr_confidence=conf)
        decoded_barcodes = [b.data for b in barcodes]

        barcode_match = tc["expected_barcode"] in decoded_barcodes
        brand_match = tc["expected_brand"].lower() in (fields.brand or "").lower()
        type_match = tc["expected_type"].lower() in (fields.product_type or "").lower()
        mrp_match = fields.mrp == tc["expected_mrp"]
        style_match = tc["expected_style"] in (fields.base_style or fields.style_code or "")
        color_match = (
            tc["expected_color"].lower() in (fields.colour_code or "").lower()
            or tc["expected_color"].lower() in (fields.colour_name or "").lower()
        )

        print(f"\nTag Image: {tc['file']}")
        print(f"  Rotation: {rot}° | OCR Confidence: {conf*100:.1f}%")
        print(f"  Barcode:  {decoded_barcodes} -> {'MATCH' if barcode_match else 'FAIL'} (Expected: {tc['expected_barcode']})")
        print(f"  Brand:    '{fields.brand}' -> {'MATCH' if brand_match else 'FAIL'}")
        print(f"  Type:     '{fields.product_type}' -> {'MATCH' if type_match else 'FAIL'}")
        print(f"  MRP:      '{fields.mrp}' -> {'MATCH' if mrp_match else 'FAIL'}")
        print(f"  Style:    '{fields.base_style or fields.style_code}' -> {'MATCH' if style_match else 'FAIL'}")
        print(f"  Color:    '{fields.colour_code or fields.colour_name}' -> {'MATCH' if color_match else 'FAIL'}")

    print("\n" + "=" * 80)

