"""End-to-End integration tests for Product Tag Scanner pipeline on real sample tags."""

import os
import cv2
import pytest
from unittest.mock import MagicMock
import requests

from modules.barcode_scanner import BarcodeScanner
from modules.models import ProductRecord
from modules.ocr_engine import get_ocr_engine
from modules.official_site_finder import OfficialSiteFinder
from modules.product_extractor import ProductExtractor
from modules.product_identifier import ProductIdentifier
from modules.product_page_finder import ProductPageFinder
from services.product_verifier import ProductVerifier


@pytest.fixture(scope="module")
def cached_engine():
    """Cache EasyOCR engine for the module."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, ".easyocr", "model")
    user_net_dir = os.path.join(base_dir, ".easyocr", "user_network")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(user_net_dir, exist_ok=True)
    import easyocr
    reader = easyocr.Reader(
        ["en"],
        gpu=False,
        model_storage_directory=model_dir,
        user_network_directory=user_net_dir,
    )
    return get_ocr_engine("easyocr", reader=reader)


@pytest.mark.slow
def test_e2e_benetton_jeans_extraction(cached_engine, monkeypatch):
    """E2E test on Benetton Jeans sample tag."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(base_dir, "assets", "sample_tags", "benetton_jeans.jpeg")
    assert os.path.exists(img_path)

    img = cv2.imread(img_path)
    text, conf, _ = cached_engine.extract_text(img)
    tag_fields = ProductExtractor.parse_fields(text, ocr_confidence=conf)
    barcodes = BarcodeScanner.scan(img, ocr_engine=cached_engine)

    record = ProductIdentifier.identify(tag_fields, barcode_results=barcodes)

    assert record.brand == "United Colors of Benetton"
    assert record.product_type == "JEANS"
    assert record.base_style is not None
    assert "26P4MB" in (record.base_style or "")

    # Mock domain validation HTTP check to ensure offline test reliability
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.url = "https://in.benetton.com"
    mock_resp.text = "<html><head><title>United Colors of Benetton India</title></head><body>Official Store</body></html>"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    site = OfficialSiteFinder.find_official_site(record)
    assert site is not None
    assert "benetton" in site.domain
    assert site.is_verified is True


@pytest.mark.slow
def test_e2e_jackjones_shirt_extraction(cached_engine, monkeypatch):
    """E2E test on Jack & Jones Shirt sample tag."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(base_dir, "assets", "sample_tags", "jackjones_shirt.jpeg")
    assert os.path.exists(img_path)

    img = cv2.imread(img_path)
    text, conf, _ = cached_engine.extract_text(img)
    tag_fields = ProductExtractor.parse_fields(text, ocr_confidence=conf)
    barcodes = BarcodeScanner.scan(img, ocr_engine=cached_engine)

    record = ProductIdentifier.identify(tag_fields, barcode_results=barcodes)

    assert record.brand == "JACK & JONES"
    assert record.product_type and "SHIRT" in record.product_type.upper()

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.url = "https://www.jackjones.in"
    mock_resp.text = "<html><head><title>JACK & JONES India Official Store</title></head><body>Men fashion</body></html>"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    site = OfficialSiteFinder.find_official_site(record)
    assert site is not None
    assert "jackjones" in site.domain
    assert site.is_verified is True


@pytest.mark.slow
def test_e2e_benetton_polo_extraction(cached_engine, monkeypatch):
    """E2E test on Benetton Polo sample tag."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    img_path = os.path.join(base_dir, "assets", "sample_tags", "benetton_polo.jpeg")
    assert os.path.exists(img_path)

    img = cv2.imread(img_path)
    text, conf, _ = cached_engine.extract_text(img)
    tag_fields = ProductExtractor.parse_fields(text, ocr_confidence=conf)
    barcodes = BarcodeScanner.scan(img, ocr_engine=cached_engine)

    record = ProductIdentifier.identify(tag_fields, barcode_results=barcodes)

    assert record.brand == "United Colors of Benetton"
    assert record.product_type in ("POLO T-SHIRT", "T-SHIRT")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.url = "https://in.benetton.com"
    mock_resp.text = "<html><head><title>United Colors of Benetton India</title></head><body>Official Store</body></html>"
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    site = OfficialSiteFinder.find_official_site(record)
    assert site is not None
    assert "benetton" in site.domain
    assert site.is_verified is True
