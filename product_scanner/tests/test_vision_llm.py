"""Unit tests for VisionLLMExtractor with mocked Gemini 1.5 Flash API responses."""

import json
from unittest.mock import MagicMock
import numpy as np
import pytest
import requests

from modules.models import TagFields
from modules.vision_llm_extractor import VisionLLMExtractor


def test_vision_llm_no_key():
    """Verify returns None if no API key is set."""
    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    res = VisionLLMExtractor.extract(dummy_img, api_key="")
    assert res is None


def test_vision_llm_successful_extraction(monkeypatch):
    """Verify structured parsing from Gemini 1.5 Flash Vision response."""
    mock_payload = {
        "brand": "United Colors of Benetton",
        "brand_source": "Printed on tag",
        "product_name": "Slim Fit Denim Jeans",
        "product_type": "JEANS",
        "category": "Apparel - Bottomwear",
        "style_code": "26P4MBOO50431-30",
        "base_style": "26P4MBOO50431",
        "size_suffix": "30",
        "size": "30",
        "mrp": 3299.0,
        "currency": "INR",
        "colour_code": "902",
        "colour_name": "Indigo Blue",
        "theme_code": "A0J",
        "manufacturer": "Benetton India Pvt Ltd",
        "customer_care_email": "c.services@benetton.co.in",
        "customer_care_phone": "+91-124-4590433",
        "country_of_origin": "India",
        "gender": "MENS",
        "season": "SUMMER",
        "net_quantity": "1 N",
        "measurements": {
            "waist": "80 cm",
            "inseam_length": "82 cm"
        },
        "raw_text": "Sample tag OCR text transcribed by vision model"
    }

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": json.dumps(mock_payload)}
                    ]
                }
            }
        ]
    }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    fields = VisionLLMExtractor.extract(dummy_img, api_key="fake-gemini-key")

    assert fields is not None
    assert fields.brand == "United Colors of Benetton"
    assert fields.base_style == "26P4MBOO50431"
    assert fields.size_suffix == "30"
    assert fields.product_type == "JEANS"
    assert fields.mrp == 3299.0
    assert fields.customer_care_email == "c.services@benetton.co.in"
    assert fields.colour_code == "902"
    assert fields.measurements.get("waist") == "80 cm"
    assert len(fields.style_code_variants) >= 1
    assert "Vision LLM" in fields.field_sources.get("base_style", "")


def test_vision_llm_api_error_graceful_handling(monkeypatch):
    """Verify returns None without crashing on HTTP errors."""
    mock_resp = MagicMock()
    mock_resp.status_code = 403
    mock_resp.text = "API Key not valid"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    fields = VisionLLMExtractor.extract(dummy_img, api_key="invalid-key")
    assert fields is None
