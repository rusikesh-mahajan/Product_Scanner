"""Unit tests for Tavily search integration."""

from unittest.mock import MagicMock
import pytest
import requests

from config.settings import settings
from services.search_service import SearchService


def test_tavily_web_search(monkeypatch):
    """Verify Tavily web search response parsing."""
    mock_payload = {
        "results": [
            {
                "title": "US Polo Assn Official Store",
                "url": "https://uspoloassn.in/products/brandon-jeans",
                "content": "Official Brandon Slim Tapered Jeans UDJEN1814",
                "score": 0.95,
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test-key")

    results = SearchService._search_tavily_web("US Polo Jeans UDJEN1814")
    assert len(results) == 1
    assert results[0]["title"] == "US Polo Assn Official Store"
    assert results[0]["url"] == "https://uspoloassn.in/products/brandon-jeans"
    assert results[0]["domain"] == "uspoloassn.in"
    assert results[0]["source"] == "tavily_web"


def test_tavily_image_search(monkeypatch):
    """Verify Tavily image search response parsing."""
    mock_payload = {
        "images": [
            "https://uspoloassn.in/cdn/shop/files/jeans_front.jpg",
            {"url": "https://uspoloassn.in/cdn/shop/files/jeans_back.jpg", "description": "Back view"}
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_payload

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-test-key")

    images = SearchService._search_tavily_images("US Polo Jeans")
    assert len(images) == 2
    assert images[0]["image_url"] == "https://uspoloassn.in/cdn/shop/files/jeans_front.jpg"
    assert images[0]["source"] == "tavily_images"
    assert images[1]["image_url"] == "https://uspoloassn.in/cdn/shop/files/jeans_back.jpg"
    assert images[1]["title"] == "Back view"


def test_tavily_api_error_handling(monkeypatch):
    """Verify graceful handling when Tavily returns an HTTP error."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "Unauthorized: Invalid API key"

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: mock_resp)
    monkeypatch.setattr(settings, "tavily_api_key", "tvly-bad-key")

    assert SearchService._search_tavily_web("test") == []
    assert SearchService._search_tavily_images("test") == []
