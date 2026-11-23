"""Tests for product lookup API providers and ProductLookupService."""

from unittest.mock import MagicMock
import requests

from api.barcode_lookup import BarcodeLookupProvider
from api.upcitemdb import UPCitemdbProvider
from config.settings import settings
from services.product_lookup_service import ProductLookupService


def test_upcitemdb_success(monkeypatch):
    """Verify UPCitemdb response parsing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "code": "OK",
        "total": 1,
        "items": [
            {
                "title": "Classic Polo Shirt",
                "brand": "Benetton",
                "description": "Short sleeve polo shirt",
                "category": "Apparel",
                "images": ["https://img.upcitemdb.com/polo.jpg"],
            }
        ],
    }

    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: mock_resp)

    provider = UPCitemdbProvider()
    result = provider.lookup("8909329255677")

    assert result is not None
    assert result.title == "Classic Polo Shirt"
    assert result.brand == "Benetton"
    assert "UPCitemdb" in result.source


def test_upcitemdb_rate_limit(monkeypatch):
    """Verify UPCitemdb gracefully handles HTTP 429 rate limit."""
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: mock_resp)

    provider = UPCitemdbProvider()
    result = provider.lookup("8909329255677")
    assert result is None


def test_provider_404(monkeypatch):
    """Verify 404 response returns None without crashing."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: mock_resp)

    prov = UPCitemdbProvider()
    assert prov.lookup("9999999999999") is None


def test_provider_timeout(monkeypatch):
    """Verify timeout exception is caught defensively."""
    def timeout_call(*args, **kwargs):
        raise requests.Timeout("Connection timed out")

    monkeypatch.setattr(requests, "request", timeout_call)

    prov = UPCitemdbProvider()
    assert prov.lookup("9999999999999") is None


def test_provider_auth_failure(monkeypatch):
    """Verify HTTP 401/403 is caught and returns None."""
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    monkeypatch.setattr(requests, "request", lambda *args, **kwargs: mock_resp)

    provider = UPCitemdbProvider()
    assert provider.lookup("1234567890123") is None


def test_barcode_lookup_no_key(monkeypatch):
    """Verify BarcodeLookup silently skips if no key configured."""
    monkeypatch.setattr(settings, "barcode_lookup_api_key", "")
    provider = BarcodeLookupProvider()
    assert provider.lookup("8909329255677") is None


def test_product_lookup_service(monkeypatch):
    """Verify ProductLookupService orchestrator delegates cleanly."""
    service = ProductLookupService()
    # Empty barcode returns None
    assert service.lookup("") is None

    # Mock provider returning a result
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.lookup.return_value = None
    service.providers = [mock_provider]

    res = service.lookup("8909329255677")
    assert res is None
    mock_provider.lookup.assert_called_once_with("8909329255677")
