"""Unit tests for OfficialSiteFinder with mocked network responses."""

from unittest.mock import MagicMock
import requests

from modules.models import ProductRecord
from modules.official_site_finder import OfficialSiteFinder


def test_is_blocked_domains():
    """Verify marketplace blocklist identification."""
    assert OfficialSiteFinder.is_blocked("amazon.in") is True
    assert OfficialSiteFinder.is_blocked("https://www.myntra.com/jeans") is True
    assert OfficialSiteFinder.is_blocked("https://flipkart.com/item") is True
    assert OfficialSiteFinder.is_blocked("ajio.com") is True
    assert OfficialSiteFinder.is_blocked("benetton.co.in") is False
    assert OfficialSiteFinder.is_blocked("jackjones.in") is False


def test_find_official_site_via_email(monkeypatch):
    """Verify discovery via customer care email domain."""
    record = ProductRecord(
        brand="United Colors of Benetton",
        customer_care_email="c.services@benetton.co.in",
    )

    # Mock domain validation HTTP response
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.url = "https://benetton.co.in"
    mock_resp.text = "<html><head><title>Benetton Official India Online Store</title></head><body>Welcome to United Colors of Benetton</body></html>"

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    site = OfficialSiteFinder.find_official_site(record)
    assert site is not None
    assert site.domain in ("benetton.co.in", "benetton.in")
    assert site.is_verified is True
    assert site.verification_method in ("email_domain", "cached_verified")


def test_domain_validation_failure(monkeypatch):
    """Verify validation fails if homepage does not mention brand."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.url = "https://unrelated-domain.com"
    mock_resp.text = "<html><head><title>Random Store</title></head><body>No brand here</body></html>"

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: mock_resp)

    is_valid, _, reason = OfficialSiteFinder.validate_domain("unrelated-domain.com", "Benetton")
    assert is_valid is False
    assert "not found" in reason
