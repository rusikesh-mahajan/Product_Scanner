"""Unit tests for image URL validation and download handling."""

from unittest.mock import MagicMock
import requests

from modules.image_downloader import ImageDownloader
from utils.validators import extract_domain, is_valid_url


def test_is_valid_url():
    """Verify URL validation patterns."""
    assert is_valid_url("https://example.com/photo.jpg") is True
    assert is_valid_url("http://domain.in/path/to/image.png?size=large") is True
    assert is_valid_url("ftp://example.com/photo.jpg") is False
    assert is_valid_url("not_a_url") is False
    assert is_valid_url("") is False


def test_extract_domain():
    """Verify domain extraction and www stripping."""
    assert extract_domain("https://www.benetton.com/product/123") == "benetton.com"
    assert extract_domain("https://myntra.com/p/item") == "myntra.com"
    assert extract_domain("http://sub.domain.co.in/path") == "sub.domain.co.in"
    assert extract_domain("invalid") == ""


def test_validate_image_url_success(monkeypatch):
    """Verify valid image HEAD check."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "image/jpeg",
        "Content-Length": "1048576",  # 1 MB
    }
    monkeypatch.setattr(requests, "head", lambda *args, **kwargs: mock_resp)

    valid, msg, ctype, size = ImageDownloader.validate_image_url("https://example.com/test.jpg")
    assert valid is True
    assert ctype == "image/jpeg"
    assert size == 1048576


def test_validate_image_url_wrong_type(monkeypatch):
    """Verify rejection of non-image MIME types."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "text/html",
        "Content-Length": "2048",
    }
    monkeypatch.setattr(requests, "head", lambda *args, **kwargs: mock_resp)

    valid, msg, ctype, size = ImageDownloader.validate_image_url("https://example.com/page.html")
    assert valid is False
    assert "Invalid content type" in msg


def test_validate_image_url_size_cap(monkeypatch):
    """Verify rejection of images exceeding the size cap."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "image/png",
        "Content-Length": str(20 * 1024 * 1024),  # 20 MB (exceeds 15 MB cap)
    }
    monkeypatch.setattr(requests, "head", lambda *args, **kwargs: mock_resp)

    valid, msg, _, _ = ImageDownloader.validate_image_url("https://example.com/giant.png")
    assert valid is False
    assert "exceeds limit" in msg

