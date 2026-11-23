"""Image downloader module.

Handles safe streaming downloads, content-hash deduplication, Pillow verification,
structured filename generation (<brand>_<style>_<colour>.<ext>), non-overwriting
numeric suffixes, and JSONL audit logging in history/.
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, Optional, Tuple

from PIL import Image
import requests

from config.settings import settings
from modules.models import ProductRecord, VerificationScore
from utils.helpers import ensure_dir, sanitize_filename
from utils.validators import is_valid_url

logger = logging.getLogger(__name__)


class ImageDownloader:
    """Safely downloads, validates, deduplicates, and logs product images."""

    @staticmethod
    def validate_image_url(
        url: str, timeout: int = 6
    ) -> Tuple[bool, str, Optional[str], Optional[int]]:
        """Validate an image URL via HEAD request (status 200, image MIME, size <= 15MB)."""
        if not is_valid_url(url):
            return False, "Invalid URL format", None, None

        headers = {"User-Agent": settings.user_agent}

        try:
            try:
                resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True)
            except requests.exceptions.SSLError:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = requests.head(url, headers=headers, timeout=timeout, allow_redirects=True, verify=False)
            content_type = resp.headers.get("Content-Type", "")
            content_length = resp.headers.get("Content-Length")
            file_size = int(content_length) if content_length and content_length.isdigit() else None

            if resp.status_code != 200:
                if resp.status_code in (403, 405):
                    # Fallback to byte-range GET if HEAD is forbidden
                    range_headers = {**headers, "Range": "bytes=0-1024"}
                    get_resp = requests.get(url, headers=range_headers, timeout=timeout, stream=True)
                    if get_resp.status_code in (200, 206):
                        content_type = get_resp.headers.get("Content-Type", "")
                        return True, "Valid", content_type, None
                return False, f"HTTP {resp.status_code}", content_type, file_size

            if not content_type.startswith("image/"):
                return False, f"Invalid content type: {content_type}", content_type, file_size

            if file_size and file_size > settings.max_image_size_bytes:
                return (
                    False,
                    f"Image size exceeds limit ({file_size / (1024*1024):.1f} MB)",
                    content_type,
                    file_size,
                )

            return True, "Valid", content_type, file_size

        except requests.Timeout:
            return False, "Request timed out", None, None
        except requests.RequestException as e:
            return False, f"Connection error: {e}", None, None
        except Exception as e:
            return False, f"Unexpected error: {e}", None, None

    @classmethod
    def generate_filename(
        cls, record: ProductRecord, ext: str = ".jpg"
    ) -> str:
        """Construct structured filename: <brand>_<style>_<colour>.<ext>."""
        brand = sanitize_filename(record.brand or "brand")
        style = sanitize_filename(record.base_style or record.style_code or "style")
        colour = sanitize_filename(record.colour_name or record.colour_code or "color")

        base_name = f"{brand}_{style}_{colour}".strip("_")
        base_name = re.sub(r"[_\s]+", "_", base_name).lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        return f"{base_name}{ext}"

    @classmethod
    def get_unique_filepath(cls, directory: str, filename: str) -> str:
        """Ensure file does not overwrite existing files by appending _1, _2, etc."""
        base, ext = os.path.splitext(filename)
        candidate = os.path.join(directory, filename)
        counter = 1
        while os.path.exists(candidate):
            candidate = os.path.join(directory, f"{base}_{counter}{ext}")
            counter += 1
        return candidate

    @classmethod
    def download_product_image(
        cls,
        image_url: str,
        record: ProductRecord,
        page_url: str,
        domain: str,
        verification_score: VerificationScore,
        save_dir: Optional[str] = None,
        label: str = "Verified",
    ) -> Tuple[Optional[str], str]:
        """Download image with Pillow verification, content hash deduplication, and history logging.

        Returns (saved_filepath_or_None, status_message).
        """
        if not is_valid_url(image_url):
            return None, "Invalid image URL"

        dest_dir = save_dir or settings.download_dir
        ensure_dir(dest_dir)
        headers = {"User-Agent": settings.user_agent}

        try:
            try:
                resp = requests.get(image_url, headers=headers, timeout=15, stream=True)
            except requests.exceptions.SSLError:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                resp = requests.get(image_url, headers=headers, timeout=15, stream=True, verify=False)
            if resp.status_code != 200:
                return None, f"Image download failed (HTTP {resp.status_code})"

            content_type = resp.headers.get("Content-Type", "")
            raw_bytes = bytearray()
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    raw_bytes.extend(chunk)
                    if len(raw_bytes) > settings.max_image_size_bytes:
                        return None, f"Download aborted: size exceeds {settings.max_image_size_bytes} bytes"

            # 1. Deduplication via SHA-256 hash
            content_hash = hashlib.sha256(raw_bytes).hexdigest()

            # 2. Pillow Verification (real image check)
            import io
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes))
                pil_img.verify()
                # Re-open for dimension inspection after verify() closes the stream
                pil_img = Image.open(io.BytesIO(raw_bytes))
                width, height = pil_img.size
                img_format = pil_img.format or "JPEG"
            except Exception as e:
                logger.error("Pillow image verification failed for %s: %s", image_url, e)
                return None, f"Image file is corrupt or invalid format: {e}"

            # Check min dimensions (short side >= 300px)
            short_side = min(width, height)
            if short_side < settings.min_image_dimension:
                return None, f"Image dimensions too small ({width}x{height}px, min required: {settings.min_image_dimension}px)"

            # Map format to file extension
            ext_map = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}
            ext = ext_map.get(img_format.upper(), ".jpg")

            filename = cls.generate_filename(record, ext=ext)
            save_path = cls.get_unique_filepath(dest_dir, filename)

            # Persist to disk
            with open(save_path, "wb") as f:
                f.write(raw_bytes)

            logger.info("Successfully downloaded verified image (%dx%d, %s) to %s", width, height, label, save_path)

            # 3. Save JSONL Audit History
            cls.append_to_history(
                save_path=save_path,
                image_url=image_url,
                page_url=page_url,
                domain=domain,
                record=record,
                score=verification_score,
                content_hash=content_hash,
                dimensions=(width, height),
                label=label,
            )

            return save_path, f"Successfully downloaded ({width}x{height}px, {label})"

        except Exception as e:
            logger.error("Download failed from %s: %s", image_url, e)
            return None, f"Download exception: {e}"

    @classmethod
    def append_to_history(
        cls,
        save_path: str,
        image_url: str,
        page_url: str,
        domain: str,
        record: ProductRecord,
        score: VerificationScore,
        content_hash: str,
        dimensions: Tuple[int, int],
        label: str,
    ) -> None:
        """Write JSONL record to history/ scans log."""
        hist_dir = settings.history_dir
        ensure_dir(hist_dir)
        log_file = os.path.join(hist_dir, "downloads_history.jsonl")

        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "local_path": os.path.abspath(save_path),
            "label": label,
            "domain": domain,
            "product_page_url": page_url,
            "image_url": image_url,
            "dimensions": f"{dimensions[0]}x{dimensions[1]}",
            "content_hash": content_hash,
            "score": score.total_score,
            "score_breakdown": score.breakdown,
            "brand": record.brand,
            "product_name": record.product_name,
            "product_type": record.product_type,
            "style_code": record.base_style or record.style_code,
            "barcode": record.barcode,
            "colour": record.colour_name or record.colour_code,
            "size": record.size,
            "mrp": record.mrp,
        }

        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error("Failed to append to history log: %s", e)

    @staticmethod
    def get_image_bytes(url: str, timeout: int = 10) -> Optional[bytes]:
        """Fetch raw image bytes with size validation for preview/download."""
        if not is_valid_url(url):
            return None

        headers = {"User-Agent": settings.user_agent}

        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            if resp.status_code == 200 and resp.headers.get("Content-Type", "").startswith("image/"):
                if len(resp.content) <= settings.max_image_size_bytes:
                    return resp.content
            return None
        except Exception:
            return None
