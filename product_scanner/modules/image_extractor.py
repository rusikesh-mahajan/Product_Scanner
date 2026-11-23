"""Product image extraction module.

Fetches product pages and extracts official high-resolution photography using:
1. JSON-LD Product schema (highest fidelity: image, sku, name, brand)
2. Open Graph meta tags (og:image)
3. HTML <img> tags, <picture>, srcset, data-zoom-image, and product galleries
4. Playwright headless Chromium fallback when page is JavaScript-rendered or empty.

Filters out logos, banners, icons, tiny thumbnails (<300px), and advertisements.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse

from bs4 import BeautifulSoup
import requests

from config.settings import settings
from modules.models import ExtractedImage, ProductPage, ProductRecord
from services.search_service import SearchService
from utils.validators import extract_domain, is_valid_url

logger = logging.getLogger(__name__)


class ImageExtractor:
    """Extracts, cleans, and ranks official product imagery from brand web pages."""

    # Keywords that suggest an image is a logo, icon, or banner rather than product photo
    NON_PRODUCT_PATTERNS = re.compile(
        r"(logo|banner|icon|badge|sprite|loader|spinner|avatar|payment|visa|mastercard|amex|footer|header|social|star|rating|cart|search|arrow|chevron|tick|close)",
        re.IGNORECASE,
    )

    @classmethod
    def extract_from_page(
        cls, page_url: str, record: Optional[ProductRecord] = None
    ) -> Tuple[List[ExtractedImage], Dict[str, Any], str]:
        """Fetch page and extract candidate product images.

        Returns (extracted_images, page_metadata, status_or_error_message).
        """
        if not is_valid_url(page_url):
            return [], {}, "Invalid URL format"

        domain = extract_domain(page_url)
        SearchService.rate_limit(domain, min_delay=settings.rate_limit_delay_seconds)

        # Check robots.txt compliance
        if not SearchService.can_fetch(page_url, user_agent=settings.user_agent):
            return [], {}, f"robots.txt disallows automated fetching of {page_url}"

        headers = {
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            try:
                resp = requests.get(
                    page_url,
                    headers=headers,
                    timeout=settings.request_timeout,
                    allow_redirects=True,
                )
            except requests.exceptions.SSLError:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                logger.warning("SSL verification failed for %s; retrying with verify=False", page_url)
                resp = requests.get(
                    page_url,
                    headers=headers,
                    timeout=settings.request_timeout,
                    allow_redirects=True,
                    verify=False,
                )

            # Defensive bot-block detection
            if resp.status_code in (401, 403):
                return [], {}, f"Site blocks automated access (HTTP {resp.status_code})"

            if resp.status_code != 200:
                return [], {}, f"Failed to load page (HTTP {resp.status_code})"

            html_text = resp.text
            final_url = resp.url

            # If HTML is suspiciously small or empty, attempt Playwright fallback
            if len(html_text.strip()) < 1000 or "enable javascript" in html_text.lower():
                pw_html = cls._fetch_with_playwright(page_url)
                if pw_html:
                    html_text = pw_html

            return cls.parse_html_images(html_text, final_url, record=record)

        except requests.Timeout:
            return [], {}, "Page request timed out"
        except requests.RequestException as e:
            return [], {}, f"Network connection error: {e}"
        except Exception as e:
            return [], {}, f"Extraction error: {e}"

    @classmethod
    def _fetch_with_playwright(cls, url: str) -> Optional[str]:
        """Fallback renderer using Playwright for JavaScript-heavy client-rendered SPAs."""
        try:
            from playwright.sync_api import sync_playwright

            logger.info("Attempting Playwright fallback for JS-rendered page: %s", url)
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(user_agent=settings.user_agent)
                page.goto(url, timeout=20000, wait_until="domcontentloaded")
                # Wait briefly for product images to mount
                page.wait_for_timeout(2500)
                content = page.content()
                browser.close()
                return content
        except Exception as e:
            logger.debug(
                "Playwright fallback not available or failed: %s (run 'playwright install chromium')",
                e,
            )
            return None

    @classmethod
    def parse_html_images(
        cls, html_content: str, base_url: str, record: Optional[ProductRecord] = None
    ) -> Tuple[List[ExtractedImage], Dict[str, Any], str]:
        """Parse HTML string for JSON-LD, OG tags, and product gallery elements."""
        soup = BeautifulSoup(html_content, "html.parser")
        images: List[ExtractedImage] = []
        seen_urls = set()
        metadata: Dict[str, Any] = {
            "title": (soup.title.string or "").strip() if soup.title else "",
            "json_ld_products": [],
            "sku_found": "",
            "brand_found": "",
            "name_found": "",
            "raw_text": soup.get_text()[:4000],
        }

        # ---------------------------------------------------------------------
        # 1. JSON-LD Product Schema (High fidelity)
        # ---------------------------------------------------------------------
        json_ld_scripts = soup.find_all("script", type="application/ld+json")
        for s in json_ld_scripts:
            try:
                if not s.string:
                    continue
                data = json.loads(s.string)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    item_type = str(item.get("@type", ""))
                    if "Product" in item_type:
                        metadata["json_ld_products"].append(item)
                        if item.get("sku"):
                            metadata["sku_found"] = str(item["sku"])
                        if item.get("name"):
                            metadata["name_found"] = str(item["name"])
                        brand_obj = item.get("brand")
                        if isinstance(brand_obj, dict):
                            metadata["brand_found"] = brand_obj.get("name", "")
                        elif isinstance(brand_obj, str):
                            metadata["brand_found"] = brand_obj

                        raw_imgs = item.get("image", [])
                        img_list = raw_imgs if isinstance(raw_imgs, list) else [raw_imgs]
                        for img_entry in img_list:
                            img_u = ""
                            if isinstance(img_entry, str):
                                img_u = img_entry
                            elif isinstance(img_entry, dict) and "url" in img_entry:
                                img_u = img_entry["url"]
                            if img_u:
                                abs_u = urllib.parse.urljoin(base_url, img_u)
                                if abs_u not in seen_urls and cls._is_acceptable_image_url(abs_u):
                                    seen_urls.add(abs_u)
                                    images.append(
                                        ExtractedImage(
                                            url=abs_u,
                                            source_type="json_ld",
                                            is_main=True,
                                            resolution_score=100,
                                        )
                                    )
            except Exception as e:
                logger.debug("Error parsing JSON-LD: %s", e)

        # ---------------------------------------------------------------------
        # 2. Open Graph og:image
        # ---------------------------------------------------------------------
        og_images = soup.find_all("meta", property=["og:image", "og:image:secure_url"])
        for meta in og_images:
            content = meta.get("content")
            if content:
                abs_u = urllib.parse.urljoin(base_url, content)
                if abs_u not in seen_urls and cls._is_acceptable_image_url(abs_u):
                    seen_urls.add(abs_u)
                    images.append(
                        ExtractedImage(
                            url=abs_u,
                            source_type="og_image",
                            is_main=len(images) == 0,
                            resolution_score=85,
                        )
                    )

        # ---------------------------------------------------------------------
        # 3. HTML Main Product & Gallery Images
        # ---------------------------------------------------------------------
        # Target product-specific containers if present
        product_containers = soup.find_all(
            ["div", "section", "figure"],
            class_=re.compile(r"(product|gallery|pdp|image-viewer|media)", re.I),
        )
        search_roots = product_containers if product_containers else [soup]

        for root in search_roots:
            img_tags = root.find_all("img")
            for img in img_tags:
                candidate_urls = cls._extract_urls_from_img_tag(img, base_url)
                alt = img.get("alt", "")

                for u in candidate_urls:
                    if u not in seen_urls and cls._is_acceptable_image_url(u, alt):
                        seen_urls.add(u)
                        images.append(
                            ExtractedImage(
                                url=u,
                                source_type="html_img",
                                alt_text=alt,
                                is_main=len(images) == 0,
                                resolution_score=60,
                            )
                        )

        # Sort images: json_ld > og_image > html_img
        source_priority = {"json_ld": 3, "og_image": 2, "html_img": 1}
        images.sort(
            key=lambda x: (source_priority.get(x.source_type, 0), x.resolution_score),
            reverse=True,
        )

        status_msg = f"Extracted {len(images)} candidate image(s)" if images else "No suitable product images found"
        return images, metadata, status_msg

    @classmethod
    def _extract_urls_from_img_tag(cls, img_tag: Any, base_url: str) -> List[str]:
        """Extract all candidate image URLs from an <img> tag (src, data-zoom, srcset)."""
        urls: List[str] = []

        # 1. High-resolution zoom attributes
        for attr in ["data-zoom-image", "data-zoom", "data-large", "data-high-res", "data-src", "data-original"]:
            val = img_tag.get(attr)
            if val and isinstance(val, str):
                urls.append(urllib.parse.urljoin(base_url, val))

        # 2. srcset (pick largest resolution)
        srcset = img_tag.get("srcset")
        if srcset and isinstance(srcset, str):
            best_src = cls._pick_largest_from_srcset(srcset, base_url)
            if best_src:
                urls.append(best_src)

        # 3. Standard src
        src = img_tag.get("src")
        if src and isinstance(src, str):
            urls.append(urllib.parse.urljoin(base_url, src))

        return urls

    @classmethod
    def _pick_largest_from_srcset(cls, srcset: str, base_url: str) -> Optional[str]:
        """Parse srcset string and return the URL with the largest width descriptor."""
        candidates = []
        for part in srcset.split(","):
            part = part.strip()
            if not part:
                continue
            pieces = part.split()
            url = pieces[0]
            width = 0
            if len(pieces) > 1:
                w_str = pieces[1].rstrip("w").rstrip("x")
                try:
                    width = int(float(w_str))
                except ValueError:
                    width = 0
            candidates.append((width, url))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return urllib.parse.urljoin(base_url, candidates[0][1])
        return None

    @classmethod
    def _is_acceptable_image_url(cls, url: str, alt_text: str = "") -> bool:
        """Filter out non-product imagery like logos, banners, icons, tiny SVG icons."""
        if not url or not is_valid_url(url):
            return False

        # Reject data URIs or inline svgs
        if url.startswith("data:"):
            return False

        url_lower = url.lower()

        # Reject SVGs (mostly icons/logos)
        if url_lower.endswith(".svg") or ".svg?" in url_lower:
            return False

        # Reject known logo/banner keywords in URL or Alt text
        if cls.NON_PRODUCT_PATTERNS.search(url_lower):
            return False
        if alt_text and cls.NON_PRODUCT_PATTERNS.search(alt_text.lower()):
            return False

        # Reject tiny dimension indicators in URL (e.g. 50x50, 100x100)
        tiny_match = re.search(r"(\d+)x(\d+)", url_lower)
        if tiny_match:
            try:
                w, h = int(tiny_match.group(1)), int(tiny_match.group(2))
                if w < 250 or h < 250:
                    return False
            except ValueError:
                pass

        return True
