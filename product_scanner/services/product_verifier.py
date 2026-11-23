"""Product verifier service.

Evaluates match confidence between product tag record, product webpage,
and candidate extracted image using a configurable 100-point rubric:
- Brand match: 20
- Exact model/style/SKU found on page: 50
- Product name/type match: 15
- Verified official domain + image on page: 15

Determines confidence tiers:
- 90-100: Auto-download, marked 'Verified official image'
- 70-89: Review needed, marked 'User-confirmed' if confirmed
- <70: Rejected with explicit explanation
"""

import logging
import re
from typing import Any, Dict, List, Optional

from config.settings import settings
from modules.models import (
    ExtractedImage,
    OfficialSite,
    ProductPage,
    ProductRecord,
    VerificationScore,
)
from utils.validators import extract_domain

logger = logging.getLogger(__name__)


class ProductVerifier:
    """Verifies authenticity and exact match of discovered product images."""

    @classmethod
    def verify(
        cls,
        record: ProductRecord,
        official_site: OfficialSite,
        page: ProductPage,
        image: ExtractedImage,
        page_metadata: Optional[Dict[str, Any]] = None,
    ) -> VerificationScore:
        """Compute verification score for an image found on an official product page."""
        meta = page_metadata or {}
        haystack = f"{page.url} {page.title} {page.snippet} {meta.get('raw_text', '')}".lower()
        json_sku = (meta.get("sku_found") or "").lower()
        json_name = (meta.get("name_found") or "").lower()
        json_brand = (meta.get("brand_found") or "").lower()

        breakdown: Dict[str, Any] = {}
        reasons: List[str] = []

        # ---------------------------------------------------------------------
        # 1. Brand Match (Weight: 20 pts)
        # ---------------------------------------------------------------------
        brand_score = 0
        tag_brand = (record.brand or "").strip()
        if tag_brand:
            brand_clean = re.sub(r"[^\w\s]", "", tag_brand).lower()
            tokens = [t for t in brand_clean.split() if t not in ("of", "and", "the", "in", "india")]

            brand_in_official_site = (tag_brand.lower() in official_site.brand.lower())
            brand_in_page = any(t in haystack for t in tokens if len(t) >= 4)
            brand_in_json = any(t in json_brand for t in tokens if len(t) >= 4) if json_brand else False

            if brand_in_official_site or brand_in_json or brand_in_page:
                brand_score = settings.score_weight_brand
                breakdown["brand"] = {
                    "score": brand_score,
                    "max": settings.score_weight_brand,
                    "status": "PASS",
                    "detail": f"Brand '{tag_brand}' matches official domain & page content",
                }
            else:
                breakdown["brand"] = {
                    "score": 0,
                    "max": settings.score_weight_brand,
                    "status": "FAIL",
                    "detail": f"Brand '{tag_brand}' could not be confirmed on page",
                }
                reasons.append("Brand name not confirmed on webpage")
        else:
            breakdown["brand"] = {
                "score": 0,
                "max": settings.score_weight_brand,
                "status": "FAIL",
                "detail": "No brand extracted from tag",
            }
            reasons.append("No brand extracted from tag")

        # ---------------------------------------------------------------------
        # 2. Exact Style / Model / SKU Match (Weight: 50 pts)
        # ---------------------------------------------------------------------
        style_score = 0
        style_tokens: List[str] = []
        if record.base_style:
            style_tokens.append(record.base_style.strip().lower())
        if record.style_code and record.style_code.lower() not in style_tokens:
            style_tokens.append(record.style_code.strip().lower())
        for v in record.style_variants:
            if v and v.lower() not in style_tokens:
                style_tokens.append(v.strip().lower())

        found_style_match: Optional[str] = None

        # Check JSON-LD SKU first (highest accuracy)
        if json_sku:
            for st in style_tokens:
                if len(st) >= 4 and (st in json_sku or json_sku in st):
                    found_style_match = f"JSON-LD SKU '{json_sku}'"
                    break

        # Check URL and page content
        if not found_style_match:
            for st in style_tokens:
                if len(st) >= 4 and st in haystack:
                    found_style_match = f"Page text/URL containing '{st}'"
                    break

        if found_style_match:
            style_score = settings.score_weight_style
            breakdown["style_code"] = {
                "score": style_score,
                "max": settings.score_weight_style,
                "status": "PASS",
                "detail": f"Exact style code found in {found_style_match}",
            }
        else:
            breakdown["style_code"] = {
                "score": 0,
                "max": settings.score_weight_style,
                "status": "FAIL",
                "detail": f"Style code ({', '.join(style_tokens[:3])}) not found on page",
            }
            reasons.append(
                f"Tag style code '{record.base_style or record.style_code}' does not appear in product page"
            )

        # ---------------------------------------------------------------------
        # 3. Product Name / Type Match (Weight: 15 pts)
        # ---------------------------------------------------------------------
        name_score = 0
        pname = (record.product_name or "").strip().lower()
        ptype = (record.product_type or "").strip().lower()

        name_match = False
        match_label = ""
        if pname and len(pname) >= 3 and (pname in haystack or pname in json_name):
            name_match = True
            match_label = f"Product Name '{pname}'"
        elif ptype and len(ptype) >= 3 and (ptype in haystack or ptype in json_name):
            name_match = True
            match_label = f"Product Type '{ptype}'"

        if name_match:
            name_score = settings.score_weight_name
            breakdown["product_name"] = {
                "score": name_score,
                "max": settings.score_weight_name,
                "status": "PASS",
                "detail": f"Matched {match_label}",
            }
        else:
            breakdown["product_name"] = {
                "score": 0,
                "max": settings.score_weight_name,
                "status": "PARTIAL/FAIL",
                "detail": f"Neither product name '{pname}' nor type '{ptype}' clearly matched",
            }

        # ---------------------------------------------------------------------
        # 4. Verified Official Domain & Page-Referenced Image (Weight: 15 pts)
        # ---------------------------------------------------------------------
        domain_image_score = 0
        img_domain = extract_domain(image.url).lower()
        official_domain = official_site.domain.lower()

        # Image must be hosted on brand domain or brand CDN or referenced by verified page
        domain_ok = (
            official_site.is_verified
            and (
                official_domain in page.domain.lower()
                or page.domain.lower() in official_domain
            )
        )
        if domain_ok and image.url:
            domain_image_score = settings.score_weight_domain_image
            breakdown["domain_and_image"] = {
                "score": domain_image_score,
                "max": settings.score_weight_domain_image,
                "status": "PASS",
                "detail": f"Image extracted directly from verified official domain '{official_site.domain}'",
            }
        else:
            breakdown["domain_and_image"] = {
                "score": 0,
                "max": settings.score_weight_domain_image,
                "status": "FAIL",
                "detail": "Image is not hosted on or referenced by verified official domain",
            }
            reasons.append("Image is not from verified official domain")

        # ---------------------------------------------------------------------
        # Total Score & Tier Classification
        # ---------------------------------------------------------------------
        total = brand_score + style_score + name_score + domain_image_score

        # Supporting attribute checks (Colour name/code on page)
        colour = (record.colour_name or record.colour_code or "").strip().lower()
        if colour and len(colour) >= 3 and colour in haystack:
            breakdown["colour_check"] = {"status": "CONFIRMED", "colour": colour}

        if total >= settings.score_auto_download_threshold:
            status = "verified"
            rejection_reason = ""
        elif total >= settings.score_review_threshold:
            status = "review"
            rejection_reason = "Confidence score (70-89) requires manual user confirmation before download"
        else:
            status = "rejected"
            rejection_reason = "; ".join(reasons) if reasons else "Confidence score below 70 threshold"

        return VerificationScore(
            brand_score=brand_score,
            style_score=style_score,
            name_score=name_score,
            domain_image_score=domain_image_score,
            total_score=total,
            status=status,
            breakdown=breakdown,
            rejection_reason=rejection_reason,
        )
