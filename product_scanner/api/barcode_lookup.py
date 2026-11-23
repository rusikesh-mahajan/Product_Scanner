"""Barcode Lookup API provider (paid service, requires BARCODE_LOOKUP_API_KEY)."""

import logging
from typing import List, Optional

from api.base import BaseProvider
from config.settings import settings
from modules.models import ProductResult

logger = logging.getLogger(__name__)


class BarcodeLookupProvider(BaseProvider):
    """Provider for Barcode Lookup API (https://www.barcodelookup.com/api)."""

    name: str = "Barcode Lookup"

    def lookup(self, barcode: str) -> Optional[ProductResult]:
        """Look up barcode if API key is configured; otherwise silently returns None."""
        if not settings.has_barcode_lookup_key:
            logger.debug("Barcode Lookup API key not provided; skipping.")
            return None

        clean_barcode = barcode.strip().replace(" ", "")
        url = "https://api.barcodelookup.com/v3/products"
        params = {
            "barcode": clean_barcode,
            "formatted": "y",
            "key": settings.barcode_lookup_api_key,
        }

        resp = self._make_request(url, params=params)
        if resp is None:
            return None

        try:
            data = resp.json()
        except Exception as e:
            logger.error("Failed to parse Barcode Lookup JSON: %s", e)
            return None

        products = data.get("products", [])
        if not products:
            logger.info("Barcode Lookup: no products returned for %s", clean_barcode)
            return None

        p = products[0]
        raw_images = p.get("images", [])
        images: List[str] = []
        for img in raw_images:
            if isinstance(img, str) and img:
                images.append(img)
            elif isinstance(img, dict) and "url" in img:
                images.append(img["url"])

        return ProductResult(
            barcode=clean_barcode,
            title=p.get("title") or p.get("product_name"),
            brand=p.get("brand"),
            description=p.get("description"),
            category=p.get("category"),
            images=images,
            source=f"API: {self.name}",
            raw_data=p,
            verified=True,
        )

