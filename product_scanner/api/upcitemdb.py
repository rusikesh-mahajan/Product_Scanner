"""UPCitemdb API provider (free trial / optional paid key)."""

import logging
from typing import Any, Dict, List, Optional

from api.base import BaseProvider
from config.settings import settings
from modules.models import ProductResult

logger = logging.getLogger(__name__)


class UPCitemdbProvider(BaseProvider):
    """Provider for UPCitemdb barcode lookups."""

    name: str = "UPCitemdb"

    def lookup(self, barcode: str) -> Optional[ProductResult]:
        """Look up product on UPCitemdb."""
        clean_barcode = barcode.strip().replace(" ", "")

        headers: Dict[str, str] = {}
        if settings.upcitemdb_user_key:
            url = "https://api.upcitemdb.com/prod/v1/lookup"
            headers["user_key"] = settings.upcitemdb_user_key
            headers["key_type"] = "3scale"
        else:
            url = "https://api.upcitemdb.com/prod/trial/lookup"

        params = {"upc": clean_barcode}

        resp = self._make_request(url, headers=headers, params=params)
        if resp is None:
            return None

        try:
            data = resp.json()
        except Exception as e:
            logger.error("Failed to parse UPCitemdb response: %s", e)
            return None

        items = data.get("items", [])
        if not items:
            logger.info("UPCitemdb: no items found for %s", clean_barcode)
            return None

        item = items[0]
        raw_images = item.get("images", [])
        images: List[str] = [img for img in raw_images if isinstance(img, str) and img]

        raw_offers = item.get("offers", [])
        offers: List[Dict[str, Any]] = []
        for o in raw_offers:
            offers.append(
                {
                    "merchant": o.get("merchant"),
                    "domain": o.get("domain"),
                    "title": o.get("title"),
                    "price": o.get("price"),
                    "link": o.get("link"),
                }
            )

        return ProductResult(
            barcode=clean_barcode,
            title=item.get("title"),
            brand=item.get("brand"),
            description=item.get("description"),
            category=item.get("category"),
            images=images,
            source=f"API: {self.name}",
            offers=offers,
            raw_data=item,
            verified=True,
        )

