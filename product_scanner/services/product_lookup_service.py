"""Product lookup service for optional barcode database queries.

Wraps UPCitemdb and Barcode Lookup API providers.
Indian barcodes (890 prefix) are rarely found in global databases, so lookup
is treated as an optional supporting signal.
"""

import logging
from typing import Optional

from api.barcode_lookup import BarcodeLookupProvider
from api.upcitemdb import UPCitemdbProvider
from config.settings import settings
from modules.models import ProductResult

logger = logging.getLogger(__name__)


class ProductLookupService:
    """Manages optional barcode lookup via UPCitemdb and Barcode Lookup."""

    def __init__(self):
        self.providers = [
            UPCitemdbProvider(),
            BarcodeLookupProvider(),
        ]

    def lookup(self, barcode: str) -> Optional[ProductResult]:
        """Execute barcode lookup across available providers. Never raises."""
        if not barcode:
            return None

        clean_barcode = barcode.strip().replace(" ", "")

        # 890-prefix barcodes rarely exist in global open databases
        if clean_barcode.startswith("890"):
            logger.info("Barcode %s has 890 prefix (GS1 India); global API lookup is optional", clean_barcode)

        for provider in self.providers:
            try:
                res = provider.lookup(clean_barcode)
                if res is not None:
                    logger.info("Found product data for %s via %s", clean_barcode, provider.name)
                    return res
            except Exception as e:
                logger.error("Provider %s error: %s", provider.name, e)
                continue

        return None
