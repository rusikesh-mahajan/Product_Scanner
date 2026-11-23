"""Product identifier module.

Merges OCR fields from TagFields and detected BarcodeResult into a unified
ProductRecord with per-field source tags. Determines search priority tokens.
"""

import logging
from typing import List, Optional

from modules.models import BarcodeResult, ProductRecord, ProductResult, TagFields

logger = logging.getLogger(__name__)


class ProductIdentifier:
    """Consolidates tag OCR signals and barcode signals into an actionable ProductRecord."""

    @classmethod
    def identify(
        cls,
        tag_fields: TagFields,
        barcode_results: Optional[List[BarcodeResult]] = None,
        api_result: Optional[ProductResult] = None,
    ) -> ProductRecord:
        """Merge OCR fields and barcode into one structured record with per-field source tags."""
        provenance = dict(tag_fields.field_sources)

        # 1. Barcode identification
        primary_barcode: Optional[str] = None
        barcode_source = "None"
        if barcode_results:
            # Pick first valid barcode, or first detected
            for b in barcode_results:
                if b.is_valid:
                    primary_barcode = b.data
                    barcode_source = b.source
                    break
            if not primary_barcode and len(barcode_results) > 0:
                primary_barcode = barcode_results[0].data
                barcode_source = barcode_results[0].source
        if primary_barcode:
            provenance["barcode"] = barcode_source

        # 2. Brand resolution: printed OCR > inferred > API
        brand = tag_fields.brand
        brand_source = tag_fields.brand_source or "OCR"
        if not brand and api_result and api_result.brand:
            brand = api_result.brand
            brand_source = f"API ({api_result.source})"
            provenance["brand"] = brand_source

        # 3. Product Name resolution: OCR > API > Product Type fallback
        product_name = tag_fields.product_name
        product_name_source = tag_fields.product_name_source or "OCR"
        if not product_name and api_result and api_result.title:
            product_name = api_result.title
            product_name_source = f"API ({api_result.source})"
            provenance["product_name"] = product_name_source

        # 4. Product Type / Commodity
        product_type = tag_fields.product_type
        product_type_source = tag_fields.product_type_source or "OCR"

        # 5. Style Code & Variants
        style_code = tag_fields.style_code
        base_style = tag_fields.base_style or style_code
        size_suffix = tag_fields.size_suffix
        style_variants = list(tag_fields.style_code_variants)

        # If base_style has variants, ensure they are present
        if base_style and base_style not in style_variants:
            style_variants.insert(0, base_style)

        # Create the unified record
        record = ProductRecord(
            brand=brand,
            brand_source=brand_source,
            product_name=product_name,
            product_name_source=product_name_source,
            product_type=product_type,
            product_type_source=product_type_source,
            style_code=style_code,
            base_style=base_style,
            size_suffix=size_suffix,
            style_variants=style_variants,
            barcode=primary_barcode,
            barcode_source=barcode_source,
            colour_name=tag_fields.colour_name,
            colour_code=tag_fields.colour_code,
            size=tag_fields.size,
            mrp=tag_fields.mrp,
            customer_care_email=tag_fields.customer_care_email,
            customer_care_phone=tag_fields.customer_care_phone,
            manufacturer=tag_fields.manufacturer,
            marketer=tag_fields.marketer,
            gender=tag_fields.gender,
            season=tag_fields.season,
            raw_ocr_text=tag_fields.raw_ocr_text,
            field_provenance=provenance,
        )

        logger.info(
            "Identified product: Brand='%s' (%s), Style='%s', Name='%s', Barcode='%s'",
            record.brand,
            record.brand_source,
            record.base_style,
            record.product_name,
            record.barcode,
        )
        return record
