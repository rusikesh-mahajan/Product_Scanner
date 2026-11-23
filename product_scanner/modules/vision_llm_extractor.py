"""Vision LLM Extractor using Google Gemini 1.5 Flash.

Replaces traditional OCR + regex heuristics with multimodal LLM extraction.
Directly reads retail apparel tags and returns structured JSON with Legal Metrology
attributes, brand identification, style numbers, and customer care info.
"""

import base64
import json
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PIL import Image
import requests

from config.settings import settings
from modules.models import TagFields
from modules.product_extractor import ProductExtractor

logger = logging.getLogger(__name__)


class VisionLLMExtractor:
    """Extracts structured apparel tag data directly using Gemini 1.5 Flash multimodal vision."""

    EXTRACTION_PROMPT = """You are an expert retail apparel tag analyzer specializing in Indian retail apparel and packaged goods (Legal Metrology Packaged Commodities standards).
Analyze the provided product tag photograph and extract all structured attributes with extreme accuracy.

Return a JSON object conforming exactly to this schema:
{
  "brand": string or null (e.g. "United Colors of Benetton", "JACK & JONES"),
  "brand_source": "Printed on tag" or "Inferred from marketer/email",
  "product_name": string or null (specific model name if printed, e.g. "JJdi WAVORIA Shirt"),
  "product_type": string or null (commodity or generic name in uppercase, e.g. "JEANS", "POLO T-SHIRT", "SHIRT"),
  "category": string or null (e.g. "Apparel - Bottomwear", "Apparel - Topwear"),
  "style_code": string or null (complete style or model number, e.g. "26P4MBOO50431-30" or "0017801005"),
  "base_style": string or null (the style code with size suffix stripped, e.g. "26P4MBOO50431"),
  "size_suffix": string or null (e.g. "30", "M"),
  "size": string or null (letter or numeric size, e.g. "30", "M", "L"),
  "mrp": float or null (Maximum Retail Price in INR number only, e.g. 3299.00),
  "currency": "INR",
  "colour_code": string or null (e.g. "902", "N22"),
  "colour_name": string or null (e.g. "Blue", "Olive Green"),
  "theme_code": string or null (e.g. "A0J", "C69"),
  "manufacturer": string or null (manufacturer name and location),
  "marketer": string or null (marketer and distributor name),
  "customer_care_email": string or null (clean official customer service email address, e.g. "c.services@benetton.co.in", "help@jackjones.in"),
  "customer_care_phone": string or null (phone or toll free number, e.g. "1800 102 4276", "+91-124-4590433"),
  "country_of_origin": string or null (e.g. "India"),
  "gender": string or null (e.g. "MENS", "WOMENS"),
  "season": string or null (e.g. "SUMMER", "WINTER"),
  "net_quantity": string or null (e.g. "1 N", "1 piece"),
  "manufacturing_date": string or null (e.g. "07/2025"),
  "measurements": {
    "waist": string or null,
    "inseam_length": string or null,
    "length": string or null,
    "chest": string or null
  },
  "raw_text": string (the complete raw text transcribed from the tag)
}

Important Instructions:
1. Brand: If brand logo/name is printed, extract it. If absent, infer from marketer or customer care email (e.g. benetton.co.in -> United Colors of Benetton).
2. Customer Care Email: Fix OCR artifacts where '@' or periods were missed (e.g. 'services benelton CO In' -> 'c.services@benetton.co.in').
3. Style Code: Split base_style and size_suffix accurately.
4. Do NOT hallucinate or guess fields not present on the tag. Set missing fields to null.
"""

    @classmethod
    def _image_to_base64_jpeg(cls, image_data: Any) -> Optional[str]:
        """Convert image (numpy array, PIL Image, or bytes) to base64 JPEG string."""
        try:
            if isinstance(image_data, np.ndarray):
                # OpenCV BGR -> RGB -> JPEG bytes
                if len(image_data.shape) == 3 and image_data.shape[2] == 3:
                    rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                else:
                    rgb = image_data
                pil_img = Image.fromarray(rgb)
            elif isinstance(image_data, Image.Image):
                pil_img = image_data
            elif isinstance(image_data, bytes):
                import io
                pil_img = Image.open(io.BytesIO(image_data))
            else:
                return None

            import io
            buf = io.BytesIO()
            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            pil_img.save(buf, format="JPEG", quality=92)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        except Exception as e:
            logger.error("Failed to encode image to base64: %s", e)
            return None

    MODELS = [
        "gemini-flash-lite-latest",
        "gemini-flash-latest",
        "gemini-2.0-flash",
    ]

    @classmethod
    def extract(
        cls, image: Any, api_key: Optional[str] = None
    ) -> Optional[TagFields]:
        """Call Gemini Vision LLM to extract tag fields directly."""
        key = api_key or settings.gemini_api_key
        if not key:
            logger.debug("No GEMINI_API_KEY provided; Vision LLM extraction skipped.")
            return None

        b64_img = cls._image_to_base64_jpeg(image)
        if not b64_img:
            logger.warning("Could not convert image to base64 for Vision LLM.")
            return None

        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": cls.EXTRACTION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_img,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
            },
        }

        for model_name in cls.MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}"
            try:
                logger.info("Sending tag image to Gemini Vision LLM (%s)...", model_name)
                resp = requests.post(url, headers=headers, json=payload, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if not candidates:
                        logger.warning("Gemini Vision (%s) returned empty candidates.", model_name)
                        continue

                    content_parts = candidates[0].get("content", {}).get("parts", [])
                    if not content_parts:
                        continue

                    raw_json_str = content_parts[0].get("text", "{}")
                    extracted = json.loads(raw_json_str)
                    logger.info("Successfully extracted tag fields using Gemini Vision LLM (%s).", model_name)
                    return cls._build_tag_fields(extracted)

                logger.warning(
                    "Gemini Vision API (%s) returned HTTP %d: %s",
                    model_name,
                    resp.status_code,
                    resp.text[:200],
                )
            except requests.Timeout:
                logger.warning("Gemini Vision API (%s) request timed out (20s).", model_name)
            except Exception as e:
                logger.warning("Gemini Vision API (%s) request failed: %s", model_name, e)

        logger.warning("All Gemini Vision LLM models failed or were unavailable; fallback will trigger.")
        return None

    @classmethod
    def _build_tag_fields(cls, d: Dict[str, Any]) -> TagFields:
        """Convert Gemini JSON response into standard TagFields object."""
        provenance = {}
        source_label = "Vision LLM (Gemini 1.5 Flash)"

        brand = d.get("brand")
        brand_src = d.get("brand_source") or source_label
        if brand:
            provenance["brand"] = brand_src

        pname = d.get("product_name")
        if pname:
            provenance["product_name"] = source_label

        ptype = d.get("product_type")
        if ptype:
            provenance["product_type"] = source_label

        style_code = d.get("style_code")
        base_style = d.get("base_style") or style_code
        size_suffix = d.get("size_suffix")
        if style_code:
            provenance["style_code"] = source_label
        if base_style:
            provenance["base_style"] = source_label

        # Generate style variants for search
        style_variants = ProductExtractor.generate_style_variants(
            base_style or style_code or ""
        )

        size = d.get("size")
        if size:
            provenance["size"] = source_label

        mrp_val = None
        raw_mrp = d.get("mrp")
        if raw_mrp is not None:
            try:
                mrp_val = float(raw_mrp)
                provenance["mrp"] = source_label
            except (ValueError, TypeError):
                pass

        measurements = {}
        raw_meas = d.get("measurements")
        if isinstance(raw_meas, dict):
            for k, v in raw_meas.items():
                if v:
                    measurements[k] = str(v)
                    provenance[f"measurement_{k}"] = source_label

        email = d.get("customer_care_email")
        if email:
            provenance["customer_care_email"] = source_label

        phone = d.get("customer_care_phone")
        if phone:
            provenance["customer_care_phone"] = source_label

        colour_code = d.get("colour_code")
        if colour_code:
            provenance["colour_code"] = source_label

        colour_name = d.get("colour_name")
        if colour_name:
            provenance["colour_name"] = source_label

        theme_code = d.get("theme_code")
        if theme_code:
            provenance["theme_code"] = source_label

        return TagFields(
            brand=brand,
            brand_source=brand_src,
            product_name=pname,
            product_name_source=source_label if pname else "",
            product_type=ptype,
            product_type_source=source_label if ptype else "",
            category=d.get("category"),
            size=size,
            measurements=measurements,
            mrp=mrp_val,
            currency=d.get("currency") or "INR",
            colour_code=colour_code,
            colour_name=colour_name,
            theme_code=theme_code,
            style_code=style_code,
            base_style=base_style,
            size_suffix=size_suffix,
            style_code_variants=style_variants,
            manufacturer=d.get("manufacturer"),
            marketer=d.get("marketer"),
            net_quantity=d.get("net_quantity"),
            manufacturing_date=d.get("manufacturing_date"),
            country_of_origin=d.get("country_of_origin"),
            gender=d.get("gender"),
            season=d.get("season"),
            customer_care_phone=phone,
            customer_care_email=email,
            raw_ocr_text=d.get("raw_text") or "",
            ocr_confidence=0.98,
            field_sources=provenance,
        )
