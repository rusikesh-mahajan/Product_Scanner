"""Rule-based tag field parser using regex and heuristics for apparel and retail tags."""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from modules.models import TagFields

logger = logging.getLogger(__name__)


class ProductExtractor:
    """Extracts structured fields from raw OCR text extracted from product tags."""

    @classmethod
    def parse_fields(cls, ocr_text: str, ocr_confidence: float = 0.0) -> TagFields:
        """Parse raw OCR text into a structured TagFields object."""
        if not ocr_text:
            return TagFields(raw_ocr_text="", ocr_confidence=0.0)

        # Normalize common OCR noise and line breaks
        text = ocr_text.replace("\r", "")
        field_sources: Dict[str, str] = {}

        # 1. Brand (printed first, then inferred)
        brand, brand_source = cls._parse_brand(text)
        if not brand:
            brand, brand_source = cls._infer_brand(text)
        if brand:
            field_sources["brand"] = brand_source

        # 2. Product Name & Product Type
        product_name = cls._parse_product_name(text)
        if product_name:
            field_sources["product_name"] = "OCR"

        product_type = cls._parse_product_type(text)
        if product_type:
            field_sources["product_type"] = "OCR"

        # 3. Style Code, Base Style, Size Suffix
        style_code, base_style, size_suffix = cls._parse_style_code(text)
        if style_code:
            field_sources["style_code"] = "OCR"
        if base_style:
            field_sources["base_style"] = "OCR"
        if size_suffix:
            field_sources["size_suffix"] = "OCR"

        # Style variants for O<->0 and I<->1 confusion
        style_variants = cls.generate_style_variants(base_style or style_code or "")

        # 4. Size & Measurements
        size = cls._parse_size(text)
        if not size and size_suffix:
            size = size_suffix
        if size:
            field_sources["size"] = "OCR"

        measurements = cls._parse_measurements(text)
        for m_key in measurements:
            field_sources[f"measurement_{m_key}"] = "OCR"

        # 5. Price (MRP)
        mrp = cls._parse_price(text)
        if mrp is not None:
            field_sources["mrp"] = "OCR"

        # 6. Colour & Theme
        colour_code, colour_name = cls._parse_colour(text)
        if colour_code:
            field_sources["colour_code"] = "OCR"
        if colour_name:
            field_sources["colour_name"] = "OCR"

        theme_code = cls._parse_theme(text)
        if theme_code:
            field_sources["theme_code"] = "OCR"

        # 7. Manufacturer & Marketer
        manufacturer = cls._parse_manufacturer(text)
        if manufacturer:
            field_sources["manufacturer"] = "OCR"

        marketer = cls._parse_marketer(text)
        if marketer:
            field_sources["marketer"] = "OCR"

        # 8. Net Quantity & Manufacturing Date
        net_quantity = cls._parse_net_quantity(text)
        if net_quantity:
            field_sources["net_quantity"] = "OCR"

        mfg_date = cls._parse_manufacturing_date(text)
        if mfg_date:
            field_sources["manufacturing_date"] = "OCR"

        # 9. Country of Origin
        country = cls._parse_country_of_origin(text)
        if country:
            field_sources["country_of_origin"] = "OCR"

        # 10. Gender & Season
        gender, season = cls._parse_gender_season(text)
        if gender:
            field_sources["gender"] = "OCR"
        if season:
            field_sources["season"] = "OCR"

        # 11. Customer Care Contact Info
        phone, email = cls._parse_contact_info(text)
        if phone:
            field_sources["customer_care_phone"] = "OCR"
        if email:
            field_sources["customer_care_email"] = "OCR"

        # Category inference
        category = cls._infer_category(product_type, product_name)
        if category:
            field_sources["category"] = "Inferred from tag text"

        return TagFields(
            brand=brand,
            brand_source=brand_source,
            product_name=product_name,
            product_name_source="OCR" if product_name else "",
            product_type=product_type,
            product_type_source="OCR" if product_type else "",
            category=category,
            size=size,
            measurements=measurements,
            mrp=mrp,
            currency="INR",
            colour_code=colour_code,
            colour_name=colour_name,
            theme_code=theme_code,
            style_code=style_code,
            base_style=base_style,
            size_suffix=size_suffix,
            style_code_variants=style_variants,
            manufacturer=manufacturer,
            marketer=marketer,
            net_quantity=net_quantity,
            manufacturing_date=mfg_date,
            country_of_origin=country,
            gender=gender,
            season=season,
            customer_care_phone=phone,
            customer_care_email=email,
            raw_ocr_text=text,
            ocr_confidence=ocr_confidence,
            field_sources=field_sources,
        )

    # -------------------------------------------------------------------------
    # Sub-parsers
    # -------------------------------------------------------------------------

    @staticmethod
    def _parse_brand(text: str) -> Tuple[Optional[str], str]:
        """Look for printed brand names on the tag."""
        # JACK & JONES (matches JACK&JONES, JACK & JONES, JACKSJONES)
        if re.search(r"\bJACK\s*[&S]?\s*JONES\b", text, re.IGNORECASE) or "JACKSJONES" in text.upper():
            return "JACK & JONES", "OCR"

        # UNITED COLORS OF BENETTON
        if re.search(r"\bUNITED\s+COLORS\s+OF\s+BENETTON\b", text, re.IGNORECASE):
            return "United Colors of Benetton", "OCR"

        # Other prominent brands if printed directly as headers
        match = re.search(
            r"\b(LEVI'?S|PUMA|NIKE|ADIDAS|TOMMY\s+HILFIGER|ZARA|H&M|USPA|ALLEN\s+SOLLY|PETER\s+ENGLAND|VAN\s+HEUSEN|ARROW)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(0).strip(), "OCR"

        return None, ""

    @staticmethod
    def _infer_brand(text: str) -> Tuple[Optional[str], str]:
        """Infer brand from email domains, marketer, or manufacturer text."""
        # Check email domain
        email_match = re.search(r"[\w.-]+@([\w.-]+)", text)
        if email_match:
            domain = email_match.group(1).lower()
            if "benetton" in domain or "benelton" in domain:
                return "United Colors of Benetton", "Inferred from tag text"
            if "jackjones" in domain or "jackjone" in domain:
                return "JACK & JONES", "Inferred from tag text"

        # Check marketer/manufacturer text
        if re.search(r"\bBene[lt]{1,2}on\b", text, re.IGNORECASE):
            return "United Colors of Benetton", "Inferred from tag text"
        if re.search(r"\bBestseller\b|\bBest\s+United\b", text, re.IGNORECASE):
            if re.search(r"\bJACK\b", text, re.IGNORECASE) or "JONES" in text.upper():
                return "JACK & JONES", "Inferred from tag text"

        return None, ""

    @staticmethod
    def _parse_product_name(text: str) -> Optional[str]:
        """Match 'Product Name:' or 'Product Name :'."""
        match = re.search(
            r"Produ[ce]t\s+Name\s*[:\.]?\s*([^\n\r]+)", text, re.IGNORECASE
        )
        if match:
            val = match.group(1).strip()
            # Clean up trailing label markers if any
            val = re.split(
                r"(?:Generic\s+Name|Style|Size|Color|MRP|Net\s+Qty)", val, flags=re.IGNORECASE
            )[0].strip()
            return val if val else None
        return None

    @staticmethod
    def _parse_product_type(text: str) -> Optional[str]:
        """Match 'Commodity:' OR 'Generic Name:'."""
        match = re.search(
            r"(?:Commodity|Generic\s+Name)\s*[:\.]?\s*([^\n\r]+)",
            text,
            re.IGNORECASE,
        )
        if match:
            val = match.group(1).strip()
            val = re.split(
                r"(?:Color|Theme|Style|Size|MRP|Net|MENS|WOMENS)", val, flags=re.IGNORECASE
            )[0].strip()
            if val:
                return val

        # Multi-line search after Commodity / Generic Name
        multi_match = re.search(
            r"(?:Commodity|Generic\s+Name)[\s\S]{1,40}?\b(POLO\s+T-SHIRT|T-SHIRT|JEANS|SHIRT|TROUSERS|PANTS|JACKET|SWEATSHIRT)\b",
            text,
            re.IGNORECASE,
        )
        if multi_match:
            return multi_match.group(1).strip().upper()

        return None

    @staticmethod
    def _parse_style_code(text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extract style code, base style, and size suffix."""
        # Case 1: Explicit 'Style No.:'
        match = re.search(r"Style\s*(?:No\.?|No[;j]|Code)?\s*[:\.]?\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if match:
            full = match.group(1).strip()
            if len(full) >= 6:
                if "-" in full:
                    parts = full.rsplit("-", 1)
                    return full, parts[0].strip(), parts[1].strip()
                return full, full, None

        # Case 2: Multi-line Style No (e.g. 'Style Noj\n0017801005' or '9017901005')
        multi_style = re.search(r"Style\s*No[.:;j]?\s*[\n\r]+\s*([0-9A-Z]{8,14})", text, re.IGNORECASE)
        if multi_style:
            code = multi_style.group(1).strip()
            return code, code, None

        # Case 3: Code beneath barcode or at the bottom with hyphen (e.g. '26P4MBOO5043I - 30')
        match = re.search(
            r"\b([A-Z0-9]{8,15})\s*[-–]\s*([A-Z0-9]{1,4})\b", text, re.IGNORECASE
        )
        if match:
            base = match.group(1).strip()
            suffix = match.group(2).strip()
            return f"{base}-{suffix}", base, suffix

        # Case 4: Code followed by size suffix on next line (e.g. '26p3P59J30291\nM' or '26P4MBOO50431\n30')
        newline_match = re.search(
            r"\b([0-9]{2}[A-Za-z0-9]{8,13})\b\s*[\n\r]+\s*([0-9]{2}|XXS|XS|S|M|L|XL|XXL)\b",
            text,
        )
        if newline_match:
            base = newline_match.group(1).strip()
            suffix = newline_match.group(2).strip()
            return f"{base}-{suffix}", base, suffix

        # Case 5: Standalone style-like code (e.g. '26P4MBOO50431')
        match = re.search(r"\b([0-9]{2}[A-Za-z0-9]{8,14})\b", text)
        if match:
            code = match.group(1).strip()
            return code, code, None

        return None, None, None

    @staticmethod
    def _parse_size(text: str) -> Optional[str]:
        """Parse size (e.g. M, L, XL, 30, 32, or '1.06 m  M')."""
        # Look for 'Size: 1.06 m  M' or similar metre + letter size
        match_combo = re.search(
            r"Size\s*[:\.]?\s*(?:\d+(?:\.\d+)?\s*[mM]\s+)?([2-4][0-9]|XXS|XS|S|M|L|XL|XXL|XXXL|FREE)\b",
            text,
            re.IGNORECASE,
        )
        if match_combo:
            return match_combo.group(1).strip().upper()

        # Simple 'Size: ...'
        match = re.search(
            r"\bSize\s*[:\.]?\s*([A-Za-z0-9]+)", text, re.IGNORECASE
        )
        if match:
            val = match.group(1).strip().upper()
            if val not in ("WAIST", "LENGTH", "CHEST"):
                return val

        return None

    @staticmethod
    def _parse_measurements(text: str) -> Dict[str, str]:
        """Parse dimensions like WAIST, INSEAM, LENGTH, CHEST."""
        measurements: Dict[str, str] = {}
        patterns = [
            ("waist", r"\bWAIST\s*[:\.]?\s*(\d+(?:\.\d+)?)\s*(cm|m|in)?\b"),
            ("inseam_length", r"\bINSEAM\s*(?:LENGTH)?\s*[:\.]?\s*(\d+(?:\.\d+)?)\s*(cm|m|in)?\b"),
            ("length", r"\bLENGTH\s*[:\.]?\s*(\d+(?:\.\d+)?)\s*(cm|m|in)?\b"),
            ("chest", r"\bCHEST\s*[:\.]?\s*(\d+(?:\.\d+)?)\s*(cm|m|in)?\b"),
        ]

        for key, pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                val = match.group(1)
                unit = match.group(2) or "cm"
                measurements[key] = f"{val} {unit}"

        # Meter measurement like "1.06 m"
        meter_match = re.search(r"\b(\d+\.\d+)\s*m\b", text, re.IGNORECASE)
        if meter_match and "length" not in measurements:
            measurements["fabric_length"] = f"{meter_match.group(1)} m"

        return measurements

    @staticmethod
    def _parse_price(text: str) -> Optional[float]:
        """Extract MRP price amount in INR with multi-line OCR noise tolerance."""
        # Single-line MRP
        match = re.search(
            r"MRP\s*[:\.]?\s*(?:Rs[-.:]?|INR|₹)?\s*([0-9]+(?:,[0-9]+)*(?:\.[0-9]{1,2})?)",
            text,
            re.IGNORECASE,
        )
        if match:
            val_str = match.group(1).replace(",", "")
            try:
                val = float(val_str)
                if val >= 100:
                    return val
            except ValueError:
                pass

        # Multi-line MRP (e.g. 'MRP\nRs-\n3299')
        multi = re.search(
            r"MRP[\s\S]{1,25}?(?:Rs[-.:]?|INR|₹)[\s\S]{0,10}?([0-9]{3,5}(?:\.[0-9]{1,2})?)",
            text,
            re.IGNORECASE,
        )
        if multi:
            try:
                return float(multi.group(1).replace(",", ""))
            except ValueError:
                pass

        return None

    @staticmethod
    def _parse_colour(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse colour into (colour_code, colour_name)."""
        # Look for 'Color: 902' or 'Color : N22'
        match = re.search(
            r"Colo(?:u)?r\s*[:;.\s]+([^\n\r]+)", text, re.IGNORECASE
        )
        if not match:
            return None, None

        val = match.group(1).strip()
        # Cut off if 'Theme' is on the same line (e.g. 'Color : N22 Theme : C69')
        val = re.split(r"Theme\s*[:\.]?", val, flags=re.IGNORECASE)[0].strip()
        val = val.lstrip(";:. ").rstrip(";:.")

        # Ignore if it accidentally grabbed commodity/type keywords
        if val.upper() in ("JEANS", "SHIRT", "POLO", "T-SHIRT"):
            # Try searching for color code (3-4 alphanumeric like 902 or N22) nearby
            code_match = re.search(r"Colo(?:u)?r[\s\S]{1,30}?\b([0-9]{3}|[A-Z][0-9]{2})\b", text, re.IGNORECASE)
            if code_match:
                return code_match.group(1).upper(), None
            return None, None

        # If short code (e.g. N22, 902, 001) -> colour_code
        if re.match(r"^[A-Z0-9]{2,4}$", val, re.IGNORECASE):
            return val.upper(), None

        # Otherwise it's a colour name (e.g. Cashmere Blue, Olive Green)
        # Strip trailing noise like 'Net Quantity' or 'MRP'
        val = re.split(r"(?:Net\s+Qty|MRP|Size)", val, flags=re.IGNORECASE)[0].strip()
        return None, val if val else None

    @staticmethod
    def _parse_theme(text: str) -> Optional[str]:
        """Parse theme code (e.g. C69, A03)."""
        match = re.search(r"Theme\s*[:\.]?\s*([A-Za-z0-9_-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip().upper()
        return None

    @staticmethod
    def _parse_manufacturer(text: str) -> Optional[str]:
        """Extract manufacturer name and location."""
        match = re.search(
            r"Manufactured\s+by\s*[:\.]?\s*([\s\S]+?)(?=Marketed|Country|Customer|For\s+customer|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            content = match.group(1).strip()
            # Clean up into single or clean lines
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            cleaned = ", ".join(lines[:3])
            return cleaned if cleaned else None
        return None

    @staticmethod
    def _parse_marketer(text: str) -> Optional[str]:
        """Extract marketer and distributor name and address."""
        match = re.search(
            r"Marketed(?:\s*&\s*Distributed)?\s+by\s*[:\.]?\s*([\s\S]+?)(?=For\s+Consumer|Customer|Country|Manufactured|$)",
            text,
            re.IGNORECASE,
        )
        if match:
            content = match.group(1).strip()
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            cleaned = ", ".join(lines[:3])
            return cleaned if cleaned else None
        return None

    @staticmethod
    def _parse_net_quantity(text: str) -> Optional[str]:
        """Extract Net Quantity (e.g. '1 N', '1N', '1 piece')."""
        match = re.search(
            r"Net\s*Quantity\s*[-:\.]?\s*([0-9]+\s*[A-Za-z]+)", text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_manufacturing_date(text: str) -> Optional[str]:
        """Extract manufacturing date (e.g. 'Jul2025', '07/2025', 'July 2025')."""
        match = re.search(
            r"Manufacturing\s*Date\s*[:\.]?\s*([A-Za-z]{3,9}\s*[0-9]{4}|[0-9]{2}/[0-9]{4})",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_country_of_origin(text: str) -> Optional[str]:
        """Extract country of origin."""
        match = re.search(
            r"Country\s+of\s+Origin\s*[:\.]?\s*([A-Za-z]+)", text, re.IGNORECASE
        )
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _parse_gender_season(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse gender and season from markers like 'MENS / SUMMER'."""
        match = re.search(
            r"\b(MEN'?S|WOMEN'?S|BOYS|GIRLS|UNISEX)\s*/\s*(SUMMER|WINTER|SPRING|AUTUMN|FALL|SS\d+|AW\d+)\b",
            text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().upper(), match.group(2).strip().upper()
        return None, None

    @staticmethod
    def _parse_contact_info(text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract customer care phone and email."""
        # Phone: e.g. +91-124-4590433, 1800 102 4276, 1800260005, 18002860005
        phone_match = re.search(
            r"(?:Phone|Toll\s*Free\s*No\.?|Ph\.?\s*No\.?)\s*[:\.]?\s*([+]?[0-9\s-]{8,16})",
            text,
            re.IGNORECASE,
        )
        phone = phone_match.group(1).strip() if phone_match else None

        # Email: standard with @
        email_match = re.search(
            r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", text
        )
        email = email_match.group(1).strip() if email_match else None

        # Fallback: OCR noise where '@' was missed after 'Email:' label (e.g. 'Email:\nservices benelton CO In')
        if not email:
            noisy_email = re.search(
                r"Email\s*[:\.]?\s*[\n\r\s]*([a-zA-Z0-9_.+-]+)[\s@]+(benetton|benelton|jackjones|bestseller)[\s.]+([a-zA-Z0-9_.\s]+)",
                text,
                re.IGNORECASE,
            )
            if noisy_email:
                user_part = noisy_email.group(1).strip()
                domain_part = noisy_email.group(2).strip().lower().replace("benelton", "benetton")
                tld_raw = noisy_email.group(3).strip().lower().split("\n")[0]
                tld_clean = re.sub(r"\s+", ".", tld_raw).strip(".")
                if not tld_clean:
                    tld_clean = "co.in"
                email = f"{user_part}@{domain_part}.{tld_clean}"

        return phone, email

    @staticmethod
    def _infer_category(product_type: Optional[str], product_name: Optional[str]) -> Optional[str]:
        """Infer high-level product category from product type or name."""
        combined = f"{product_type or ''} {product_name or ''}".lower()
        if any(w in combined for w in ["jean", "denim", "trousers", "pants"]):
            return "Apparel - Bottomwear"
        if any(w in combined for w in ["polo", "shirt", "t-shirt", "tee", "top"]):
            return "Apparel - Topwear"
        if any(w in combined for w in ["shoe", "sneaker", "boot", "sandal"]):
            return "Footwear"
        if any(w in combined for w in ["phone", "headphone", "cable", "charger", "laptop"]):
            return "Electronics"
        return "Apparel" if (product_type or product_name) else None

    @staticmethod
    def generate_style_variants(code: str) -> List[str]:
        """Generate OCR confusion variants for style codes (O<->0, I<->1).

        E.g. '26P4MBOO5043I' -> '26P4MB005043I', etc.
        """
        if not code or len(code) < 4:
            return [code] if code else []

        code = code.strip().upper()
        variants: Set[str] = {code}

        # Targeted substitutions for OCR confusion:
        # If 'OO' appears in an alphanumeric code surrounded by letters and digits, try '00'
        if "OO" in code:
            variants.add(code.replace("OO", "00"))
        if "00" in code:
            variants.add(code.replace("00", "OO"))

        # Trailing 'I' vs '1' or 'O' vs '0'
        if code.endswith("I"):
            variants.add(code[:-1] + "1")
        elif code.endswith("1"):
            variants.add(code[:-1] + "I")

        # Full substitution of O and 0
        o_to_zero = code.replace("O", "0")
        zero_to_o = code.replace("0", "O")
        variants.add(o_to_zero)
        variants.add(zero_to_o)

        # Full substitution of I and 1
        i_to_one = code.replace("I", "1")
        one_to_i = code.replace("1", "I")
        variants.add(i_to_one)
        variants.add(one_to_i)

        # Cross combinations
        variants.add(o_to_zero.replace("I", "1"))
        variants.add(code.replace("OO", "00").replace("I", "1"))

        return list(variants)

