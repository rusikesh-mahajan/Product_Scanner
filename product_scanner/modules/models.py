"""Shared data models for the Product Tag Scanner & Official Image Finder."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BarcodeResult:
    """Result from barcode scanning."""

    data: str  # The barcode digit string
    type: str  # EAN-13, EAN-8, UPC-A, UPC-E, CODE-128, etc.
    source: str  # "pyzbar", "cv2", "OCR digits", "manual"
    confidence: float = 1.0
    rect: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    is_valid: bool = True
    validation_message: str = ""


@dataclass
class TagFields:
    """Parsed fields extracted from a product tag via OCR."""

    brand: Optional[str] = None
    brand_source: str = ""  # "OCR", "Inferred from tag text", "User-entered"
    product_name: Optional[str] = None
    product_name_source: str = "OCR"
    product_type: Optional[str] = None  # Commodity / Generic Name
    product_type_source: str = "OCR"
    category: Optional[str] = None
    size: Optional[str] = None  # Letter size (M, L) or numeric (30)
    measurements: Dict[str, str] = field(default_factory=dict)
    mrp: Optional[float] = None
    currency: str = "INR"
    colour_code: Optional[str] = None
    colour_name: Optional[str] = None
    theme_code: Optional[str] = None
    style_code: Optional[str] = None
    base_style: Optional[str] = None
    size_suffix: Optional[str] = None
    style_code_variants: List[str] = field(default_factory=list)
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    marketer: Optional[str] = None
    net_quantity: Optional[str] = None
    manufacturing_date: Optional[str] = None
    country_of_origin: Optional[str] = None
    gender: Optional[str] = None
    season: Optional[str] = None
    customer_care_phone: Optional[str] = None
    customer_care_email: Optional[str] = None
    raw_ocr_text: str = ""
    ocr_confidence: float = 0.0
    field_sources: Dict[str, str] = field(default_factory=dict)

    def to_display_dict(self) -> Dict[str, Dict[str, str]]:
        """Convert to a dict for UI display: {label: {value, source}}."""
        rows: List[Tuple[str, Optional[str], str]] = [
            ("Brand", self.brand, self.brand_source or self.field_sources.get("brand", "OCR")),
            ("Product Name", self.product_name, self.product_name_source),
            ("Product Type", self.product_type, self.product_type_source),
            ("Category", self.category, self.field_sources.get("category", "OCR")),
            ("Size", self.size, self.field_sources.get("size", "OCR")),
            ("Style Code", self.style_code, self.field_sources.get("style_code", "OCR")),
            ("Base Style", self.base_style, self.field_sources.get("base_style", "OCR")),
            ("Size Suffix", self.size_suffix, self.field_sources.get("size_suffix", "OCR")),
            (
                "MRP",
                f"{self.mrp:.2f} {self.currency}" if self.mrp is not None else None,
                self.field_sources.get("mrp", "OCR"),
            ),
            ("Colour Code", self.colour_code, self.field_sources.get("colour_code", "OCR")),
            ("Colour Name", self.colour_name, self.field_sources.get("colour_name", "OCR")),
            ("Theme Code", self.theme_code, self.field_sources.get("theme_code", "OCR")),
            ("Model Number", self.model_number, self.field_sources.get("model_number", "OCR")),
            ("Manufacturer", self.manufacturer, self.field_sources.get("manufacturer", "OCR")),
            ("Marketer", self.marketer, self.field_sources.get("marketer", "OCR")),
            ("Net Quantity", self.net_quantity, self.field_sources.get("net_quantity", "OCR")),
            (
                "Manufacturing Date",
                self.manufacturing_date,
                self.field_sources.get("manufacturing_date", "OCR"),
            ),
            (
                "Country of Origin",
                self.country_of_origin,
                self.field_sources.get("country_of_origin", "OCR"),
            ),
            ("Gender", self.gender, self.field_sources.get("gender", "OCR")),
            ("Season", self.season, self.field_sources.get("season", "OCR")),
            (
                "Customer Care Phone",
                self.customer_care_phone,
                self.field_sources.get("customer_care_phone", "OCR"),
            ),
            (
                "Customer Care Email",
                self.customer_care_email,
                self.field_sources.get("customer_care_email", "OCR"),
            ),
        ]
        result: Dict[str, Dict[str, str]] = {}
        for label, value, source in rows:
            if value is not None:
                result[label] = {"value": value, "source": source}
        if self.measurements:
            for key, val in self.measurements.items():
                label = key.replace("_", " ").title()
                result[label] = {
                    "value": val,
                    "source": self.field_sources.get(f"measurement_{key}", "OCR"),
                }
        return result

    def to_flat_dict(self) -> Dict[str, Any]:
        """Flat dict for CSV export."""
        d: Dict[str, Any] = {}
        for label, info in self.to_display_dict().items():
            d[label] = info["value"]
            d[f"{label} (source)"] = info["source"]
        return d


@dataclass
class ProductResult:
    """Result from a product API lookup."""

    barcode: Optional[str] = None
    title: Optional[str] = None
    brand: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    images: List[str] = field(default_factory=list)
    source: str = ""
    offers: List[Dict[str, Any]] = field(default_factory=list)
    raw_data: Dict[str, Any] = field(default_factory=dict)
    verified: bool = False


@dataclass
class ImageCandidate:
    """A candidate product image."""

    url: str
    source: str
    title: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    domain: str = ""
    is_verified: bool = False
    style_code_match: bool = False
    content_type: Optional[str] = None
    file_size: Optional[int] = None
    source_page_url: str = ""


@dataclass
class OfficialSite:
    """Discovered and validated official brand website."""

    domain: str
    brand: str
    homepage_url: str = ""
    verification_method: str = ""  # "email_domain", "web_search_validated", "config_cached"
    is_verified: bool = False
    verification_details: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ProductPage:
    """A product page discovered on the official website."""

    url: str
    domain: str
    title: str = ""
    snippet: str = ""
    style_match: bool = False
    name_match: bool = False
    confidence: float = 0.0
    search_query: str = ""


@dataclass
class ExtractedImage:
    """An image extracted from a product webpage."""

    url: str
    source_type: str = ""  # "json_ld", "og_image", "html_img", "gallery"
    width: Optional[int] = None
    height: Optional[int] = None
    alt_text: str = ""
    is_main: bool = False
    resolution_score: int = 0


@dataclass
class VerificationScore:
    """Scoring breakdown and verdict for a candidate image."""

    brand_score: int = 0          # max 20
    style_score: int = 0          # max 50
    name_score: int = 0           # max 15
    domain_image_score: int = 0   # max 15
    total_score: int = 0          # max 100
    status: str = "rejected"      # "verified" (>=90), "review" (70-89), "rejected" (<70)
    user_confirmed: bool = False
    breakdown: Dict[str, Any] = field(default_factory=dict)
    rejection_reason: str = ""


@dataclass
class ProductRecord:
    """Unified product record merging OCR tag fields, barcode and API data with provenance."""

    brand: Optional[str] = None
    brand_source: str = ""
    product_name: Optional[str] = None
    product_name_source: str = ""
    product_type: Optional[str] = None
    product_type_source: str = ""
    style_code: Optional[str] = None
    base_style: Optional[str] = None
    size_suffix: Optional[str] = None
    style_variants: List[str] = field(default_factory=list)
    barcode: Optional[str] = None
    barcode_source: str = ""
    colour_name: Optional[str] = None
    colour_code: Optional[str] = None
    size: Optional[str] = None
    mrp: Optional[float] = None
    customer_care_email: Optional[str] = None
    customer_care_phone: Optional[str] = None
    manufacturer: Optional[str] = None
    marketer: Optional[str] = None
    gender: Optional[str] = None
    season: Optional[str] = None
    raw_ocr_text: str = ""
    field_provenance: Dict[str, str] = field(default_factory=dict)

    def get_search_tokens(self) -> Dict[str, Any]:
        """Return prioritized search tokens."""
        return {
            "style": self.base_style or self.style_code or "",
            "brand": self.brand or "",
            "product_name": self.product_name or "",
            "product_type": self.product_type or "",
            "colour": self.colour_name or self.colour_code or "",
            "barcode": self.barcode or "",
            "email_domain": (
                self.customer_care_email.split("@")[-1].lower()
                if self.customer_care_email and "@" in self.customer_care_email
                else ""
            ),
        }
