"""Application settings loaded from environment variables."""

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    """Centralized application settings for Product Tag Scanner & Official Image Finder."""

    # API Keys (populated from env in __post_init__)
    gemini_api_key: str = ""
    barcode_lookup_api_key: str = ""
    upcitemdb_user_key: str = ""
    serper_api_key: str = ""
    brave_api_key: str = ""
    mock_apis: bool = False

    # Request & Networking settings
    request_timeout: int = 12
    max_retries: int = 3
    retry_base_delay: float = 1.0
    rate_limit_delay_seconds: float = 1.5
    user_agent: str = (
        "ProductTagScanner/1.0 (+https://github.com/product-scanner; retail-research-bot)"
    )

    # Verification scoring weights & thresholds
    score_weight_brand: int = 20
    score_weight_style: int = 50
    score_weight_name: int = 15
    score_weight_domain_image: int = 15

    score_auto_download_threshold: int = 90
    score_review_threshold: int = 70

    # Image settings
    max_image_size_bytes: int = 15 * 1024 * 1024  # 15 MB
    min_image_dimension: int = 300  # min pixels on short side
    supported_upload_types: List[str] = field(
        default_factory=lambda: ["jpg", "jpeg", "png"]
    )
    supported_image_content_types: List[str] = field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp", "image/gif"]
    )

    # Blur detection
    blur_threshold: float = 100.0

    # OCR settings
    default_ocr_engine: str = "easyocr"
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])

    # Path settings
    download_dir: str = os.path.join("downloads", "product_images")
    history_dir: str = "history"
    config_dir: str = "config"

    # Authorized retailer toggle (e.g. Tata CLiQ, Myntra)
    allow_authorized_retailers: bool = False

    def __post_init__(self) -> None:
        """Load values from environment variables if not already provided."""
        if not self.gemini_api_key:
            self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        if not self.barcode_lookup_api_key:
            self.barcode_lookup_api_key = os.getenv("BARCODE_LOOKUP_API_KEY", "")
        if not self.upcitemdb_user_key:
            self.upcitemdb_user_key = os.getenv("UPCITEMDB_USER_KEY", "")
        if not self.serper_api_key:
            self.serper_api_key = os.getenv("SERPER_API_KEY", "")
        if not self.brave_api_key:
            self.brave_api_key = os.getenv("BRAVE_API_KEY", "")
        if not self.mock_apis:
            self.mock_apis = os.getenv("MOCK_APIS", "0") == "1"

    @property
    def has_gemini_key(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_barcode_lookup_key(self) -> bool:
        return bool(self.barcode_lookup_api_key)

    @property
    def has_serper_key(self) -> bool:
        return bool(self.serper_api_key)

    @property
    def has_brave_key(self) -> bool:
        return bool(self.brave_api_key)

    @property
    def has_any_search_key(self) -> bool:
        return self.has_serper_key or self.has_brave_key


# Global settings instance
settings = Settings()
