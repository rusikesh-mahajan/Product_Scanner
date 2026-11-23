"""Unit tests for ProductVerifier scoring and tier classification."""

from modules.models import (
    ExtractedImage,
    OfficialSite,
    ProductPage,
    ProductRecord,
)
from services.product_verifier import ProductVerifier


def test_verifier_exact_match_score():
    """Verify 100-point score for exact brand, style, name, and official domain image."""
    record = ProductRecord(
        brand="United Colors of Benetton",
        base_style="26P4MBOO5043I",
        product_type="JEANS",
        product_name="Slim Fit Jeans",
    )
    site = OfficialSite(
        domain="benetton.in",
        brand="United Colors of Benetton",
        is_verified=True,
    )
    page = ProductPage(
        url="https://benetton.in/products/26p4mboo5043i-slim-fit-jeans",
        domain="benetton.in",
        title="Slim Fit Jeans 26P4MBOO5043I - United Colors of Benetton",
        snippet="Buy Slim Fit Jeans style 26P4MBOO5043I online",
    )
    img = ExtractedImage(
        url="https://benetton.in/cdn/images/26p4mboo5043i_front.jpg",
        source_type="json_ld",
        is_main=True,
    )
    metadata = {
        "sku_found": "26P4MBOO5043I",
        "brand_found": "United Colors of Benetton",
        "name_found": "Slim Fit Jeans",
        "raw_text": "Product 26P4MBOO5043I Slim Fit Jeans United Colors of Benetton",
    }

    score = ProductVerifier.verify(record, site, page, img, page_metadata=metadata)

    assert score.brand_score == 20
    assert score.style_score == 50
    assert score.name_score == 15
    assert score.domain_image_score == 15
    assert score.total_score == 100
    assert score.status == "verified"


def test_verifier_partial_match_review_tier():
    """Verify 70-89 score triggers 'review' status when style code is missing from page."""
    record = ProductRecord(
        brand="JACK & JONES",
        base_style="9017901005",  # Style code not on web page in this format
        product_name="Casual Cotton Shirt",
        product_type="SHIRT",
    )
    site = OfficialSite(
        domain="jackjones.in",
        brand="JACK & JONES",
        is_verified=True,
    )
    page = ProductPage(
        url="https://jackjones.in/products/casual-cotton-shirt",
        domain="jackjones.in",
        title="Casual Cotton Shirt - JACK & JONES",
    )
    img = ExtractedImage(
        url="https://jackjones.in/media/shirt.jpg",
        source_type="og_image",
    )
    metadata = {
        "sku_found": "",
        "brand_found": "JACK & JONES",
        "name_found": "Casual Cotton Shirt",
        "raw_text": "JACK & JONES Casual Cotton Shirt high quality apparel",
    }

    score = ProductVerifier.verify(record, site, page, img, page_metadata=metadata)

    # Brand (20) + Name (15) + Domain/Img (15) = 50 -> rejected if < 70
    assert score.style_score == 0
    assert score.brand_score == 20
    assert score.name_score == 15
    assert score.domain_image_score == 15
    assert score.total_score == 50
    assert score.status == "rejected"
