"""Unit tests for ImageExtractor parsing."""

from modules.image_extractor import ImageExtractor


def test_parse_json_ld_images():
    """Verify JSON-LD Product schema image extraction."""
    html = """
    <html>
      <head>
        <script type="application/ld+json">
        {
          "@context": "https://schema.org/",
          "@type": "Product",
          "name": "Men Polo T-Shirt",
          "sku": "26P3P59J3029I",
          "brand": {"@type": "Brand", "name": "Benetton"},
          "image": [
            "https://benetton.in/img/polo_main_large.jpg",
            "https://benetton.in/img/polo_side.jpg"
          ]
        }
        </script>
      </head>
      <body><h1>Men Polo</h1></body>
    </html>
    """
    images, meta, status = ImageExtractor.parse_html_images(html, "https://benetton.in/products/polo")
    assert len(images) == 2
    assert images[0].source_type == "json_ld"
    assert images[0].url == "https://benetton.in/img/polo_main_large.jpg"
    assert meta["sku_found"] == "26P3P59J3029I"


def test_reject_logos_and_icons():
    """Verify non-product images (logos, icons, badges) are rejected."""
    assert ImageExtractor._is_acceptable_image_url("https://example.com/logo.png") is False
    assert ImageExtractor._is_acceptable_image_url("https://example.com/header_banner.jpg") is False
    assert ImageExtractor._is_acceptable_image_url("https://example.com/icon.svg") is False
    assert ImageExtractor._is_acceptable_image_url("https://example.com/cart_icon.png") is False
    assert ImageExtractor._is_acceptable_image_url("https://example.com/products/polo_front_800x1000.jpg") is True
