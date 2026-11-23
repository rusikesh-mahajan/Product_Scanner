# Product Tag Scanner & Official Product Image Finder

An automated computer vision, deep-learning OCR, and retail intelligence pipeline designed to scan physical apparel tags (specialized for Indian retail brands such as United Colors of Benetton, JACK & JONES, etc.), discover the brand's official store, verify exact product pages, and extract verified official high-resolution catalog photography.

---

## 🌟 Key Capabilities

1. **Automated Pipeline**: Upload a photo of a tag -> OCR reads metadata -> identifies product -> discovers official website -> searches catalogue -> verifies product -> extracts high-resolution image -> downloads to disk.
2. **Robust Computer Vision & OCR**:
   - Blur detection via Laplacian variance.
   - Tag contour detection, straightening, and perspective cropping.
   - 4-orientation evaluation (0°, 90°, 180°, 270°) with EasyOCR cached models.
   - Barcode scanning via pyzbar with OpenCV and digit-OCR fallbacks.
   - Rule-based parser handling commodity/generic name, style codes, base style / size suffixes, MRP in INR, colour codes/names, manufacturer/marketer, customer care email/phone, and O/0 or I/1 OCR confusion variants.
3. **Official Website Discovery (`OfficialSiteFinder`)**:
   - **Strongest Signal**: Customer-care email domain printed on tag (e.g., `c.services@benetton.co.in` -> `benetton.co.in`, `help@jackjones.in` -> `jackjones.in`).
   - Web search discovery using targeted queries (`"<brand> official website India"`).
   - Strict blocklist filtering rejecting marketplaces and aggregators (Amazon, Flipkart, Myntra, Ajio, Nykaa, Meesho, Tata CLiQ, eBay, Alibaba, social media, coupon/review sites).
   - Dynamic validation and persistence in `config/official_domains.json`.
4. **Product Page Discovery (`ProductPageFinder`)**:
   - Targeted `site:<domain> "<base_style>"` queries.
   - robots.txt compliance checking (`urllib.robotparser`).
   - Per-domain rate limiting (1.5s delay).
5. **High-Resolution Image Extraction (`ImageExtractor`)**:
   - JSON-LD `Product` schema extraction (`image`, `sku`, `name`, `brand`).
   - Open Graph (`og:image`) fallback.
   - Responsive `srcset` parser picking the highest-resolution variant.
   - Filtering of logos, banners, icons, and tiny thumbnails (< 300px on short side).
   - Optional Playwright headless Chromium fallback for JavaScript-rendered SPAs.
6. **Configurable Verification Scoring (`ProductVerifier`)**:
   - Brand match: 20 pts
   - Exact style/SKU match on page / URL / JSON-LD: 50 pts
   - Product name/type match: 15 pts
   - Verified official domain & page-referenced image: 15 pts
   - **Tiers**:
     - `90–100`: Auto-download, labelled `Verified official image`.
     - `70–89`: User confirmation required, labelled `User-confirmed`.
     - `< 70`: Rejected with transparent breakdown.

---

## 🏗️ Project Architecture

```
product_scanner/
├── app.py                          # Streamlit application with staged pipeline UI
├── requirements.txt                # Pinned project dependencies
├── .env.example                    # Environment credentials template
├── README.md                       # Comprehensive documentation
├── config/
│   ├── __init__.py
│   ├── settings.py                 # Central settings & scoring thresholds
│   ├── blocked_domains.json        # Marketplace/aggregator blocklist
│   └── official_domains.json       # Dynamically cached verified official brand domains
├── modules/
│   ├── __init__.py
│   ├── barcode_scanner.py          # pyzbar + OpenCV + OCR digits fallback
│   ├── barcode_validator.py        # Check digit validator & suggestions
│   ├── image_downloader.py         # Streaming download, Pillow verify, SHA-256 dedup
│   ├── image_extractor.py          # JSON-LD, OG, and HTML product photography extractor
│   ├── image_processor.py          # Blur check, tag detection & perspective crop
│   ├── image_search.py             # Image search query builder & ranking
│   ├── models.py                   # Shared dataclasses
│   ├── ocr_engine.py               # Cached EasyOCR engine & multi-orientation
│   ├── official_site_finder.py     # Official website discovery & domain validation
│   ├── product_extractor.py        # Rule-based tag text parser (core IP)
│   ├── product_identifier.py       # TagFields + Barcode merger with provenance
│   └── product_page_finder.py      # site:<domain> query generator & page finder
├── services/
│   ├── __init__.py
│   ├── search_service.py           # Serper & Brave search wrapper with rate limiting & robots.txt
│   ├── product_lookup_service.py   # Optional barcode database lookups (UPCitemdb, BarcodeLookup)
│   └── product_verifier.py         # 100-point rubric verification engine
├── downloads/
│   └── product_images/             # Downloaded official product photography (<brand>_<style>_<colour>.<ext>)
├── history/
│   └── downloads_history.jsonl     # Full audit trail of verified downloads
├── tests/
│   ├── conftest.py
│   ├── test_barcode_validator.py
│   ├── test_e2e_pipeline.py        # End-to-end integration tests
│   ├── test_image_extractor.py
│   ├── test_image_validation.py
│   ├── test_integration.py
│   ├── test_official_site_finder.py
│   ├── test_product_extractor.py
│   ├── test_providers.py
│   └── test_verifier.py
└── assets/
    └── sample_tags/                # Sample tag photos (Benetton jeans, polo, Jack & Jones shirt)
```

---

## 🚀 Installation & Setup

### 1. Prerequisites
- Python 3.11 (Windows 11 recommended).
- Virtual environment in `venv/`.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: For JavaScript-rendered SPA fallback)*:
```bash
playwright install chromium
```

### 3. Configure API Keys
Copy `.env.example` to `.env` and provide your search API key:
```ini
# Serper.dev (https://serper.dev/)
SERPER_API_KEY=

# Brave Search API (https://brave.com/search/api/)
BRAVE_API_KEY=your_brave_api_key_here
```

### 4. Run the Streamlit Application
On Windows with Application Control (WDAC), run via python module:
```bash
python -m streamlit run app.py
```
Or execute the included batch file:
```cmd
run.bat
```

---

## 🧪 Running Tests

Execute the fast unit test suite:
```bash
python -m pytest -v -m "not slow"
```

Execute all tests including end-to-end CV and OCR tests on real sample tags:
```bash
python -m pytest -v
```

---

## ⚖️ Verification Scoring System

Every candidate image discovered on an official webpage is evaluated against a 100-point rubric:

| Evaluation Factor | Maximum Points | Verification Method |
|:---|:---:|:---|
| **Brand Match** | 20 | Brand verified on official domain & page content / JSON-LD |
| **Exact Style / SKU Match** | 50 | Exact style code or OCR confusion variant found in JSON-LD SKU, page text, or URL |
| **Product Name / Type Match** | 15 | Commodity / generic name / product name confirmed on page |
| **Verified Official Domain & Image** | 15 | Image is referenced by a verified official brand domain |
| **Total Score** | **100** | |

### Confidence Tiers
- **Score 90–100 (`Verified`)**: Automatically downloaded to `downloads/product_images/` with filename `<brand>_<style>_<colour>.<ext>`.
- **Score 70–89 (`Review`)**: Image is displayed with a score breakdown and a **"Confirm this is the right product"** button. If confirmed, saved and labelled `User-confirmed`.
- **Score < 70 (`Rejected`)**: Never downloaded. The app reports why it was rejected and displays closest unverified candidates.

---

## ⚠️ Real-World Limitations & Disclaimers

1. **Indian 890 Barcode Reality**: Barcodes with prefix `890` are registered with GS1 India and rarely exist in global open databases (Open Food Facts, trial UPCitemdb). Failure to find an 890-prefix barcode in global APIs is normal and expected.
2. **Style Code Mismatches**: Retail apparel tags often print internal factory style numbers (e.g. `9017901005`) which may differ from customer-facing web store SKUs. The scoring engine evaluates this honestly without inflating confidence scores.
3. **Anti-Bot & JavaScript Frontends**: Brand websites returning HTTP 403 or Cloudflare challenges are flagged as "Site blocks automated access" and skipped without attempting fragile circumvention.
4. **Copyright Notice**: Product photographs are fetched directly from brand web indices for identification and verification purposes only. Antigravity does not host or claim ownership of third-party copyrighted photography.
