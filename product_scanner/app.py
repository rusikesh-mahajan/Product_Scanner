"""Product Tag Scanner & Official Product Image Finder - Streamlit Application.

Automates the complete pipeline for Indian apparel tags:
Reads tag via OCR & Barcode -> Identifies product -> Discovers official brand website
-> Locates exact product page -> Verifies match (scoring 0-100) -> Extracts official photo
-> Downloads image with provenance & audit logging.
"""

import html
import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageFile
import streamlit as st

# Allow PIL to load truncated/incomplete JPEG streams from WhatsApp/messengers
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Register HEIF/HEIC support for iPhone images saved with .jpeg extension
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:
    pass

from config.settings import Settings, settings
from modules.barcode_scanner import BarcodeScanner
from modules.barcode_validator import BarcodeValidator
from modules.image_downloader import ImageDownloader
from modules.image_extractor import ImageExtractor
from modules.image_processor import ImageProcessor
from modules.image_search import ImageSearcher
from modules.models import (
    BarcodeResult,
    ExtractedImage,
    ImageCandidate,
    OfficialSite,
    ProductPage,
    ProductRecord,
    TagFields,
    VerificationScore,
)
from modules.ocr_engine import find_best_orientation, get_ocr_engine
from modules.official_site_finder import OfficialSiteFinder
from modules.product_extractor import ProductExtractor
from modules.product_identifier import ProductIdentifier
from modules.product_page_finder import ProductPageFinder
from modules.vision_llm_extractor import VisionLLMExtractor
from services.product_lookup_service import ProductLookupService
from services.product_verifier import ProductVerifier
from utils.validators import validate_barcode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("app")

# -----------------------------------------------------------------------------
# Streamlit Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Product Tag Scanner & Official Image Finder",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-verified {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-review {
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-rejected {
        background-color: #FEE2E2;
        color: #B91C1C;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 700;
        display: inline-block;
    }
    .stage-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-size: 0.92rem;
    }
    .score-card {
        background: #F1F5F9;
        border-radius: 8px;
        padding: 14px;
        border: 1px solid #CBD5E1;
        margin-top: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Cached Resources
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EasyOCR deep learning models...")
def load_easyocr_reader():
    """Cache the EasyOCR reader so model weights are loaded only once."""
    try:
        import easyocr

        base_dir = os.path.dirname(__file__)
        model_dir = os.path.join(base_dir, ".easyocr", "model")
        user_network_dir = os.path.join(base_dir, ".easyocr", "user_network")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(user_network_dir, exist_ok=True)
        kwargs = {
            "gpu": False,
            "model_storage_directory": model_dir,
            "user_network_directory": user_network_dir,
        }
        logger.info("Initializing cached EasyOCR Reader (CPU mode)...")
        return easyocr.Reader(["en"], **kwargs)
    except Exception as e:
        logger.error("Failed to load EasyOCR: %s", e)
        return None


# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "scan_history" not in st.session_state:
    st.session_state["scan_history"] = []

if "active_scan" not in st.session_state:
    st.session_state["active_scan"] = None

if "pipeline_result" not in st.session_state:
    st.session_state["pipeline_result"] = None

if "user_confirmed" not in st.session_state:
    st.session_state["user_confirmed"] = False


# -----------------------------------------------------------------------------
# Pure HTML Table Renderer (Zero pandas / pyarrow dependencies)
# -----------------------------------------------------------------------------
def render_styled_table(rows: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> None:
    """Render a responsive styled HTML table with no DLL or pyarrow dependency."""
    if not rows:
        st.info("No data available.")
        return

    cols = columns or list(rows[0].keys())

    header_cells = "".join(
        f"<th style='padding: 8px 12px; text-align: left; font-weight: 600; background: #1e293b; color: #f8fafc; border-bottom: 2px solid #334155; font-size: 13.5px;'>{html.escape(str(c))}</th>"
        for c in cols
    )

    rows_html = []
    for i, r in enumerate(rows):
        bg = "#0f172a" if i % 2 == 0 else "#1e293b"
        tds = "".join(
            f"<td style='padding: 7px 12px; border-bottom: 1px solid #334155; color: #e2e8f0; font-size: 13px;'>{html.escape(str(r.get(c, '')) if r.get(c) is not None else '')}</td>"
            for c in cols
        )
        rows_html.append(f"<tr style='background: {bg};'>{tds}</tr>")

    table_html = f"""
    <div style='overflow-x: auto; border: 1px solid #334155; border-radius: 6px; margin: 8px 0;'>
      <table style='width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, sans-serif;'>
        <thead><tr>{header_cells}</tr></thead>
        <tbody>{"".join(rows_html)}</tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Image Decoding
# -----------------------------------------------------------------------------
def decode_uploaded_image(file_obj) -> Tuple[Optional[np.ndarray], str]:
    """Robustly decode any uploaded image object using PIL, OpenCV, and imageio."""
    bytes_data = None
    try:
        if hasattr(file_obj, "getvalue"):
            bytes_data = file_obj.getvalue()
        if not bytes_data and hasattr(file_obj, "read"):
            file_obj.seek(0)
            bytes_data = file_obj.read()
    except Exception as e:
        return None, f"Could not read buffer: {e}"

    if not bytes_data:
        return None, "File buffer is empty (0 bytes)."

    errs = []
    # Strategy 1: PIL
    try:
        pil_img = Image.open(io.BytesIO(bytes_data))
        pil_img.load()
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR), ""
    except Exception as e:
        errs.append(f"PIL: {e}")

    # Strategy 2: OpenCV
    try:
        arr = np.frombuffer(bytes_data, np.uint8)
        cv_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if cv_img is not None:
            return cv_img, ""
        errs.append("OpenCV imdecode returned None")
    except Exception as e:
        errs.append(f"OpenCV: {e}")

    # Strategy 3: imageio
    try:
        import imageio.v3 as iio
        raw = iio.imread(io.BytesIO(bytes_data))
        if len(raw.shape) == 2:
            return cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR), ""
        elif raw.shape[2] == 4:
            return cv2.cvtColor(raw, cv2.COLOR_RGBA2BGR), ""
        elif raw.shape[2] == 3:
            return cv2.cvtColor(raw, cv2.COLOR_RGB2BGR), ""
    except Exception as e:
        errs.append(f"imageio: {e}")

    return None, "; ".join(errs)


# -----------------------------------------------------------------------------
# Processing Engine
# -----------------------------------------------------------------------------
def process_tag_image(
    filename: str,
    image: np.ndarray,
    ocr_engine_name: str,
    enable_crop: bool = True,
    enable_multi_rot: bool = True,
) -> Dict[str, Any]:
    """Execute tag processing: blur check, crop, OCR, barcode detection, and field parsing."""
    results: Dict[str, Any] = {"filename": filename, "original_image": image}

    # 1. Blur Detection
    is_blurry, variance = ImageProcessor.detect_blur(image)
    results["is_blurry"] = is_blurry
    results["blur_variance"] = variance

    # 2. Tag Detection & Straightening
    cropped_img = image
    tag_detected = False
    if enable_crop:
        cropped_img, tag_detected = ImageProcessor.detect_and_crop_tag(image)

    results["cropped_image"] = cropped_img
    results["tag_detected"] = tag_detected

    # 3. Tag Field Extraction (Vision LLM or EasyOCR fallback)
    parsed_fields: Optional[TagFields] = None

    if ocr_engine_name == "gemini_vision" and settings.has_gemini_key:
        logger.info("Extracting tag fields via Vision LLM (Gemini 1.5 Flash)...")
        parsed_fields = VisionLLMExtractor.extract(cropped_img, api_key=settings.gemini_api_key)
        if parsed_fields:
            results["ocr_text"] = parsed_fields.raw_ocr_text
            results["ocr_confidence"] = parsed_fields.ocr_confidence
            results["ocr_details"] = []
            results["processed_tag_img"] = cropped_img
            results["rotation_angle"] = 0

    # Fallback to EasyOCR if Gemini Vision not used or returned None
    if parsed_fields is None:
        if ocr_engine_name == "gemini_vision" and not settings.has_gemini_key:
            logger.info("GEMINI_API_KEY not configured; seamlessly using EasyOCR fallback.")
        cached_reader = load_easyocr_reader()
        ocr_engine = get_ocr_engine("easyocr", reader=cached_reader)

        best_img = cropped_img
        best_angle = 0
        if enable_multi_rot:
            best_img, best_angle, _ = find_best_orientation(cropped_img, ocr_engine)

        results["processed_tag_img"] = best_img
        results["rotation_angle"] = best_angle

        ocr_text, avg_conf, ocr_details = ocr_engine.extract_text(best_img)
        results["ocr_text"] = ocr_text
        results["ocr_confidence"] = avg_conf
        results["ocr_details"] = ocr_details
        parsed_fields = ProductExtractor.parse_fields(ocr_text, ocr_confidence=avg_conf)

    # 4. Barcode Detection
    best_scan_img = results.get("processed_tag_img") if results.get("processed_tag_img") is not None else cropped_img
    barcodes = BarcodeScanner.scan(best_scan_img)
    if not barcodes and cropped_img is not image:
        barcodes = BarcodeScanner.scan(image)

    results["barcodes"] = barcodes
    results["tag_fields"] = parsed_fields

    return results


# -----------------------------------------------------------------------------
# Automated Official Search Pipeline Execution
# -----------------------------------------------------------------------------
def execute_official_pipeline(
    record: ProductRecord, progress_callback=None
) -> Dict[str, Any]:
    """Execute end-to-end automated pipeline to find official verified product image."""
    pipeline_data: Dict[str, Any] = {
        "record": record,
        "official_site": None,
        "product_pages": [],
        "best_page": None,
        "extracted_images": [],
        "best_image": None,
        "page_metadata": {},
        "score": None,
        "download_path": None,
        "download_status": "",
        "fallback_candidates": [],
        "status_message": "",
    }

    def update_stage(msg: str):
        if progress_callback:
            progress_callback(msg)
        logger.info("[Pipeline] %s", msg)

    # 1. Discover Official Website
    update_stage("🌐 Discovering brand's official website...")
    official_site = OfficialSiteFinder.find_official_site(record)
    pipeline_data["official_site"] = official_site

    if not official_site:
        pipeline_data["status_message"] = (
            f"Could not discover or verify an official website for brand '{record.brand}'. "
            "Marketplaces (Amazon, Myntra, Flipkart, Ajio) are blocked to ensure authenticity."
        )
        # Search web fallback for closest candidates
        update_stage("🔎 Searching online catalog for closest unverified candidates...")
        tag_fields_mock = TagFields(
            brand=record.brand,
            product_name=record.product_name,
            product_type=record.product_type,
            base_style=record.base_style,
            style_code=record.style_code,
            colour_name=record.colour_name,
            colour_code=record.colour_code,
        )
        pipeline_data["fallback_candidates"] = ImageSearcher.search(tag_fields_mock)
        return pipeline_data

    # 2. Search Product Catalogue on Official Domain
    update_stage(f"🔎 Searching official catalogue on {official_site.domain}...")
    pages = ProductPageFinder.find_product_pages(record, official_site)
    pipeline_data["product_pages"] = pages

    if not pages:
        pipeline_data["status_message"] = (
            f"Official website '{official_site.domain}' is verified, but no product page matching "
            f"style code '{record.base_style or record.style_code}' or product name was found in its index."
        )
        update_stage("🔎 Searching online catalog for closest unverified candidates...")
        tag_fields_mock = TagFields(
            brand=record.brand,
            product_name=record.product_name,
            product_type=record.product_type,
            base_style=record.base_style,
            style_code=record.style_code,
            colour_name=record.colour_name,
            colour_code=record.colour_code,
        )
        pipeline_data["fallback_candidates"] = ImageSearcher.search(tag_fields_mock)
        return pipeline_data

    # 3. Extract Official High-Res Photography (try top candidate pages)
    extracted_images = []
    best_img = None
    page_meta = {}
    chosen_page = None
    extract_status = ""

    for cand_page in pages[:5]:
        update_stage(f"📸 Extracting official photography from {cand_page.url}...")
        imgs, meta, status = ImageExtractor.extract_from_page(
            cand_page.url, record=record
        )
        if imgs:
            chosen_page = cand_page
            extracted_images = imgs
            best_img = imgs[0]
            page_meta = meta
            break
        else:
            extract_status = status
            logger.info("Candidate %s yielded no valid photography (%s); checking next...", cand_page.url, status)

    pipeline_data["best_page"] = chosen_page or pages[0]
    pipeline_data["extracted_images"] = extracted_images
    pipeline_data["page_metadata"] = page_meta
    pipeline_data["best_image"] = best_img

    if not extracted_images:
        pipeline_data["status_message"] = (
            f"Product page(s) found on {official_site.domain}, but automated photography extraction could not complete: {extract_status}."
        )
        update_stage("🔎 Searching online catalog for closest unverified candidates...")
        tag_fields_mock = TagFields(
            brand=record.brand,
            product_name=record.product_name,
            product_type=record.product_type,
            base_style=record.base_style,
            style_code=record.style_code,
            colour_name=record.colour_name,
            colour_code=record.colour_code,
        )
        pipeline_data["fallback_candidates"] = ImageSearcher.search(tag_fields_mock)
        return pipeline_data

    # 4. Verify Product Authenticity & Scoring
    update_stage("⚖️ Verifying product authenticity and calculating confidence score...")
    score = ProductVerifier.verify(
        record=record,
        official_site=official_site,
        page=chosen_page,
        image=best_img,
        page_metadata=page_meta,
    )
    pipeline_data["score"] = score

    # 5. Automated Download if Verified (>= 90)
    if score.status == "verified":
        update_stage("⬇️ Score >= 90: Auto-downloading verified official image...")
        save_path, dl_msg = ImageDownloader.download_product_image(
            image_url=best_img.url,
            record=record,
            page_url=best_page.url,
            domain=official_site.domain,
            verification_score=score,
            label="Verified",
        )
        pipeline_data["download_path"] = save_path
        pipeline_data["download_status"] = dl_msg
    else:
        pipeline_data["download_status"] = "Pending user review"

    return pipeline_data


# -----------------------------------------------------------------------------
# Sidebar: Engine Settings & API Status
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Control Panel")

    st.subheader("API Connections")
    if settings.has_gemini_key:
        st.markdown("🟢 **Gemini 1.5 Flash** *(Vision LLM)*")
    else:
        st.markdown("🟡 **Gemini Vision LLM** *(No Key)*")
        st.caption("Add `GEMINI_API_KEY` in `.env` for Vision LLM extraction.")

    if settings.has_serper_key:
        st.markdown("🟢 **Serper.dev** *(Web & Image Search)*")
    else:
        st.markdown("⚪ **Serper.dev** *(Optional)*")

    if settings.has_brave_key:
        st.markdown("🟢 **Brave Search** *(Web & Image Search)*")
    else:
        st.markdown("🔴 **Brave Search** *(No Key)*")
        st.caption("Provide `BRAVE_API_KEY` or `SERPER_API_KEY` in `.env`.")

    if settings.has_barcode_lookup_key:
        st.markdown("🟢 **Barcode Lookup** *(Active)*")
    else:
        st.markdown("⚪ **Barcode Lookup** *(Optional)*")

    st.divider()

    st.subheader("Extraction Engine")
    engine_options = ["gemini_vision", "easyocr"]
    default_engine_idx = 0 if settings.has_gemini_key else 0
    ocr_choice = st.selectbox(
        "Tag Extraction Engine",
        engine_options,
        index=default_engine_idx,
        format_func=lambda x: "✨ Gemini 1.5 Flash (Vision LLM)" if x == "gemini_vision" else "📷 EasyOCR (Local Deep Learning)",
        help="Gemini 1.5 Flash extracts all apparel tag fields with high contextual precision. EasyOCR is the local fallback engine.",
    )
    deskew_option = st.checkbox("Auto Tag Deskew & Crop", value=True)
    orientation_option = st.checkbox("Multi-Orientation OCR (EasyOCR)", value=True)

    st.divider()
    st.subheader("Scoring Thresholds")
    st.caption(
        f"• Auto-Download: **≥ {settings.score_auto_download_threshold}**\n"
        f"• User Review: **{settings.score_review_threshold}–{settings.score_auto_download_threshold-1}**\n"
        f"• Rejection: **< {settings.score_review_threshold}**"
    )


# -----------------------------------------------------------------------------
# SECTION 1: Header, Upload & Preview
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🏷️ Product Tag Scanner & Official Image Finder</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Reads retail apparel tags, discovers official brand stores, verifies exact product pages, and acquires official high-resolution imagery.</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload product tag photo [JPG, JPEG, PNG]",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
)

active_raw_img: Optional[np.ndarray] = None
active_name: str = ""

if uploaded_files:
    img, err_detail = decode_uploaded_image(uploaded_files)
    if img is not None:
        active_raw_img = img
        active_name = uploaded_files.name
    else:
        st.error(f"Could not decode image: {err_detail}")


# -----------------------------------------------------------------------------
# Scan Trigger & Processing
# -----------------------------------------------------------------------------
if active_raw_img is not None:
    col_u1, col_u2 = st.columns([1, 2])
    with col_u1:
        st.image(cv2.cvtColor(active_raw_img, cv2.COLOR_BGR2RGB), caption=f"Tag Preview: {active_name}", use_container_width=True)

    with col_u2:
        st.markdown("### 🚀 Scan & Discover")
        st.caption("Extract product attributes and automatically find the official product image.")
        btn_scan = st.button("🚀 Scan Tag & Find Official Image", type="primary", use_container_width=True)

    # Detect if image changed
    image_changed = (
        st.session_state["active_scan"] is None
        or st.session_state["active_scan"].get("filename") != active_name
    )

    if btn_scan or image_changed:
        st.session_state["pipeline_result"] = None
        st.session_state["user_confirmed"] = False

        with st.status("Processing product tag...", expanded=True) as status:
            st.write("🔍 Running contour detection, barcode scan & EasyOCR...")
            start_t = time.time()
            scan_data = process_tag_image(
                active_name,
                active_raw_img,
                ocr_choice,
                enable_crop=deskew_option,
                enable_multi_rot=orientation_option,
            )
            scan_data["elapsed"] = time.time() - start_t
            st.session_state["active_scan"] = scan_data

            # 2. Product Identification
            st.write("📦 Consolidating product identification signals...")
            tag_fields = scan_data["tag_fields"]
            barcodes = scan_data["barcodes"]

            # Optional barcode API lookup
            api_res = None
            if barcodes:
                lookup_srv = ProductLookupService()
                api_res = lookup_srv.lookup(barcodes[0].data)

            prod_record = ProductIdentifier.identify(
                tag_fields=tag_fields,
                barcode_results=barcodes,
                api_result=api_res,
            )
            scan_data["prod_record"] = prod_record

            # 3. Official Pipeline Execution
            st.write("🌐 Discovering official website & product page...")
            pipeline_res = execute_official_pipeline(
                prod_record,
                progress_callback=lambda msg: st.write(msg),
            )
            st.session_state["pipeline_result"] = pipeline_res

            status.update(label="Analysis & discovery complete!", state="complete", expanded=False)

    scan = st.session_state.get("active_scan")
    pipe = st.session_state.get("pipeline_result")

    if scan and pipe:
        record: ProductRecord = scan["prod_record"]

        # ---------------------------------------------------------------------
        # SECTION 2: Extracted Information (Editable with provenance tags)
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📋 1. Extracted Tag Information")

        # Notification about Indian barcode lookup
        if record.barcode and record.barcode.startswith("890"):
            st.info(
                "💡 **Notice on 890-prefix barcodes**: Registered in GS1 India; lookup in global databases is optional. "
                "The product attributes below are derived directly from the tag OCR signals."
            )

        with st.expander("✏️ View & Edit Extracted Fields (Click to Amend & Re-Run)", expanded=False):
            col_e1, col_e2, col_e3 = st.columns(3)
            with col_e1:
                edit_brand = st.text_input("Brand", value=record.brand or "")
                edit_name = st.text_input("Product Name", value=record.product_name or "")
                edit_type = st.text_input("Product Type", value=record.product_type or "")
            with col_e2:
                edit_style = st.text_input("Style / Model Code", value=record.base_style or record.style_code or "")
                edit_colour = st.text_input("Colour", value=record.colour_name or record.colour_code or "")
                edit_size = st.text_input("Size", value=record.size or "")
            with col_e3:
                edit_mrp = st.text_input("MRP", value=str(record.mrp) if record.mrp is not None else "")
                edit_email = st.text_input("Customer Care Email", value=record.customer_care_email or "")
                edit_barcode = st.text_input("Barcode", value=record.barcode or "")

            if st.button("🔄 Apply Changes & Re-Run Official Search"):
                record.brand = edit_brand or None
                record.product_name = edit_name or None
                record.product_type = edit_type or None
                record.base_style = edit_style or None
                record.style_code = edit_style or None
                record.colour_name = edit_colour or None
                record.size = edit_size or None
                record.customer_care_email = edit_email or None
                record.barcode = edit_barcode or None
                try:
                    record.mrp = float(edit_mrp) if edit_mrp else None
                except ValueError:
                    pass

                with st.spinner("Re-executing official product discovery with edited fields..."):
                    new_pipe = execute_official_pipeline(record)
                    st.session_state["pipeline_result"] = new_pipe
                    st.rerun()

        # Attribute table display
        attr_rows = [
            {"Attribute": "Brand", "Value": record.brand or "Unknown", "Source Signal": record.brand_source or "OCR"},
            {"Attribute": "Product Name", "Value": record.product_name or record.product_type or "Unknown", "Source Signal": record.product_name_source or "OCR"},
            {"Attribute": "Style Code", "Value": record.base_style or record.style_code or "None", "Source Signal": "OCR"},
            {"Attribute": "Barcode", "Value": record.barcode or "None", "Source Signal": record.barcode_source or "None"},
            {"Attribute": "Colour", "Value": record.colour_name or record.colour_code or "None", "Source Signal": "OCR"},
            {"Attribute": "Size", "Value": record.size or "None", "Source Signal": "OCR"},
            {"Attribute": "MRP", "Value": f"₹{record.mrp:.2f}" if record.mrp else "None", "Source Signal": "OCR"},
            {"Attribute": "Customer Care Email", "Value": record.customer_care_email or "None", "Source Signal": "OCR (Used for domain discovery)"},
        ]
        render_styled_table(attr_rows, columns=["Attribute", "Value", "Source Signal"])

        # ---------------------------------------------------------------------
        # SECTION 3 & 4: Discovery Results & Verification
        # ---------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 🎯 2. Official Product Image Discovery")

        official_site: Optional[OfficialSite] = pipe.get("official_site")
        best_page: Optional[ProductPage] = pipe.get("best_page")
        best_img: Optional[ExtractedImage] = pipe.get("best_image")
        score: Optional[VerificationScore] = pipe.get("score")

        if score and best_img and official_site and best_page:
            col_r1, col_r2 = st.columns([1, 1])

            with col_r1:
                # Display image
                try:
                    st.image(best_img.url, caption=f"Official Image ({best_img.source_type})", use_container_width=True)
                except Exception:
                    st.warning("Could not render image preview. Direct link below.")

            with col_r2:
                # Status Badge
                if score.status == "verified":
                    st.markdown('<span class="badge-verified">✅ Verified Official Image</span>', unsafe_allow_html=True)
                elif score.status == "review":
                    if st.session_state.get("user_confirmed"):
                        st.markdown('<span class="badge-review">👍 User-Confirmed Official Image</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge-review">⚠️ Review Needed (Score 70-89)</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-rejected">❌ Unverified / Rejected</span>', unsafe_allow_html=True)

                st.markdown(f"#### **{record.brand or 'Brand'}** - {record.product_name or record.product_type or 'Item'}")
                st.write(f"**Style Code**: `{record.base_style or record.style_code or 'N/A'}`")
                st.write(f"**Official Website**: [{official_site.domain}]({official_site.homepage_url}) `({official_site.verification_method})`")
                st.write(f"**Product Page**: [View on {official_site.domain}]({best_page.url})")

                # Score Breakdown Card
                st.markdown(
                    f"""
                    <div class="score-card">
                        <div style="font-size: 1.15rem; font-weight: 700; color: #1E293B;">
                            Confidence Score: {score.total_score} / 100
                        </div>
                        <div style="font-size: 0.85rem; color: #475569; margin-top: 4px;">
                            • Brand Match: {score.brand_score}/20<br>
                            • Style/SKU Match: {score.style_score}/50<br>
                            • Product Name/Type Match: {score.name_score}/15<br>
                            • Official Domain & Image: {score.domain_image_score}/15
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Download / Confirmation Actions
                st.markdown("#### Actions")
                if score.status == "verified" or st.session_state.get("user_confirmed"):
                    img_bytes = ImageDownloader.get_image_bytes(best_img.url)
                    if img_bytes:
                        clean_fn = ImageDownloader.generate_filename(record)
                        st.download_button(
                            label="⬇️ Download Official High-Res Photo",
                            data=img_bytes,
                            file_name=clean_fn,
                            mime="image/jpeg",
                            type="primary",
                        )
                    if pipe.get("download_path"):
                        st.success(f"💾 Saved to local downloads: `{pipe['download_path']}`")
                elif score.status == "review":
                    st.warning(
                        "⚠️ Style code was not an exact character-for-character match in the page text (common for retailer SKUs). "
                        "Please verify if the photograph matches your tag."
                    )
                    if st.button("👍 Confirm this is the right product", type="primary"):
                        st.session_state["user_confirmed"] = True
                        # Execute download with User-Confirmed label
                        save_p, msg = ImageDownloader.download_product_image(
                            image_url=best_img.url,
                            record=record,
                            page_url=best_page.url,
                            domain=official_site.domain,
                            verification_score=score,
                            label="User-confirmed",
                        )
                        pipe["download_path"] = save_p
                        st.rerun()

                st.caption(
                    "⚖️ **Copyright & Usage Notice**: Product images are downloaded directly from the official brand's website "
                    "for verification and reference. All trademarks and copyrighted imagery belong to their respective rights holders."
                )

        else:
            # Rejection or Not Found
            st.error("❌ **Official Product Image Could Not Be Verified**")
            msg = pipe.get("status_message") or (score.rejection_reason if score else "Verification score below threshold")
            st.markdown(f"**Reason**: {msg}")

            # Show closest unverified candidates if any exist
            fallbacks: List[ImageCandidate] = pipe.get("fallback_candidates", [])
            if fallbacks:
                st.markdown("#### Closest Catalog Images (Unverified Candidates)")
                st.caption("These images were discovered online but **could not be verified on the official brand website**:")
                f_cols = st.columns(3)
                for f_idx, cand in enumerate(fallbacks[:6]):
                    with f_cols[f_idx % 3]:
                        st.markdown(f"**#{f_idx+1}** `[{cand.domain}]` *(Unverified)*")
                        try:
                            st.image(cand.url, use_container_width=True)
                        except Exception:
                            st.caption("(Image preview unavailable)")
                        if cand.source_page_url:
                            st.caption(f"[Page Link]({cand.source_page_url})")

        # ---------------------------------------------------------------------
        # Original Tag & OCR Visuals Expander
        # ---------------------------------------------------------------------
        st.markdown("---")
        with st.expander("🏷️ View Original Tag Photo & Raw OCR Text", expanded=False):
            v1, v2 = st.columns(2)
            with v1:
                st.markdown("**Original Tag**")
                st.image(cv2.cvtColor(scan["original_image"], cv2.COLOR_BGR2RGB), use_container_width=True)
            with v2:
                st.markdown(f"**Straightened & Cropped Region** *(Rotation: {scan.get('rotation_angle', 0)}°)*")
                st.image(cv2.cvtColor(scan["processed_tag_img"], cv2.COLOR_BGR2RGB), use_container_width=True)

            st.markdown(f"**Raw OCR Text** *(Confidence: {scan.get('ocr_confidence', 0)*100:.1f}%)*")
            st.code(scan.get("ocr_text", ""), language="text")

else:
    st.info("👆 Please upload a tag image above to begin scanning.")
