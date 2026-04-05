"""
app.py — Streamlit web app for Invoice OCR
Run: streamlit run app.py
"""

import sys
import os
import time
import json
import io
import warnings
from pathlib import Path

# ── Suppress warnings before heavy imports ────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
warnings.filterwarnings("ignore")

import streamlit as st
from PIL import Image
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

# ── Page config — must be FIRST streamlit call ────────────────────────────
st.set_page_config(
    page_title="InvoiceAI · Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# Custom CSS — refined dark financial aesthetic
# =============================================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

[data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
}

[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* ── Hide default streamlit padding ── */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1117; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }

/* ── Upload zone override ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: rgba(22, 27, 34, 0.8) !important;
    border: 1.5px dashed #30363d !important;
    border-radius: 12px !important;
    transition: all 0.25s ease !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #d4a853 !important;
    background: rgba(212, 168, 83, 0.04) !important;
}

[data-testid="stFileUploaderDropzone"] * {
    color: #7d8590 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #d4a853, #c49440) !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.02em !important;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(212, 168, 83, 0.3) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] * { color: #d4a853 !important; }

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid #21262d !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    background: rgba(22, 27, 34, 0.9) !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    color: #e6edf3 !important;
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background: rgba(22, 27, 34, 0.8) !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] * {
    color: #7d8590 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

[data-testid="stMetricValue"] * {
    color: #e6edf3 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 1.3rem !important;
}

[data-testid="stMetricDelta"] * { color: #3fb950 !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: 1px solid #21262d !important;
    gap: 0 !important;
}

[data-testid="stTabs"] [role="tab"] {
    color: #7d8590 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.6rem 1.2rem !important;
    background: transparent !important;
    transition: all 0.2s !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #d4a853 !important;
    border-bottom-color: #d4a853 !important;
}

/* ── Code / JSON ── */
[data-testid="stCodeBlock"] {
    background: #161b22 !important;
    border: 1px solid #21262d !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# Header component
# =============================================================================

def render_header():
    st.markdown("""
    <div style="
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-bottom: 1px solid #21262d;
        padding: 2rem 3rem 1.5rem;
        margin-bottom: 0;
    ">
        <div style="max-width: 1200px; margin: 0 auto;">
            <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 0.4rem;">
                <div style="
                    width: 36px; height: 36px;
                    background: linear-gradient(135deg, #d4a853, #c49440);
                    border-radius: 8px;
                    display: flex; align-items: center; justify-content: center;
                    font-size: 18px;
                ">📄</div>
                <span style="
                    font-family: 'Playfair Display', serif;
                    font-size: 1.5rem;
                    font-weight: 600;
                    color: #e6edf3;
                    letter-spacing: -0.01em;
                ">InvoiceAI</span>
                <span style="
                    background: rgba(212,168,83,0.15);
                    color: #d4a853;
                    font-size: 0.65rem;
                    font-weight: 600;
                    letter-spacing: 0.12em;
                    padding: 3px 8px;
                    border-radius: 4px;
                    border: 1px solid rgba(212,168,83,0.3);
                    text-transform: uppercase;
                    margin-left: 4px;
                ">BETA</span>
            </div>
            <p style="
                color: #7d8590;
                font-size: 0.85rem;
                font-weight: 400;
                margin: 0;
                padding-left: 50px;
            ">Intelligent field extraction from tractor loan quotations · Powered by Qwen2-VL</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# Field card components
# =============================================================================

def field_card(label: str, value, icon: str, mono: bool = False, highlight: bool = False):
    val_str = str(value) if value is not None else "—"
    val_color = "#e6edf3" if value is not None else "#484f58"
    border = "1px solid rgba(212,168,83,0.4)" if highlight else "1px solid #21262d"
    font = "'DM Mono', monospace" if mono else "'DM Sans', sans-serif"
    font_size = "1.05rem" if mono else "1.1rem"

    st.markdown(f"""
    <div style="
        background: #161b22;
        border: {border};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        height: 100%;
        transition: border-color 0.2s;
    ">
        <div style="
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 0.5rem;
        ">
            <span style="font-size: 0.9rem;">{icon}</span>
            <span style="
                color: #7d8590;
                font-size: 0.72rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            ">{label}</span>
        </div>
        <div style="
            color: {val_color};
            font-family: {font};
            font-size: {font_size};
            font-weight: 500;
            word-break: break-word;
            line-height: 1.3;
        ">{val_str}</div>
    </div>
    """, unsafe_allow_html=True)


def detection_badge(label: str, present: bool, bbox=None, icon: str = ""):
    if present:
        bg = "rgba(63, 185, 80, 0.12)"
        border = "1px solid rgba(63, 185, 80, 0.35)"
        dot_color = "#3fb950"
        status_text = "Detected"
        status_color = "#3fb950"
    else:
        bg = "rgba(248, 81, 73, 0.08)"
        border = "1px solid rgba(248, 81, 73, 0.25)"
        dot_color = "#f85149"
        status_text = "Not found"
        status_color = "#f85149"

    bbox_html = ""
    if present and bbox:
        bbox_html = f"""
        <div style="
            margin-top: 0.5rem;
            font-family: 'DM Mono', monospace;
            font-size: 0.7rem;
            color: #7d8590;
            background: rgba(0,0,0,0.2);
            padding: 4px 8px;
            border-radius: 5px;
            letter-spacing: 0.03em;
        ">bbox [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]</div>
        """

    st.markdown(f"""
    <div style="
        background: {bg};
        border: {border};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        height: 100%;
    ">
        <div style="
            display: flex; align-items: center; gap: 8px;
            margin-bottom: 0.4rem;
        ">
            <span style="font-size: 0.9rem;">{icon}</span>
            <span style="
                color: #7d8590;
                font-size: 0.72rem;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.1em;
            ">{label}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 8px; margin-top: 0.3rem;">
            <div style="
                width: 8px; height: 8px;
                background: {dot_color};
                border-radius: 50%;
                box-shadow: 0 0 6px {dot_color};
            "></div>
            <span style="
                color: {status_color};
                font-weight: 600;
                font-size: 1.0rem;
            ">{status_text}</span>
        </div>
        {bbox_html}
    </div>
    """, unsafe_allow_html=True)


def confidence_bar(score: float):
    pct = int(score * 100)
    if pct >= 85:
        color = "#3fb950"
        label = "High"
    elif pct >= 65:
        color = "#d4a853"
        label = "Medium"
    else:
        color = "#f85149"
        label = "Low"

    st.markdown(f"""
    <div style="
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1rem;
    ">
        <div style="
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 0.6rem;
        ">
            <span style="
                color: #7d8590;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 500;
            ">Overall Confidence</span>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="
                    color: {color};
                    font-size: 0.72rem;
                    font-weight: 600;
                    letter-spacing: 0.06em;
                    text-transform: uppercase;
                ">{label}</span>
                <span style="
                    color: {color};
                    font-family: 'DM Mono', monospace;
                    font-size: 1.4rem;
                    font-weight: 500;
                ">{pct}%</span>
            </div>
        </div>
        <div style="
            height: 5px;
            background: #21262d;
            border-radius: 3px;
            overflow: hidden;
        ">
            <div style="
                height: 100%;
                width: {pct}%;
                background: linear-gradient(90deg, {color}88, {color});
                border-radius: 3px;
                transition: width 0.8s ease;
            "></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def stat_pill(label: str, value: str, icon: str = ""):
    st.markdown(f"""
    <div style="
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    ">
        <span style="font-size: 0.75rem;">{icon}</span>
        <span style="color: #7d8590; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;">{label}</span>
        <span style="color: #e6edf3; font-family: 'DM Mono', monospace; font-size: 0.82rem; font-weight: 500;">{value}</span>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, subtitle: str = ""):
    sub_html = f'<span style="color:#7d8590; font-size:0.8rem; font-weight:400; margin-left:0.5rem;">{subtitle}</span>' if subtitle else ""
    st.markdown(f"""
    <div style="
        display: flex;
        align-items: baseline;
        gap: 0;
        margin: 1.5rem 0 0.8rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #21262d;
    ">
        <span style="
            color: #e6edf3;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        ">{title}</span>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown('<div style="height:1px; background:#21262d; margin: 1rem 0;"></div>', unsafe_allow_html=True)


# =============================================================================
# Pipeline runner (cached model, fresh inference per call)
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_pipeline_components():
    """Load heavy components once and cache them."""
    import yaml
    from ingestion.preprocessor import PreprocessConfig
    from vlm.qwen_extractor import QwenExtractor, QwenExtractorConfig
    from detection.stamp_detector import StampDetectorConfig
    from detection.signature_detector import SignatureDetectorConfig
    from matching.master_loader import MasterData
    from utils.device_utils import resolve_device, resolve_dtype

    cfg_path = ROOT / "configs" / "config.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}

    def _get(d, *keys, default=None):
        node = d
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node

    device = resolve_device(_get(cfg, "vlm", "device", default="auto"))
    dtype  = resolve_dtype(_get(cfg, "vlm", "dtype", default="auto"), device)

    vlm_cfg = QwenExtractorConfig(
        model_id=_get(cfg, "vlm", "model_id", default="Qwen/Qwen2-VL-2B-Instruct"),
        device=device, dtype=dtype,
        max_new_tokens=_get(cfg, "vlm", "max_new_tokens", default=256),
        run_verification=False,
        fallback_on_failure=True,
    )

    return {
        "vlm":        QwenExtractor(vlm_cfg),
        "pre_cfg":    PreprocessConfig(),
        "stamp_cfg":  StampDetectorConfig(),
        "sig_cfg":    SignatureDetectorConfig(),
        "master":     MasterData(master_dir=ROOT / "master_data"),
    }


def run_pipeline(pil_image: Image.Image, components: dict) -> dict:
    """Run the full pipeline on one PIL image."""
    from ingestion.preprocessor import preprocess
    from detection.stamp_detector import detect_stamp
    from detection.signature_detector import detect_signature
    from extraction.field_parser import parse_fields
    from extraction.consensus import build_consensus
    from extraction.confidence import compute_confidence
    from matching.fuzzy_matcher import match_all_fields
    from utils.image_utils import resize_for_model
    from utils.device_utils import ocr_should_run

    t0 = time.time()

    # 1. Preprocess
    pre = preprocess(pil_image, components["pre_cfg"])
    rgb = pre.rgb_clean

    # 2. OCR (platform aware)
    ocr_result = None
    if ocr_should_run("auto"):
        try:
            from ocr.paddle_ocr import run_ocr_multi_lang
            ocr_result = run_ocr_multi_lang(pre.gray_enhanced)
        except Exception:
            ocr_result = None

    # 3. VLM extraction
    vlm_img = resize_for_model(rgb, max_dim=1024)
    vlm_result = components["vlm"].extract(vlm_img)

    # 4. Detection on original image (not CLAHE-altered)
    stamp_r = detect_stamp(pil_image, components["stamp_cfg"])
    sig_r   = detect_signature(pil_image, ocr_result, components["sig_cfg"])

    # 5. Parse OCR fields
    parsed = parse_fields(ocr_result) if ocr_result else None

    # 6. Consensus
    ocr_fields = parsed.to_dict() if parsed else {}
    ocr_conf   = {
        "dealer_name": parsed.dealer_conf if parsed else 0.0,
        "model_name":  parsed.model_conf  if parsed else 0.0,
        "horse_power": parsed.hp_conf     if parsed else 0.0,
        "asset_cost":  parsed.cost_conf   if parsed else 0.0,
    }
    vlm_fields = vlm_result.to_dict()
    vlm_conf   = {k: 0.75 for k in vlm_fields}
    det_fields = {
        "stamp_present":     stamp_r.present,
        "signature_present": sig_r.present,
    }
    det_conf = {
        "stamp_present":     stamp_r.confidence,
        "signature_present": sig_r.confidence,
    }

    consensus = build_consensus(
        ocr_fields=ocr_fields, ocr_confidences=ocr_conf,
        vlm_fields=vlm_fields, vlm_confidences=vlm_conf,
        detection_fields=det_fields, detection_confidences=det_conf,
    )

    # 7. Fuzzy match
    try:
        matches = match_all_fields(
            dealer_name=consensus.dealer_name.value,
            model_name=consensus.model_name.value,
            master_data=components["master"],
        )
        final_dealer = (matches["dealer"].matched
                        if matches["dealer"].matched_above_threshold
                        else consensus.dealer_name.value)
        final_model  = (matches["model"].matched
                        if matches["model"].matched_above_threshold
                        else consensus.model_name.value)
    except Exception:
        final_dealer = consensus.dealer_name.value
        final_model  = consensus.model_name.value

    # 8. Confidence
    try:
        conf_report  = compute_confidence(consensus)
        overall_conf = conf_report.overall
    except Exception:
        overall_conf = consensus.overall_confidence

    elapsed = time.time() - t0

    return {
        "dealer_name":       final_dealer,
        "model_name":        final_model,
        "horse_power":       consensus.horse_power.value,
        "asset_cost":        consensus.asset_cost.value,
        "signature_present": consensus.signature_present.value,
        "signature_bbox":    sig_r.bbox if sig_r.present else None,
        "stamp_present":     consensus.stamp_present.value,
        "stamp_bbox":        stamp_r.bbox if stamp_r.present else None,
        "confidence":        overall_conf,
        "processing_time":   elapsed,
        "rotation_applied":  pre.rotation_applied,
        "skew_applied":      pre.skew_applied,
        "vlm_fields_found":  vlm_result.fields_found,
        "ocr_enabled":       ocr_result is not None,
    }


# =============================================================================
# Main app layout
# =============================================================================

def format_currency(value) -> str:
    if value is None:
        return "—"
    try:
        v = int(value)
        if v >= 10_00_000:
            return f"₹ {v/10_00_000:.2f}L  ({v:,})"
        elif v >= 1_000:
            return f"₹ {v:,}"
        return f"₹ {v}"
    except Exception:
        return str(value)


def main():
    render_header()

    # ── Main content padding ──────────────────────────────────────────────
    st.markdown('<div style="max-width:1200px; margin:0 auto; padding: 2rem 3rem;">', unsafe_allow_html=True)

    # ── Two-column layout ─────────────────────────────────────────────────
    left, right = st.columns([1, 1.3], gap="large")

    with left:
        st.markdown("""
        <div style="margin-bottom: 1.2rem;">
            <h2 style="
                font-family: 'Playfair Display', serif;
                font-size: 1.6rem;
                color: #e6edf3;
                font-weight: 600;
                line-height: 1.3;
                margin-bottom: 0.4rem;
            ">Upload Invoice</h2>
            <p style="color:#7d8590; font-size:0.85rem; line-height:1.6;">
                Supports PDF, PNG, JPG, JPEG, TIFF.<br>
                Handles scanned, handwritten, and multilingual documents.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            label="Drop your invoice here",
            type=["png", "jpg", "jpeg", "tiff", "tif", "pdf"],
            label_visibility="collapsed",
        )

        if uploaded:
            # Show preview
            if uploaded.type == "application/pdf":
                try:
                    from ingestion.pdf_converter import convert_pdf_to_images
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    pages = convert_pdf_to_images(tmp_path, dpi=150)
                    preview_img = pages[0]
                    os.unlink(tmp_path)
                except Exception as e:
                    st.error(f"Could not preview PDF: {e}")
                    preview_img = None
            else:
                preview_img = Image.open(uploaded).convert("RGB")

            if preview_img:
                st.markdown('<div style="margin: 1rem 0 0.5rem;">', unsafe_allow_html=True)
                st.image(preview_img, use_container_width=True, caption="")
                st.markdown('</div>', unsafe_allow_html=True)

                # File info pills
                st.markdown(f"""
                <div style="margin-top:0.5rem;">
                    <span style="
                        background:#161b22; border:1px solid #21262d;
                        border-radius:20px; padding:3px 10px;
                        color:#7d8590; font-size:0.72rem;
                        letter-spacing:0.05em; margin-right:6px;
                    ">{uploaded.name}</span>
                    <span style="
                        background:#161b22; border:1px solid #21262d;
                        border-radius:20px; padding:3px 10px;
                        color:#7d8590; font-size:0.72rem;
                        letter-spacing:0.05em;
                    ">{uploaded.size / 1024:.1f} KB</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

        run_btn = st.button(
            "⚡  Extract Fields",
            use_container_width=True,
            disabled=(uploaded is None),
        )

        # Tips section
        st.markdown("""
        <div style="
            margin-top: 1.5rem;
            background: rgba(22,27,34,0.6);
            border: 1px solid #21262d;
            border-radius: 10px;
            padding: 1rem 1.2rem;
        ">
            <p style="
                color: #7d8590;
                font-size: 0.72rem;
                text-transform: uppercase;
                letter-spacing: 0.1em;
                font-weight: 500;
                margin-bottom: 0.7rem;
            ">What gets extracted</p>
            <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;">
        """ + "".join([
            f'<div style="color:#adbac7; font-size:0.82rem; display:flex; align-items:center; gap:6px;">'
            f'<span style="color:#d4a853;">{icon}</span> {text}</div>'
            for icon, text in [
                ("🏪", "Dealer name"),
                ("🚜", "Model name"),
                ("⚙️", "Horse power"),
                ("💰", "Asset cost"),
                ("✍️", "Signature"),
                ("🔵", "Stamp"),
            ]
        ]) + """
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Right panel — results ─────────────────────────────────────────────
    with right:
        if not uploaded:
            # Empty state
            st.markdown("""
            <div style="
                height: 500px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border: 1.5px dashed #21262d;
                border-radius: 14px;
                background: rgba(22,27,34,0.4);
            ">
                <div style="
                    width: 56px; height: 56px;
                    background: rgba(212,168,83,0.1);
                    border: 1px solid rgba(212,168,83,0.2);
                    border-radius: 14px;
                    display: flex; align-items: center;
                    justify-content: center;
                    font-size: 24px;
                    margin-bottom: 1rem;
                ">📋</div>
                <p style="color:#484f58; font-size:0.9rem; text-align:center; line-height:1.6;">
                    Upload an invoice to see<br>extracted fields here
                </p>
            </div>
            """, unsafe_allow_html=True)

        elif run_btn or st.session_state.get("last_result"):

            if run_btn:
                # Fresh extraction
                with st.spinner("Loading model and extracting fields…"):
                    try:
                        components = load_pipeline_components()

                        # Re-open image from uploaded bytes
                        uploaded.seek(0)
                        if uploaded.type == "application/pdf":
                            from ingestion.pdf_converter import convert_pdf_to_images
                            import tempfile
                            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                tmp.write(uploaded.read())
                                tmp_path = tmp.name
                            pages = convert_pdf_to_images(tmp_path, dpi=200)
                            pil_img = pages[0]
                            os.unlink(tmp_path)
                        else:
                            pil_img = Image.open(uploaded).convert("RGB")

                        result = run_pipeline(pil_img, components)
                        st.session_state["last_result"] = result
                        st.session_state["last_filename"] = uploaded.name

                    except Exception as e:
                        st.error(f"Extraction failed: {e}")
                        st.session_state.pop("last_result", None)
                        result = None
            else:
                result = st.session_state.get("last_result")

            if result:
                # ── Confidence bar ────────────────────────────────────────
                confidence_bar(result["confidence"])

                # ── Processing stats ──────────────────────────────────────
                st.markdown(f"""
                <div style="display:flex; flex-wrap:wrap; gap:0; margin-bottom:0.3rem;">
                    <span style="
                        display:inline-flex; align-items:center; gap:5px;
                        background:#161b22; border:1px solid #21262d;
                        border-radius:20px; padding:3px 10px; margin:0 5px 5px 0;
                        color:#7d8590; font-size:0.72rem; letter-spacing:0.05em;
                    ">⏱ {result['processing_time']:.1f}s</span>
                    <span style="
                        display:inline-flex; align-items:center; gap:5px;
                        background:#161b22; border:1px solid #21262d;
                        border-radius:20px; padding:3px 10px; margin:0 5px 5px 0;
                        color:#7d8590; font-size:0.72rem; letter-spacing:0.05em;
                    ">🧠 {result['vlm_fields_found']}/6 VLM fields</span>
                    <span style="
                        display:inline-flex; align-items:center; gap:5px;
                        background:#161b22; border:1px solid #21262d;
                        border-radius:20px; padding:3px 10px; margin:0 5px 5px 0;
                        color:#7d8590; font-size:0.72rem; letter-spacing:0.05em;
                    ">🔄 {result['rotation_applied']}° corrected</span>
                    <span style="
                        display:inline-flex; align-items:center; gap:5px;
                        background:rgba(63,185,80,0.1); border:1px solid rgba(63,185,80,0.25);
                        border-radius:20px; padding:3px 10px; margin:0 5px 5px 0;
                        color:#3fb950; font-size:0.72rem; letter-spacing:0.05em;
                    ">💰 $0.00 cost</span>
                </div>
                """, unsafe_allow_html=True)

                # ── Tabs: Fields / Raw JSON ───────────────────────────────
                tab_fields, tab_json = st.tabs(["  Extracted Fields  ", "  Raw JSON  "])

                with tab_fields:
                    # Dealer + Model
                    section_header("Document Identity")
                    c1, c2 = st.columns(2)
                    with c1:
                        field_card("Dealer Name", result["dealer_name"], "🏪")
                    with c2:
                        field_card("Tractor Model", result["model_name"], "🚜")

                    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

                    # HP + Cost
                    section_header("Financial Details")
                    c1, c2 = st.columns(2)
                    with c1:
                        hp_val = f"{result['horse_power']} HP" if result["horse_power"] else None
                        field_card("Horse Power", hp_val, "⚙️", mono=True, highlight=True)
                    with c2:
                        field_card(
                            "Asset Cost",
                            format_currency(result["asset_cost"]),
                            "💰", mono=True, highlight=True,
                        )

                    st.markdown('<div style="height:0.6rem;"></div>', unsafe_allow_html=True)

                    # Signature + Stamp
                    section_header("Document Authenticity")
                    c1, c2 = st.columns(2)
                    with c1:
                        detection_badge(
                            "Dealer Signature",
                            bool(result["signature_present"]),
                            result["signature_bbox"],
                            "✍️",
                        )
                    with c2:
                        detection_badge(
                            "Dealer Stamp",
                            bool(result["stamp_present"]),
                            result["stamp_bbox"],
                            "🔵",
                        )

                with tab_json:
                    json_output = {
                        "doc_id": st.session_state.get("last_filename", "document"),
                        "fields": {
                            "dealer_name": result["dealer_name"],
                            "model_name":  result["model_name"],
                            "horse_power": result["horse_power"],
                            "asset_cost":  result["asset_cost"],
                            "signature": {
                                "present": bool(result["signature_present"]),
                                "bbox":    result["signature_bbox"],
                            },
                            "stamp": {
                                "present": bool(result["stamp_present"]),
                                "bbox":    result["stamp_bbox"],
                            },
                        },
                        "confidence":          round(result["confidence"], 4),
                        "processing_time_sec": round(result["processing_time"], 3),
                        "cost_estimate_usd":   0.0,
                    }
                    st.code(
                        json.dumps(json_output, indent=2, ensure_ascii=False),
                        language="json",
                    )

                    # Download button
                    st.download_button(
                        label="⬇  Download JSON",
                        data=json.dumps(json_output, indent=2, ensure_ascii=False),
                        file_name=f"{Path(st.session_state.get('last_filename','result')).stem}_result.json",
                        mime="application/json",
                        use_container_width=True,
                    )
        else:
            # Uploaded but not yet run
            st.markdown("""
            <div style="
                height: 300px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                border: 1.5px dashed #21262d;
                border-radius: 14px;
                background: rgba(22,27,34,0.4);
            ">
                <p style="color:#484f58; font-size:0.9rem; text-align:center;">
                    Click <strong style="color:#d4a853">⚡ Extract Fields</strong><br>
                    to begin extraction
                </p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        border-top: 1px solid #21262d;
        padding: 1rem 3rem;
        margin-top: 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <span style="color:#484f58; font-size:0.75rem;">
            InvoiceAI · Built for IDFC First Bank Hackathon · Runs 100% locally
        </span>
        <span style="color:#484f58; font-size:0.75rem;">
            Qwen2-VL · PaddleOCR · OpenCV · RapidFuzz
        </span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
