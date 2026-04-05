"""
dashboard.py
------------
Streamlit dashboard for Invoice OCR Pipeline — Banker's View.
Run: streamlit run dashboard.py

Place this file at your project root (same level as executable.py).
"""

import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# ── Path setup ───────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT / "src"))

RESULTS_PATH = _ROOT / "sample_output" / "result.json"
CUMULATIVE_PATH = _ROOT / "Results_Cumulative.xlsx"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Intelligence | Banker's Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* Background */
  .stApp { background: #0d1117; }
  section[data-testid="stSidebar"] { background: #111827; border-right: 1px solid #1f2937; }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }

  /* Top banner */
  .top-banner {
    background: linear-gradient(135deg, #0f2027 0%, #1a3a4a 50%, #0f2027 100%);
    border: 1px solid #1e4d6b;
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .banner-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #e8f4f8;
    margin: 0;
    letter-spacing: -0.5px;
  }
  .banner-sub {
    color: #64b5d6;
    font-size: 0.85rem;
    font-weight: 400;
    margin-top: 4px;
    letter-spacing: 2px;
    text-transform: uppercase;
  }
  .banner-badge {
    background: #0e3549;
    border: 1px solid #1e6a8a;
    color: #7ecfed;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 1px;
  }

  /* KPI cards */
  .kpi-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 14px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
  }
  .kpi-card:hover { border-color: #2563eb44; }
  .kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #2563eb);
  }
  .kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #93c5d6;
    margin-bottom: 8px;
  }
  .kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: #f0f6ff;
    line-height: 1;
  }
  .kpi-delta {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 6px;
  }
  .kpi-delta.pos { color: #34d399; }
  .kpi-delta.neg { color: #f87171; }

  /* Upload section */
  .upload-zone {
    background: linear-gradient(135deg, #0d1f35 0%, #0a1628 100%);
    border: 2px dashed #1e4d6b;
    border-radius: 20px;
    padding: 48px 32px;
    text-align: center;
    transition: border-color 0.3s, background 0.3s;
    margin: 20px 0;
  }
  .upload-zone:hover {
    border-color: #2d8ab5;
    background: linear-gradient(135deg, #0d2a45 0%, #0d1f35 100%);
  }
  .upload-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #c8e6f5;
    margin-bottom: 8px;
  }
  .upload-sub { color: #7eb8d4; font-size: 0.88rem; }

  /* Result card */
  .result-card {
    background: #111827;
    border: 1px solid #1e3a4a;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .result-field-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7eb8d4;
    margin-bottom: 4px;
  }
  .result-field-value {
    font-size: 1.05rem;
    color: #d4eaf7;
    font-weight: 500;
  }

  /* Confidence bar */
  .conf-bar-bg {
    background: #1f2937;
    border-radius: 999px;
    height: 8px;
    width: 100%;
    margin-top: 6px;
  }
  .conf-bar-fill {
    height: 8px;
    border-radius: 999px;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    transition: width 0.8s ease;
  }

  /* Status pills */
  .pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }
  .pill-yes { background: #064e3b; color: #34d399; border: 1px solid #065f46; }
  .pill-no  { background: #1f2937; color: #cbd5e1; border: 1px solid #374151; }
  .pill-warn { background: #451a03; color: #fb923c; border: 1px solid #7c2d12; }

  /* Section headers */
  .section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #c8dff0;
    border-bottom: 1px solid #1f2937;
    padding-bottom: 10px;
    margin: 28px 0 18px 0;
  }

  /* Sidebar items */
  .sidebar-stat {
    background: #1a2535;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    border-left: 3px solid #2563eb;
  }
  .sidebar-stat-label { font-size: 0.7rem; color: #93c5d6; text-transform: uppercase; letter-spacing: 1px; }
  .sidebar-stat-val { font-size: 1.1rem; color: #c8dff0; font-weight: 600; margin-top: 2px; }

  /* Plotly charts dark background fix */
  .js-plotly-plot { border-radius: 12px; overflow: hidden; }

  /* Streamlit overrides */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #0e7490) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    padding: 14px 40px !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
    cursor: pointer !important;
  }
  .stButton > button:hover { opacity: 0.85 !important; }

  /* Radio buttons — navigation */
  div[data-testid="stSidebar"] div[role="radiogroup"] label {
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    color: #93c5d6 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: background 0.15s, color 0.15s !important;
    border: 1px solid transparent !important;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: #1a2e44 !important;
    color: #e2f3fc !important;
    border-color: #1e4d6b !important;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #0e2d45 !important;
    color: #7ecfed !important;
    border-color: #2563eb !important;
    font-weight: 600 !important;
  }
  /* Hide the default radio circle */
  div[data-testid="stSidebar"] div[role="radiogroup"] label input[type="radio"] {
    display: none !important;
  }
  div[data-testid="stSidebar"] div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
    color: inherit !important;
    font-size: inherit !important;
    margin: 0 !important;
  }
  div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
  }
  div[data-testid="stFileUploader"] > div {
    background: #0d1f35 !important;
    border: 2px dashed #1e4d6b !important;
    border-radius: 16px !important;
    padding: 20px !important;
  }
  .stDataFrame { border-radius: 12px; overflow: hidden; }
  div[data-testid="metric-container"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#8bafc7"),
    margin=dict(l=16, r=16, t=32, b=16),
    xaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", zerolinecolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", linecolor="#1f2937", zerolinecolor="#1f2937"),
)


def kpi_card(label: str, value: str, delta: str = "", accent: str = "#2563eb", delta_pos: bool = True):
    delta_class = "pos" if delta_pos else "neg"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{accent}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def load_cumulative() -> pd.DataFrame:
    if not CUMULATIVE_PATH.exists():
        return pd.DataFrame()
    df = pd.read_excel(CUMULATIVE_PATH).dropna(subset=["doc_id"])
    df["Signature_Present"] = df["Signature_Present"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    df["Stamp_Present"] = df["Stamp_Present"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    return df


def fmt_currency(val) -> str:
    if val is None or pd.isna(val):
        return "—"
    if val >= 1_00_00_000:
        return f"₹{val/1_00_00_000:.2f} Cr"
    elif val >= 1_00_000:
        return f"₹{val/1_00_000:.1f} L"
    return f"₹{int(val):,}"


def compliance_badge(val: bool | None) -> str:
    if val is True:
        return '<span class="pill pill-yes">✓ Present</span>'
    elif val is False:
        return '<span class="pill pill-no">✗ Absent</span>'
    return '<span class="pill pill-warn">? Unknown</span>'


@st.cache_resource(show_spinner=False)
def get_pipeline_components():
    """
    Load pipeline components exactly once per Streamlit server process and
    cache them via st.cache_resource — survives every page rerun and every
    navigation between tabs.  The VLM model is loaded here so it is warm
    before the first upload arrives.
    """
    from executable import build_pipeline_components, load_config
    cfg = load_config(_ROOT / "configs" / "config.yaml")
    components = build_pipeline_components(cfg)
    # Force model load now so the first upload doesn't pay the 30-60s penalty.
    components["vlm_extractor"]._ensure_loaded()
    return components


def run_pipeline(file_path: Path) -> dict | None:
    """Infer on a single document using the already-loaded pipeline components."""
    try:
        from executable import process_document
        components = get_pipeline_components()  # instant — returns cached object
        return process_document(file_path, components)
    except Exception as exc:
        st.error(f"Pipeline error: {exc}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 20px 0 12px 0;">
          <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#c8dff0;">🏦 Loan Intelligence</div>
          <div style="font-size:0.72rem; color:#7eb8d4; letter-spacing:2px; text-transform:uppercase; margin-top:4px;">Asset Finance Division</div>
        </div>
        """, unsafe_allow_html=True)
        st.divider()

        if not df.empty:
            sig_rate = df["Signature_Present"].mean() * 100 if "Signature_Present" in df else 0
            stamp_rate = df["Stamp_Present"].mean() * 100 if "Stamp_Present" in df else 0
            avg_conf = df["confidence"].mean() * 100 if "confidence" in df else 0
            total_portfolio = df["asset_cost"].sum() if "asset_cost" in df else 0

            st.markdown(f"""
            <div class="sidebar-stat"><div class="sidebar-stat-label">Total Portfolio Value</div>
            <div class="sidebar-stat-val">{fmt_currency(total_portfolio)}</div></div>

            <div class="sidebar-stat" style="border-left-color:#06b6d4"><div class="sidebar-stat-label">Avg Extraction Confidence</div>
            <div class="sidebar-stat-val">{avg_conf:.1f}%</div></div>

            <div class="sidebar-stat" style="border-left-color:#34d399"><div class="sidebar-stat-label">Signature Compliance</div>
            <div class="sidebar-stat-val">{sig_rate:.1f}%</div></div>

            <div class="sidebar-stat" style="border-left-color:#f59e0b"><div class="sidebar-stat-label">Stamp Compliance</div>
            <div class="sidebar-stat-val">{stamp_rate:.1f}%</div></div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div style="font-size:0.72rem; color:#93c5d6; text-transform:uppercase; letter-spacing:1px; padding:4px 0;">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("", ["📊  Portfolio Dashboard", "📄  Process Invoice"], label_visibility="collapsed")
        st.divider()
        st.markdown('<div style="font-size:0.68rem; color:#7eb8d4; text-align:center;">Invoice OCR Pipeline v1.0<br>Qwen2-VL · CUDA Accelerated</div>', unsafe_allow_html=True)

    return page.split("  ")[1].strip()


# ══════════════════════════════════════════════════════════════════════════════
# Dashboard Page
# ══════════════════════════════════════════════════════════════════════════════

def render_dashboard(df: pd.DataFrame):
    # Banner
    st.markdown(f"""
    <div class="top-banner">
      <div>
        <div class="banner-title">Portfolio Intelligence Dashboard</div>
        <div class="banner-sub">Asset Finance · Tractor Loan Division</div>
      </div>
      <div>
        <span class="banner-badge">🟢 LIVE DATA · {len(df)} INVOICES</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if df.empty:
        st.warning("No data found. Place Results_Cumulative.xlsx in the project root.")
        return

    # ── Row 1: Top KPIs ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)

    total_docs = len(df)
    total_val = df["asset_cost"].sum()
    avg_conf = df["confidence"].mean()
    sig_compliance = df["Signature_Present"].mean()
    stamp_compliance = df["Stamp_Present"].mean()
    both_compliant = ((df["Signature_Present"] == True) & (df["Stamp_Present"] == True)).mean()

    with c1:
        kpi_card("Total Invoices Processed", f"{total_docs:,}", "Cumulative", "#2563eb")
    with c2:
        kpi_card("Total Portfolio Value", fmt_currency(total_val), "Asset Cost", "#0891b2")
    with c3:
        kpi_card("Avg AI Confidence", f"{avg_conf*100:.1f}%", f"Min {df['confidence'].min()*100:.0f}% / Max {df['confidence'].max()*100:.0f}%", "#7c3aed")
    with c4:
        kpi_card("Signature Compliance", f"{sig_compliance*100:.1f}%", f"{int(sig_compliance*total_docs)} of {total_docs} signed", "#059669", sig_compliance > 0.5)
    with c5:
        kpi_card("Full Compliance Rate", f"{both_compliant*100:.1f}%", "Sig + Stamp both present", "#d97706", both_compliant > 0.5)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 2: Charts ────────────────────────────────────────────────────────
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">Top Dealers by Portfolio Value</div>', unsafe_allow_html=True)
        dealer_val = (
            df.groupby("dealer_name")["asset_cost"]
            .sum()
            .sort_values(ascending=False)
            .head(12)
            .reset_index()
        )
        fig = px.bar(
            dealer_val, x="asset_cost", y="dealer_name", orientation="h",
            color="asset_cost",
            color_continuous_scale=["#1e3a5f", "#2563eb", "#06b6d4"],
            labels={"asset_cost": "Portfolio Value (₹)", "dealer_name": ""},
        )
        fig.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=360)
        fig.update_traces(
            hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
            marker_line_width=0,
        )
        fig.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Compliance Breakdown</div>', unsafe_allow_html=True)
        comp_data = {
            "Category": ["Sig ✓ Stamp ✓", "Sig ✓ Stamp ✗", "Sig ✗ Stamp ✓", "Sig ✗ Stamp ✗"],
            "Count": [
                ((df["Signature_Present"] == True) & (df["Stamp_Present"] == True)).sum(),
                ((df["Signature_Present"] == True) & (df["Stamp_Present"] == False)).sum(),
                ((df["Signature_Present"] == False) & (df["Stamp_Present"] == True)).sum(),
                ((df["Signature_Present"] == False) & (df["Stamp_Present"] == False)).sum(),
            ],
            "Color": ["#34d399", "#f59e0b", "#f59e0b", "#f87171"]
        }
        fig2 = go.Figure(go.Pie(
            labels=comp_data["Category"],
            values=comp_data["Count"],
            hole=0.62,
            marker_colors=comp_data["Color"],
            textinfo="percent",
            textfont=dict(size=12),
            hovertemplate="<b>%{label}</b><br>%{value} invoices (%{percent})<extra></extra>",
        ))
        fig2.add_annotation(
            text=f"<b style='font-size:24px'>{both_compliant*100:.0f}%</b><br>Compliant",
            x=0.5, y=0.5, showarrow=False,
            font=dict(color="#c8dff0", family="DM Sans", size=14),
        )
        fig2.update_layout(**CHART_LAYOUT, showlegend=True, height=360,
                           legend=dict(orientation="v", x=1, y=0.5, font=dict(size=11)))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 3: Model distribution + Confidence histogram ────────────────────
    col_c, col_d = st.columns([2, 3])

    with col_c:
        st.markdown('<div class="section-header">Top Tractor Models</div>', unsafe_allow_html=True)
        model_counts = df["model_name"].value_counts().head(10).reset_index()
        model_counts.columns = ["Model", "Count"]
        fig3 = px.bar(
            model_counts, x="Count", y="Model", orientation="h",
            color="Count", color_continuous_scale=["#1e3a5f", "#7c3aed"],
        )
        fig3.update_layout(**CHART_LAYOUT, coloraxis_showscale=False, height=320)
        fig3.update_traces(marker_line_width=0)
        fig3.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-header">AI Confidence Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(
            df, x="confidence", nbins=30,
            color_discrete_sequence=["#2563eb"],
            labels={"confidence": "Confidence Score", "count": "Invoices"},
        )
        fig4.add_vline(x=avg_conf, line_dash="dash", line_color="#06b6d4",
                       annotation_text=f"Avg {avg_conf*100:.1f}%",
                       annotation_font_color="#06b6d4")
        fig4.add_vline(x=0.75, line_dash="dot", line_color="#f59e0b",
                       annotation_text="Alert threshold",
                       annotation_font_color="#f59e0b")
        fig4.update_layout(**CHART_LAYOUT, height=320, bargap=0.05)
        fig4.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 4: Risk flags table ─────────────────────────────────────────────
    st.markdown('<div class="section-header">⚠️ Risk Flags — Low Confidence or Missing Compliance</div>', unsafe_allow_html=True)
    risk_df = df[
        (df["confidence"] < 0.75) |
        (df["Signature_Present"] == False) |
        (df["Stamp_Present"] == False)
    ][["doc_id", "dealer_name", "model_name", "asset_cost", "confidence",
       "Signature_Present", "Stamp_Present"]].copy()
    risk_df["asset_cost"] = risk_df["asset_cost"].apply(fmt_currency)
    risk_df["confidence"] = (risk_df["confidence"] * 100).round(1).astype(str) + "%"
    risk_df.columns = ["Document ID", "Dealer", "Model", "Asset Cost", "Confidence", "Signed", "Stamped"]
    st.dataframe(
        risk_df.head(20),
        use_container_width=True,
        hide_index=True,
    )
    if len(risk_df) > 20:
        st.caption(f"Showing 20 of {len(risk_df)} flagged records.")


# ══════════════════════════════════════════════════════════════════════════════
# Process Invoice Page
# ══════════════════════════════════════════════════════════════════════════════

def render_process_invoice():
    st.markdown("""
    <div class="top-banner">
      <div>
        <div class="banner-title">Process New Invoice</div>
        <div class="banner-sub">AI-Powered Field Extraction · Qwen2-VL</div>
      </div>
      <span class="banner-badge">🤖 VLM READY</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Big Upload Button ───────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center; margin: 10px 0 24px 0;">
      <div style="font-family:'DM Serif Display',serif; font-size:1.1rem; color:#94b8cc; margin-bottom:16px;">
        Drop an invoice below and the pipeline will extract all fields automatically
      </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "📂  Upload Invoice",
        type=["pdf", "png", "jpg", "jpeg", "tiff"],
        label_visibility="collapsed",
        help="Supported: PDF, PNG, JPG, JPEG, TIFF",
    )

    if not uploaded:
        st.markdown("""
        <div style="text-align:center; padding:32px; color:#94b8cc;">
          <div style="font-size:3rem; margin-bottom:12px;">🏦</div>
          <div style="font-family:'DM Serif Display',serif; font-size:1.4rem; color:#7eb8d4;">
            Awaiting Invoice Upload
          </div>
          <div style="font-size:0.85rem; color:#94b8cc; margin-top:8px;">
            Upload a tractor loan invoice to extract dealer, model, cost, HP, signatures &amp; stamps
          </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Show preview
    col_prev, col_info = st.columns([1, 1])
    with col_prev:
        st.markdown('<div class="section-header">Document Preview</div>', unsafe_allow_html=True)
        if uploaded.type == "application/pdf":
            st.info("PDF uploaded — preview not available. Processing will proceed.")
        else:
            try:
                img = Image.open(uploaded)
                st.image(img, use_column_width=True, caption=uploaded.name)
            except Exception:
                st.warning("Could not render preview.")

    with col_info:
        st.markdown('<div class="section-header">File Info</div>', unsafe_allow_html=True)
        size_kb = uploaded.size / 1024
        st.markdown(f"""
        <div class="result-card">
          <div style="margin-bottom:16px">
            <div class="result-field-label">File Name</div>
            <div class="result-field-value">{uploaded.name}</div>
          </div>
          <div style="margin-bottom:16px">
            <div class="result-field-label">File Type</div>
            <div class="result-field-value">{uploaded.type}</div>
          </div>
          <div>
            <div class="result-field-label">File Size</div>
            <div class="result-field-value">{size_kb:.1f} KB</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div style="margin-top:20px"></div>', unsafe_allow_html=True)
        run_btn = st.button("🚀  Run Invoice Extraction", use_container_width=True)

    # ── Run extraction ──────────────────────────────────────────────────────
    if run_btn:
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = Path(tmp.name)

        status_box = st.empty()
        with status_box.container():
            with st.spinner("⚙️  Running pipeline — model loading + extraction (30–60s on first run)..."):
                t_start = time.time()
                result = run_pipeline(tmp_path)
                elapsed = time.time() - t_start

        tmp_path.unlink(missing_ok=True)

        if result is None:
            st.error("Extraction failed. Check logs for details.")
            return

        status_box.success(f"✅ Extraction complete in {elapsed:.1f}s")

        # ── Display results ─────────────────────────────────────────────────
        st.markdown('<div class="section-header">Extracted Fields</div>', unsafe_allow_html=True)

        fields = result.get("fields", result)
        conf = result.get("confidence", result.get("overall_confidence", 0))

        dealer = fields.get("dealer_name") or "—"
        model  = fields.get("model_name")  or "—"
        hp     = fields.get("horse_power")
        cost   = fields.get("asset_cost")
        sig    = fields.get("signature", {})
        stamp  = fields.get("stamp", {})

        if isinstance(sig, dict):
            sig_present = sig.get("present")
        else:
            sig_present = sig

        if isinstance(stamp, dict):
            stamp_present = stamp.get("present")
        else:
            stamp_present = stamp

        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Dealer Name</div>
              <div class="result-field-value" style="font-size:1.15rem">{dealer}</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Tractor Model</div>
              <div class="result-field-value" style="font-size:1.15rem">{model}</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Asset Cost</div>
              <div class="result-field-value" style="font-size:1.15rem; color:#34d399">{fmt_currency(cost)}</div>
            </div>""", unsafe_allow_html=True)

        r4, r5, r6 = st.columns(3)
        with r4:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Horse Power</div>
              <div class="result-field-value">{f'{int(hp)} HP' if hp else '—'}</div>
            </div>""", unsafe_allow_html=True)
        with r5:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Signature</div>
              <div class="result-field-value">{compliance_badge(sig_present)}</div>
            </div>""", unsafe_allow_html=True)
        with r6:
            st.markdown(f"""
            <div class="result-card">
              <div class="result-field-label">Stamp</div>
              <div class="result-field-value">{compliance_badge(stamp_present)}</div>
            </div>""", unsafe_allow_html=True)

        # Confidence meter
        st.markdown(f"""
        <div class="result-card" style="margin-top:8px">
          <div class="result-field-label">AI Extraction Confidence</div>
          <div style="display:flex; align-items:center; gap:16px; margin-top:8px">
            <div class="conf-bar-bg" style="flex:1">
              <div class="conf-bar-fill" style="width:{conf*100:.1f}%"></div>
            </div>
            <div style="color:#7ecfed; font-size:1.1rem; font-weight:600; min-width:52px">{conf*100:.1f}%</div>
          </div>
          <div style="margin-top:8px; font-size:0.78rem; color:#94b8cc">
            {'⚠️ Below 75% threshold — manual review recommended' if conf < 0.75 else '✓ Above confidence threshold'}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Raw JSON toggle
        with st.expander("🔍 View Raw JSON Output"):
            st.json(result)

        # Banker's verdict
        st.markdown('<div class="section-header">Banker\'s Verdict</div>', unsafe_allow_html=True)
        issues = []
        if not sig_present:
            issues.append("Missing signature — invoice not legally signed")
        if not stamp_present:
            issues.append("Missing stamp — document may be unverified")
        if conf < 0.75:
            issues.append(f"Low AI confidence ({conf*100:.1f}%) — field extraction uncertain")
        if cost and cost > 50_00_000:
            issues.append(f"High-value asset ({fmt_currency(cost)}) — enhanced due diligence required")

        if not issues:
            st.markdown("""
            <div style="background:#064e3b; border:1px solid #065f46; border-radius:12px; padding:20px 24px;">
              <div style="font-size:1.4rem; color:#34d399; font-weight:700">✅ CLEAR FOR PROCESSING</div>
              <div style="color:#a7f3d0; margin-top:6px; font-size:0.88rem">
                All compliance fields verified. Invoice is ready for loan disbursement workflow.
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            issues_html = "".join(f"<li style='margin:6px 0; color:#fca5a5'>⚠ {i}</li>" for i in issues)
            st.markdown(f"""
            <div style="background:#450a0a; border:1px solid #7f1d1d; border-radius:12px; padding:20px 24px;">
              <div style="font-size:1.4rem; color:#f87171; font-weight:700">🔴 MANUAL REVIEW REQUIRED</div>
              <ul style="margin-top:12px; padding-left:16px; font-size:0.88rem">
                {issues_html}
              </ul>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    df = load_cumulative()
    page = render_sidebar(df)

    # ── Warm up model on first load ──────────────────────────────────────────
    # get_pipeline_components() is cached — calling it here means the model
    # starts loading the moment the app opens, not when the user clicks upload.
    # A spinner in the sidebar shows status without blocking the dashboard view.
    if "model_ready" not in st.session_state:
        st.session_state["model_ready"] = False

    if not st.session_state["model_ready"]:
        with st.sidebar:
            with st.spinner("⚙️ Loading AI model..."):
                try:
                    get_pipeline_components()
                    st.session_state["model_ready"] = True
                except Exception as exc:
                    st.error(f"Model load failed: {exc}")
    else:
        with st.sidebar:
            st.markdown(
                '<div style="text-align:center; font-size:0.75rem; color:#34d399; '
                'padding:6px 0;">✓ Model ready</div>',
                unsafe_allow_html=True,
            )

    if page == "Portfolio Dashboard":
        render_dashboard(df)
    else:
        render_process_invoice()


if __name__ == "__main__":
    main()
