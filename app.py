from __future__ import annotations

import html
import json
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import FastMarkerCluster, Fullscreen, HeatMap, MeasureControl, MiniMap
from streamlit_folium import st_folium


# =============================================================================
# PAGE AND PROJECT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="England EDM Water & Spill-Risk Observatory",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"

RISK_ORDER = ["Low", "Medium", "High"]
RISK_COLOURS = {
    "Low": "#4A9C7D",       # colour-blind-safe teal
    "Medium": "#E2A45C",    # warm amber
    "High": "#D66565",      # soft coral red
}
RISK_SYMBOLS = {"Low": "●", "Medium": "◆", "High": "▲"}

INK = "#173D3A"
MUTED = "#5D7772"
PALE_MINT = "#EAF6F0"
PALE_BLUE = "#E9F4F8"
PALE_LAVENDER = "#F1ECF7"
PALE_AMBER = "#FFF3DD"
PAPER = "#FBFDF9"


# =============================================================================
# ENVIRONMENTAL DESIGN SYSTEM
# =============================================================================

st.markdown(
    """
    <style>
      :root {
        --edm-ink: #173D3A;
        --edm-muted: #5D7772;
        --edm-mint: #EAF6F0;
        --edm-blue: #E9F4F8;
        --edm-lavender: #F1ECF7;
        --edm-amber: #FFF3DD;
        --edm-paper: #FBFDF9;
        --edm-teal: #4A9C7D;
        --edm-water: #68AFC2;
        --edm-coral: #D66565;
      }

      html, body, [class*="css"] {
        font-family: "Atkinson Hyperlegible", Verdana, Tahoma, Arial, sans-serif;
        color: var(--edm-ink);
        font-size: 15.5px;
        line-height: 1.52;
        letter-spacing: 0.01em;
      }

      p, li, label, input, button, select, textarea {
        font-family: "Atkinson Hyperlegible", Verdana, Tahoma, Arial, sans-serif !important;
        line-height: 1.52 !important;
        letter-spacing: 0.01em;
      }

      h1, h2, h3, h4 {
        font-family: "Atkinson Hyperlegible", Verdana, Tahoma, Arial, sans-serif !important;
        letter-spacing: 0 !important;
      }

      .stApp {
        background:
          radial-gradient(circle at 86% 4%, rgba(178, 220, 226, 0.35), transparent 24rem),
          radial-gradient(circle at 8% 92%, rgba(190, 224, 204, 0.38), transparent 25rem),
          linear-gradient(145deg, #FBFDF9 0%, #F3FAF7 48%, #F5F9FC 100%);
      }

      .block-container {
        max-width: 1540px;
        padding-top: 0.65rem;
        padding-bottom: 3rem;
      }

      #MainMenu, footer, [data-testid="stDecoration"] {
        visibility: hidden;
      }

      section[data-testid="stSidebar"] {
        background:
          linear-gradient(180deg, rgba(227, 244, 239, 0.98), rgba(232, 244, 249, 0.98));
        border-right: 1px solid rgba(54, 121, 112, 0.18);
      }

      section[data-testid="stSidebar"] > div {
        padding-top: 1.25rem;
      }

      section[data-testid="stSidebar"] label,
      section[data-testid="stSidebar"] p,
      section[data-testid="stSidebar"] span {
        color: var(--edm-ink);
      }

      div[role="radiogroup"] label {
        background: rgba(255,255,255,0.68);
        border: 1px solid rgba(62, 127, 117, 0.15);
        border-radius: 12px;
        padding: 0.55rem 0.72rem;
        margin-bottom: 0.28rem;
        transition: background 0.18s ease, transform 0.18s ease;
      }

      div[role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.82);
        transform: translateX(2px);
      }

      .edm-brand {
        padding: 0.75rem 0.25rem 1rem;
        color: var(--edm-ink);
      }

      .edm-brand-mark {
        display: inline-flex;
        width: 44px;
        height: 44px;
        align-items: center;
        justify-content: center;
        margin-right: 9px;
        border-radius: 14px 14px 18px 18px;
        color: white;
        background: linear-gradient(145deg, #6BB8C6, #3D8D82);
        box-shadow: 0 8px 22px rgba(44, 112, 104, 0.20);
        font-size: 24px;
      }

      .edm-hero {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1.45fr) minmax(250px, 0.62fr);
        gap: 1rem;
        align-items: center;
        padding: 1.2rem 1.5rem;
        margin: 0.1rem 0 0.85rem;
        border: 1px solid rgba(61, 129, 118, 0.18);
        border-radius: 24px;
        background:
          linear-gradient(125deg, rgba(231, 247, 239, 0.98), rgba(231, 244, 249, 0.96));
        box-shadow: 0 16px 42px rgba(35, 91, 83, 0.10);
      }

      .edm-hero h1 {
        margin: 0;
        max-width: 880px;
        color: var(--edm-ink);
        font-size: clamp(1.85rem, 3vw, 2.8rem);
        line-height: 1.08;
        letter-spacing: 0;
      }

      .edm-hero p {
        max-width: 850px;
        margin: 0.55rem 0 0;
        color: var(--edm-muted);
        font-size: 0.98rem;
        line-height: 1.48;
      }

      .edm-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin-bottom: 0.65rem;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        color: #276A61;
        background: rgba(255,255,255,0.70);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
      }

      .edm-water-art {
        min-height: 170px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .edm-cloud-a { animation: edmCloudA 8s ease-in-out infinite alternate; }
      .edm-cloud-b { animation: edmCloudB 10s ease-in-out infinite alternate; }
      .edm-rain-drop { animation: edmRain 1.6s ease-in-out infinite; }
      .edm-rain-drop:nth-child(2n) { animation-delay: .45s; }
      .edm-rain-drop:nth-child(3n) { animation-delay: .85s; }
      .edm-river-flow {
        stroke-dasharray: 28 16;
        animation: edmRiver 3.2s linear infinite;
      }

      @keyframes edmCloudA {
        from { transform: translateX(-8px); }
        to { transform: translateX(16px); }
      }

      @keyframes edmCloudB {
        from { transform: translateX(10px); }
        to { transform: translateX(-15px); }
      }

      @keyframes edmRain {
        0% { transform: translateY(-4px); opacity: 0; }
        30% { opacity: .85; }
        100% { transform: translateY(22px); opacity: 0; }
      }

      @keyframes edmRiver {
        to { stroke-dashoffset: -88; }
      }

      .edm-section-header {
        margin: 0.15rem 0 0.65rem;
        padding: 0.72rem 1rem;
        border-left: 6px solid #4A9C7D;
        border-radius: 0 16px 16px 0;
        background: linear-gradient(90deg, rgba(234,246,240,0.95), rgba(255,255,255,0.50));
      }

      .edm-section-header h2 {
        margin: 0;
        color: var(--edm-ink);
        font-size: 1.42rem;
        line-height: 1.15;
      }

      .edm-section-header p {
        margin: 0.22rem 0 0;
        color: var(--edm-muted);
        line-height: 1.4;
        font-size: .91rem;
      }

      .edm-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.65rem;
        margin: 0.5rem 0 0.85rem;
      }

      .edm-metric-card {
        position: relative;
        overflow: hidden;
        min-height: 104px;
        padding: 0.78rem 0.9rem;
        border: 1px solid rgba(48, 112, 103, 0.14);
        border-radius: 18px;
        background: rgba(255,255,255,0.72);
        box-shadow: 0 8px 24px rgba(38, 91, 84, 0.07);
      }

      .edm-metric-card::after {
        content: "";
        position: absolute;
        right: -25px;
        bottom: -35px;
        width: 95px;
        height: 95px;
        border-radius: 50%;
        background: var(--card-accent, #B9DFD1);
        opacity: 0.32;
      }

      .edm-metric-label {
        color: var(--edm-muted);
        font-size: 0.83rem;
        font-weight: 700;
        letter-spacing: 0.025em;
        text-transform: uppercase;
      }

      .edm-metric-value {
        margin: 0.28rem 0 0.2rem;
        color: var(--edm-ink);
        font-size: 1.62rem;
        font-weight: 750;
        line-height: 1.05;
      }

      .edm-metric-note {
        max-width: 90%;
        color: var(--edm-muted);
        font-size: 0.83rem;
        line-height: 1.35;
      }

      .edm-panel {
        padding: 1rem 1.15rem;
        margin: 0.65rem 0;
        border: 1px solid rgba(48, 112, 103, 0.14);
        border-radius: 18px;
        background: rgba(255,255,255,0.64);
        box-shadow: 0 8px 24px rgba(38, 91, 84, 0.06);
      }

      .edm-plain-card {
        padding: 1rem 1.05rem;
        margin: .55rem 0;
        border: 1px solid rgba(48, 112, 103, 0.16);
        border-radius: 16px;
        background: rgba(255,255,255,.78);
        box-shadow: 0 7px 20px rgba(38, 91, 84, .07);
      }

      .edm-plain-card h3 {
        margin: 0 0 .35rem;
        color: var(--edm-ink);
        font-size: 1.08rem;
      }

      .edm-rank-row {
        padding: .72rem .78rem;
        margin: .48rem 0;
        border-radius: 13px;
        background: rgba(255,255,255,.82);
        border: 1px solid rgba(48,112,103,.14);
        border-left: 7px solid var(--rank-colour, #68AFC2);
      }

      .edm-rank-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 30px;
        height: 30px;
        margin-right: .45rem;
        border-radius: 50%;
        background: #E9F4F8;
        color: #245B61;
        font-weight: 800;
      }

      .edm-rank-name { font-weight: 800; color: var(--edm-ink); }
      .edm-rank-detail { margin-top: .25rem; color: var(--edm-muted); font-size: .84rem; }

      .edm-risk-guide {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: .8rem;
        margin: .65rem 0 1.1rem;
      }

      .edm-risk-guide > div {
        padding: .9rem 1rem;
        border-radius: 15px;
        background: rgba(255,255,255,.78);
        border-top: 8px solid var(--risk-colour);
      }

      .edm-simple-label {
        color: #365F5B;
        font-size: .84rem;
        font-weight: 750;
      }

      .edm-access-note {
        padding: .7rem .8rem;
        border-radius: 12px;
        background: rgba(255,255,255,.72);
        color: #365F5B;
        font-size: .82rem;
      }

      :focus-visible {
        outline: 4px solid #2B7D89 !important;
        outline-offset: 3px !important;
      }

      .edm-banner {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        padding: 0.65rem 0.82rem;
        margin: 0.45rem 0 0.75rem;
        border-radius: 14px;
        color: var(--edm-ink);
        background: var(--banner-bg, #E9F4F8);
        border-left: 6px solid var(--banner-edge, #68AFC2);
        line-height: 1.42;
        font-size: .91rem;
      }

      .edm-journey {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.75rem 0 1.2rem;
      }

      .edm-journey-step {
        padding: 1rem;
        border-radius: 17px;
        background: linear-gradient(145deg, rgba(255,255,255,0.78), rgba(234,246,240,0.70));
        border: 1px solid rgba(60, 125, 115, 0.14);
      }

      .edm-journey-number {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        color: white;
        background: #4A9C7D;
        font-weight: 800;
      }

      div[data-testid="stTabs"] button {
        border-radius: 12px 12px 0 0;
      }

      div[data-testid="stDataFrame"] {
        border: 1px solid rgba(48, 112, 103, 0.14);
        border-radius: 14px;
        overflow: hidden;
      }

      div[data-testid="stSelectbox"] > div > div,
      div[data-testid="stTextInput"] input,
      div[data-testid="stMultiSelect"] > div > div {
        background: rgba(255,255,255,0.78);
        border-radius: 11px;
      }

      .stButton > button, .stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid rgba(55, 120, 110, 0.24);
        color: var(--edm-ink);
        background: linear-gradient(145deg, #F8FCFA, #E9F5F1);
      }

      div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: .38rem;
        padding: .42rem;
        margin: .15rem 0 .75rem;
        border: 1px solid rgba(55,120,110,.16);
        border-radius: 16px;
        background: rgba(255,255,255,.62);
        box-shadow: 0 6px 18px rgba(38,91,84,.05);
      }

      div[data-testid="stRadio"] > div[role="radiogroup"] label {
        flex: 1 1 145px;
        justify-content: center;
        min-height: 38px;
        padding: .38rem .55rem;
        margin: 0;
        font-size: .86rem;
        font-weight: 750;
        text-align: center;
        background: linear-gradient(145deg, rgba(255,255,255,.88), rgba(232,246,240,.78));
      }

      .edm-page-grid {
        display: grid;
        grid-template-columns: repeat(6, minmax(0,1fr));
        gap: .62rem;
        margin: .5rem 0 .9rem;
      }

      .edm-page-card {
        min-height: 105px;
        padding: .8rem;
        border: 1px solid rgba(52,114,105,.14);
        border-radius: 17px;
        background: linear-gradient(145deg, rgba(255,255,255,.88), var(--page-tint,#EAF6F0));
        box-shadow: 0 8px 20px rgba(38,91,84,.06);
        transition: transform .18s ease, box-shadow .18s ease;
      }

      .edm-page-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(38,91,84,.11);
      }

      .edm-page-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 50%;
        font-size: 1.55rem;
        background: linear-gradient(145deg, rgba(255,255,255,.94), var(--page-tint,#EAF6F0));
        border: 1px solid rgba(67,126,118,.14);
        box-shadow: 0 5px 12px rgba(38,91,84,.08);
      }
      .edm-page-card h3 { margin: .25rem 0 .12rem; font-size: .97rem; }
      .edm-page-card p {
        margin: 0;
        color: var(--edm-muted);
        font-size: .79rem;
        line-height: 1.35 !important;
      }

      .edm-sewer-story {
        overflow: hidden;
        padding: .6rem .7rem .35rem;
        margin: .5rem 0 .85rem;
        border: 1px solid rgba(52,114,105,.15);
        border-radius: 20px;
        background: linear-gradient(180deg,#EDF8FA 0%,#EAF6F0 55%,#E7F2E9 100%);
        box-shadow: 0 10px 26px rgba(38,91,84,.07);
      }

      .edm-sewer-story svg { width: 100%; height: auto; display: block; }
      .edm-flow-water {
        stroke-dasharray: 18 11;
        animation: edmPipeFlow 2.2s linear infinite;
      }
      .edm-flow-node {
        transition: transform .18s ease, filter .18s ease;
        transform-origin: center;
      }
      .edm-flow-node:hover {
        transform: translateY(-4px);
        filter: drop-shadow(0 5px 5px rgba(32,90,82,.18));
      }
      @keyframes edmPipeFlow { to { stroke-dashoffset: -58; } }

      div[data-testid="stPlotlyChart"] {
        padding: .35rem;
        border: 1px solid rgba(52,114,105,.12);
        border-radius: 18px;
        background: rgba(255,255,255,.58);
        box-shadow: 0 8px 22px rgba(38,91,84,.055);
      }

      @media (max-width: 1050px) {
        .edm-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .edm-page-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .edm-hero { grid-template-columns: 1fr; }
        .edm-water-art { min-height: 135px; }
      }

      @media (max-width: 680px) {
        .edm-metric-grid, .edm-journey, .edm-risk-guide { grid-template-columns: 1fr; }
        .edm-page-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .edm-hero { padding: 1.25rem; border-radius: 18px; }
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
      }

      @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
          animation-duration: .01ms !important;
          animation-iteration-count: 1 !important;
          scroll-behavior: auto !important;
        }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# DATA AND MODEL HELPERS
# =============================================================================

@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, compression="gzip")


@st.cache_resource(show_spinner=False)
def load_model_bundle():
    path = ARTIFACT_DIR / "final_trained_2026_forecast_model.joblib"
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_input_metadata():
    with open(ARTIFACT_DIR / "input_metadata.json", encoding="utf-8") as handle:
        return json.load(handle)


def pretty(value) -> str:
    return str(value).replace("_", " ").strip().title()


def safe_text(value, fallback="Not available") -> str:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return html.escape(fallback)
    return html.escape(str(value).strip())


def value_text(value, decimals=0, suffix="") -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "Not available"
    return f"{float(numeric):,.{decimals}f}{suffix}"


def available_values(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    return sorted(
        frame[column].dropna().astype(str).str.strip().loc[lambda s: s.ne("")].unique().tolist()
    )


def first_existing(frame: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((column for column in candidates if column in frame.columns), None)


def make_risk_ranking(
    frame: pd.DataFrame,
    risk_column: str,
    group_column: str | None,
) -> pd.DataFrame:
    """Create a plain-English ranking for the current map view."""
    if frame.empty or not group_column or group_column not in frame.columns:
        return pd.DataFrame()

    working = frame.loc[
        frame[risk_column].isin(RISK_ORDER)
        & frame[group_column].notna()
    ].copy()
    working[group_column] = working[group_column].astype(str).str.strip()
    working = working.loc[working[group_column].ne("")]
    if working.empty:
        return pd.DataFrame()

    counts = (
        working.groupby([group_column, risk_column])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=RISK_ORDER, fill_value=0)
        .reset_index()
    )
    counts["Mapped locations"] = counts[RISK_ORDER].sum(axis=1)

    spill_column = first_existing(
        working,
        [
            "place_total_counted_spills",
            "total_counted_spills_in_period",
            "counted_spills",
        ],
    )
    if spill_column:
        spill_values = working.assign(
            _spill_value=pd.to_numeric(working[spill_column], errors="coerce")
        )
        spill_group = spill_values.groupby(group_column, as_index=False)["_spill_value"]
        # Place totals are repeated on every outlet in the same place, whereas
        # outlet totals need summing for a water-company ranking.
        spills = (
            spill_group.max()
            if spill_column == "place_total_counted_spills"
            else spill_group.sum(min_count=1)
        ).rename(columns={"_spill_value": "Recorded spills"})
        counts = counts.merge(spills, on=group_column, how="left")
    else:
        counts["Recorded spills"] = np.nan

    counts["Risk score"] = (
        counts["High"] * 3
        + counts["Medium"] * 2
        + counts["Low"]
    )
    counts = counts.sort_values(
        ["High", "Medium", "Recorded spills", "Mapped locations", group_column],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    counts.insert(0, "Rank", np.arange(1, len(counts) + 1))
    return counts


def render_rank_list(
    ranking: pd.DataFrame,
    name_column: str,
    empty_message: str,
    limit: int = 8,
):
    if ranking.empty:
        st.info(empty_message)
        return

    rows = []
    for _, row in ranking.head(limit).iterrows():
        colour = (
            RISK_COLOURS["High"] if int(row.get("High", 0)) > 0
            else RISK_COLOURS["Medium"] if int(row.get("Medium", 0)) > 0
            else RISK_COLOURS["Low"]
        )
        spills = row.get("Recorded spills")
        spill_text = (
            f" · {float(spills):,.0f} recorded spills"
            if pd.notna(spills)
            else ""
        )
        rows.append(
            f"""
            <div class="edm-rank-row" style="--rank-colour:{colour};">
              <div><span class="edm-rank-number">{int(row['Rank'])}</span>
              <span class="edm-rank-name">{html.escape(str(row[name_column]))}</span></div>
              <div class="edm-rank-detail">
                ▲ High {int(row.get('High', 0)):,} &nbsp; ◆ Medium {int(row.get('Medium', 0)):,}
                &nbsp; ● Low {int(row.get('Low', 0)):,}{spill_text}
              </div>
            </div>
            """
        )
    st.html("".join(rows))


def render_risk_guide():
    st.html(
        """
        <div class="edm-risk-guide" aria-label="Plain-English risk guide">
          <div style="--risk-colour:#4A9C7D;"><b>&#9679; Low</b><br>Lower concern</div>
          <div style="--risk-colour:#E2A45C;"><b>&#9670; Medium</b><br>Closer attention</div>
          <div style="--risk-colour:#D66565;"><b>&#9650; High</b><br>Priority review</div>
        </div>
        """
    )


def kpi_value(frame: pd.DataFrame, phrase: str, fallback=np.nan):
    if frame.empty or not {"KPI", "Value"}.issubset(frame.columns):
        return fallback
    match = frame["KPI"].astype(str).str.contains(phrase, case=False, regex=False, na=False)
    return frame.loc[match, "Value"].iloc[0] if match.any() else fallback


def download_table(frame: pd.DataFrame, filename: str):
    st.download_button(
        "Download filtered evidence",
        data=frame.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        use_container_width=False,
    )


# =============================================================================
# REUSABLE VISUAL COMPONENTS
# =============================================================================

def render_hero():
    st.html(
        """
        <section class="edm-hero">
          <div>
            <div class="edm-kicker">💧 England's water and spill information</div>
            <h1>England's water and spill-risk story</h1>
            <p>
              Explore recorded locations, compare places and see the separate 2026 forecast.
            </p>
          </div>
          <div class="edm-water-art" aria-label="Animated clouds, rain and flowing river illustration">
            <svg viewBox="0 0 520 250" width="100%" role="img" aria-label="Clouds releasing rain above a flowing river and green leaves">
              <defs>
                <linearGradient id="riverGradient" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0" stop-color="#A9DCE7"/>
                  <stop offset=".5" stop-color="#68AFC2"/>
                  <stop offset="1" stop-color="#77C4B2"/>
                </linearGradient>
              </defs>
              <circle cx="265" cy="129" r="112" fill="#FFFFFF" opacity=".40"/>
              <g class="edm-cloud-a" fill="#F9FCFC" stroke="#9CCAD3" stroke-width="3">
                <ellipse cx="152" cy="70" rx="72" ry="30"/>
                <circle cx="124" cy="55" r="34"/>
                <circle cx="168" cy="46" r="42"/>
                <circle cx="205" cy="64" r="28"/>
              </g>
              <g class="edm-cloud-b" fill="#F5FAF8" stroke="#A8D8D0" stroke-width="3" opacity=".95">
                <ellipse cx="385" cy="88" rx="63" ry="27"/>
                <circle cx="356" cy="73" r="29"/>
                <circle cx="394" cy="64" r="37"/>
                <circle cx="426" cy="82" r="24"/>
              </g>
              <g stroke="#70B9CC" stroke-width="5" stroke-linecap="round">
                <line class="edm-rain-drop" x1="115" y1="98" x2="104" y2="120"/>
                <line class="edm-rain-drop" x1="154" y1="100" x2="143" y2="124"/>
                <line class="edm-rain-drop" x1="193" y1="98" x2="181" y2="122"/>
                <line class="edm-rain-drop" x1="358" y1="114" x2="347" y2="136"/>
                <line class="edm-rain-drop" x1="396" y1="114" x2="385" y2="138"/>
                <line class="edm-rain-drop" x1="428" y1="111" x2="417" y2="133"/>
              </g>
              <path d="M15 171 C78 137 126 151 174 178 C224 207 273 205 326 173 C380 140 437 146 507 179 L507 244 L15 244Z" fill="url(#riverGradient)" opacity=".55"/>
              <path class="edm-river-flow" d="M18 183 C81 151 129 161 176 188 C224 216 275 214 329 183 C383 151 441 156 505 189" fill="none" stroke="#F7FCFB" stroke-width="11" stroke-linecap="round"/>
              <path class="edm-river-flow" d="M20 215 C79 188 128 195 178 220 C226 244 277 241 327 213 C379 184 437 190 501 216" fill="none" stroke="#4D9EB5" stroke-width="7" stroke-linecap="round" opacity=".72"/>
              <path d="M55 168 C31 145 20 151 24 177 C44 184 57 178 55 168Z" fill="#7EBD91"/>
              <path d="M56 168 C60 140 76 132 92 153 C83 173 70 179 56 168Z" fill="#B6DEBF"/>
              <path d="M461 175 C479 147 498 149 502 173 C488 190 474 190 461 175Z" fill="#7EBD91"/>
            </svg>
          </div>
        </section>
        """,
    )


def render_page_cards():
    """Show every dashboard area on the landing page without long instructions."""
    st.html(
        """
        <div class="edm-page-grid" aria-label="Dashboard sections">
          <div class="edm-page-card" style="--page-tint:#E3F3F7;">
            <div class="edm-page-icon">🌧️</div><h3>Interactive maps</h3>
            <p>Recorded outlets and 2026 estimates.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#FBE4E4;">
            <div class="edm-page-icon">⛈️</div><h3>Priority locations</h3>
            <p>High-risk places, outlets and companies.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#E5F3EA;">
            <div class="edm-page-icon">🌦️</div><h3>Places &amp; companies</h3>
            <p>Simple rankings and yearly patterns.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#F0EAF6;">
            <div class="edm-page-icon">🌈</div><h3>2026 predictions</h3>
            <p>Forecast risks and affected locations.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#FFF0DD;">
            <div class="edm-page-icon">☁️💧</div><h3>Find a location</h3>
            <p>Search a site and view its probabilities.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#EDF3DE;">
            <div class="edm-page-icon">🌊</div><h3>Evidence</h3>
            <p>Sources, quality checks and limitations.</p>
          </div>
        </div>
        """
    )
    destinations = [
        ("Open map", "🌧️ Explore the map"),
        ("Priority list", "⛈️ Priority locations"),
        ("Compare", "🌦️ Places and companies"),
        ("2026 forecast", "🌈 2026 predictions"),
        ("Find a site", "☁️💧 Check one location"),
        ("Evidence", "🌊 About the evidence"),
    ]
    for column, (button_text, destination) in zip(st.columns(6), destinations):
        with column:
            st.button(
                button_text,
                key=f"home_{destination}",
                use_container_width=True,
                on_click=lambda target=destination: st.session_state.update(
                    main_navigation=target
                ),
            )


def render_sewer_story():
    """Compact animated explanation of a combined sewer and overflow route."""
    st.html(
        """
        <div class="edm-sewer-story">
          <svg viewBox="0 0 1200 355" role="img" aria-labelledby="sewer-title sewer-desc">
            <title id="sewer-title">How water moves through a combined sewer system</title>
            <desc id="sewer-desc">Rain and household wastewater enter one combined sewer. Normal flows travel to treatment. During heavy rain an overflow can release excess mixed water to a river.</desc>
            <defs>
              <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stop-color="#DDF2F7"/><stop offset="1" stop-color="#F8FCFA"/>
              </linearGradient>
              <linearGradient id="river" x1="0" y1="0" x2="1" y2="0">
                <stop offset="0" stop-color="#B8E1EA"/><stop offset="1" stop-color="#74B8C9"/>
              </linearGradient>
              <marker id="arrowBlue" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
                <path d="M0,0 L9,4.5 L0,9 z" fill="#4E9FB6"/>
              </marker>
              <marker id="arrowCoral" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto">
                <path d="M0,0 L9,4.5 L0,9 z" fill="#D27670"/>
              </marker>
            </defs>
            <rect width="1200" height="355" rx="20" fill="url(#sky)"/>
            <path d="M0 175 Q160 150 330 177 T650 172 T950 177 T1200 164 V355 H0Z" fill="#DCEEDB"/>

            <g class="edm-flow-node" transform="translate(35,18)">
              <g fill="#FAFDFD" stroke="#8EBFCC" stroke-width="3">
                <ellipse cx="105" cy="50" rx="75" ry="29"/><circle cx="72" cy="39" r="31"/>
                <circle cx="114" cy="30" r="39"/><circle cx="153" cy="45" r="27"/>
              </g>
              <g stroke="#68AFC2" stroke-width="5" stroke-linecap="round">
                <line class="edm-rain-drop" x1="67" y1="79" x2="56" y2="106"/>
                <line class="edm-rain-drop" x1="108" y1="80" x2="97" y2="108"/>
                <line class="edm-rain-drop" x1="150" y1="78" x2="139" y2="105"/>
              </g>
              <text x="105" y="132" text-anchor="middle" fill="#245B61" font-size="17" font-weight="700">Heavy rain</text>
            </g>

            <g class="edm-flow-node" transform="translate(230,82)">
              <path d="M0 80 L70 20 L140 80Z" fill="#D88477"/><rect x="18" y="78" width="105" height="90" rx="5" fill="#FFF7E8" stroke="#A98D72" stroke-width="3"/>
              <rect x="59" y="110" width="28" height="58" fill="#9FC7CE"/><rect x="28" y="98" width="25" height="25" fill="#B7DDE5"/>
              <path d="M105 102 Q138 114 126 148" fill="none" stroke="#68AFC2" stroke-width="6"/>
              <text x="70" y="194" text-anchor="middle" fill="#245B61" font-size="17" font-weight="700">Homes</text>
              <text x="70" y="214" text-anchor="middle" fill="#5D7772" font-size="13">wastewater + roof water</text>
            </g>

            <g class="edm-flow-node" transform="translate(410,154)">
              <rect x="0" y="34" width="118" height="62" rx="10" fill="#C7D2D3" stroke="#6E8585" stroke-width="3"/>
              <g stroke="#5B7070" stroke-width="6"><line x1="20" y1="44" x2="20" y2="86"/><line x1="48" y1="44" x2="48" y2="86"/><line x1="76" y1="44" x2="76" y2="86"/><line x1="104" y1="44" x2="104" y2="86"/></g>
              <text x="59" y="121" text-anchor="middle" fill="#245B61" font-size="17" font-weight="700">Road drains</text>
            </g>

            <path class="edm-flow-water" d="M350 245 C430 245 485 260 555 272" fill="none" stroke="#4E9FB6" stroke-width="8" marker-end="url(#arrowBlue)"/>
            <path class="edm-flow-water" d="M470 250 C510 252 535 260 565 272" fill="none" stroke="#4E9FB6" stroke-width="8"/>

            <g class="edm-flow-node" transform="translate(545,205)">
              <rect width="205" height="106" rx="20" fill="#D8E7E4" stroke="#4A837B" stroke-width="4"/>
              <path d="M18 54 C55 35 86 70 120 50 C151 32 171 61 190 49" fill="none" stroke="#4E9FB6" stroke-width="8"/>
              <text x="103" y="31" text-anchor="middle" fill="#173D3A" font-size="18" font-weight="800">Combined sewer</text>
              <text x="103" y="88" text-anchor="middle" fill="#5D7772" font-size="13">one pipe carries both flows</text>
            </g>

            <path class="edm-flow-water" d="M750 262 C805 262 835 234 875 219" fill="none" stroke="#4E9FB6" stroke-width="9" marker-end="url(#arrowBlue)"/>
            <g class="edm-flow-node" transform="translate(856,117)">
              <rect x="0" y="46" width="145" height="118" rx="14" fill="#F4F1E4" stroke="#7B9B87" stroke-width="4"/>
              <circle cx="42" cy="86" r="25" fill="#B7D9C6" stroke="#5C9076" stroke-width="3"/><circle cx="103" cy="86" r="25" fill="#B7D9C6" stroke="#5C9076" stroke-width="3"/>
              <rect x="27" y="120" width="91" height="27" rx="6" fill="#DDEADF"/>
              <text x="72" y="24" text-anchor="middle" fill="#245B61" font-size="17" font-weight="800">Treatment works</text>
              <text x="72" y="184" text-anchor="middle" fill="#5D7772" font-size="13">normal route</text>
            </g>

            <path d="M648 206 C710 140 785 122 1020 242" fill="none" stroke="#D27670" stroke-width="8" stroke-dasharray="15 10" marker-end="url(#arrowCoral)"/>
            <g class="edm-flow-node" transform="translate(714,67)">
              <rect width="155" height="63" rx="14" fill="#FFF0DD" stroke="#D09B58" stroke-width="3"/>
              <text x="78" y="26" text-anchor="middle" fill="#7A4C16" font-size="16" font-weight="800">Storm overflow</text>
              <text x="78" y="47" text-anchor="middle" fill="#7A6041" font-size="12">only when capacity is exceeded</text>
            </g>

            <path d="M994 251 C1058 221 1112 231 1200 255 V355 H972Z" fill="url(#river)"/>
            <path class="edm-river-flow" d="M992 275 C1052 248 1118 259 1190 284" fill="none" stroke="#EFFBFC" stroke-width="9"/>
            <g transform="translate(1080,292)" fill="none" stroke="#3C7F8F" stroke-width="3"><path d="M0 8 Q18 -8 36 8 Q18 26 0 8Z"/><circle cx="27" cy="6" r="2" fill="#3C7F8F"/></g>
            <text x="1090" y="215" text-anchor="middle" fill="#245B61" font-size="18" font-weight="800">River</text>
            <text x="1090" y="235" text-anchor="middle" fill="#5D7772" font-size="13">treated water or overflow route</text>
          </svg>
        </div>
        """
    )


def section_header(title: str, subtitle: str):
    st.html(
        f"""
        <div class="edm-section-header">
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
    )


def banner(text: str, icon="ℹ️", background=PALE_BLUE, edge="#68AFC2"):
    st.html(
        f"""
        <div class="edm-banner" style="--banner-bg:{background};--banner-edge:{edge};">
          <span aria-hidden="true">{icon}</span><div>{text}</div>
        </div>
        """,
    )


def metric_cards(cards: list[dict]):
    pieces = []
    for card in cards:
        pieces.append(
            f"""
            <div class="edm-metric-card" style="--card-accent:{card.get('accent', '#B9DFD1')};">
              <div class="edm-metric-label">{html.escape(str(card['label']))}</div>
              <div class="edm-metric-value">{html.escape(str(card['value']))}</div>
              <div class="edm-metric-note">{html.escape(str(card.get('note', '')))}</div>
            </div>
            """
        )
    # st.html renders the cards as HTML only. Using Markdown here can expose
    # indented <div> text as a visible code block on the public dashboard.
    st.html('<div class="edm-metric-grid">' + "".join(pieces) + "</div>")


def plot_style(figure: go.Figure, height=480):
    figure.update_layout(
        height=height,
        margin=dict(l=34, r=22, t=58, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.35)",
        font=dict(
            family="Atkinson Hyperlegible, Verdana, Arial, sans-serif",
            color=INK,
            size=12,
        ),
        title=dict(font=dict(color=INK, size=16), x=0.02, xanchor="left"),
        legend_title_text="",
        colorway=["#73B7AA", "#86BBD8", "#D2B6DD", "#F0C987", "#E7A3A3"],
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#B8D6CE",
            font_color=INK,
            font_size=12,
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,.58)",
            bordercolor="rgba(55,120,110,.12)",
            borderwidth=1,
        ),
    )
    figure.update_xaxes(
        gridcolor="rgba(68,120,110,0.10)",
        linecolor="rgba(68,120,110,0.18)",
        zeroline=False,
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor="rgba(68,120,110,0.10)",
        linecolor="rgba(68,120,110,0.18)",
        zeroline=False,
        automargin=True,
    )
    return figure


def risk_donut(frame: pd.DataFrame, risk_column: str, title: str) -> go.Figure:
    counts = (
        frame[risk_column]
        .astype("string")
        .value_counts()
        .reindex(RISK_ORDER, fill_value=0)
    )
    figure = go.Figure(
        go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.58,
            sort=False,
            marker=dict(
                colors=[RISK_COLOURS[label] for label in counts.index],
                line=dict(color="#FBFDF9", width=3),
            ),
            textinfo="label+percent",
            hovertemplate="%{label}<br>%{value:,} locations<br>%{percent}<extra></extra>",
        )
    )
    figure.update_layout(
        title=title,
        annotations=[dict(text=f"{int(counts.sum()):,}<br><span style='font-size:12px'>locations</span>", x=0.5, y=0.5, showarrow=False, font=dict(size=18, color=INK))],
        showlegend=False,
    )
    return plot_style(figure, height=420)


# =============================================================================
# FOLIUM MAP (CLUSTERED MARKERS + DENSITY VIEW)
# =============================================================================

def popup_for_row(row: pd.Series, risk_column: str, prediction: bool) -> str:
    """Build a readable map popup using the same evidence as the Colab map."""
    place = row.get("official_place_name", row.get("town_or_city", "Not available"))
    site = row.get("site_name", row.get("source_site_name_ea_consents_database", "Not available"))
    company = row.get("water_company_name", "Not available")
    receiving = row.get("receiving_water", row.get("source_receiving_water", "Not available"))
    catchment = row.get("catchment_name", row.get("catchment", "Not available"))
    grid = row.get("parsed_grid_reference", "Not available")
    permit = row.get("permit_reference", "Not available")
    years_observed = row.get("years_observed", "2023–2025")
    risk_history = row.get("risk_history", "Not available")
    risk = row.get(risk_column, "Not available")
    relationship = safe_text(row.get("official_place_relationship"))
    distance = value_text(row.get("distance_to_official_place_km"), 2, " km")
    x_coordinate = value_text(row.get("easting_x"), 1)
    y_coordinate = value_text(row.get("northing_y"), 1)
    annual_boxes = []
    for year in (2023, 2024, 2025):
        annual_value = row.get(f"place_counted_spills_{year}")
        annual_boxes.append(
            f"<div style='padding:5px;text-align:center;background:#F4FAF8;"
            f"border:1px solid #C8DDD7;border-radius:6px'><b>{year}</b><br>"
            f"{value_text(annual_value)} spills</div>"
        )
    annual_spills = "".join(annual_boxes)

    if prediction:
        observed = row.get("observed_2025_risk", row.get("actual_2025_risk_label", "Not available"))
        confidence = pd.to_numeric(pd.Series([row.get("prediction_confidence")]), errors="coerce").iloc[0]
        confidence_flag = row.get("confidence_flag", "Not available")
        probabilities = []
        for label, column in [("Low", "probability_low"), ("Medium", "probability_medium"), ("High", "probability_high")]:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            probabilities.append(f"{label}: {value:.1%}" if pd.notna(value) else f"{label}: not available")
        evidence_rows = f"""
          <div style="margin-top:8px;padding:9px;background:#FFF3DD;border-radius:9px;">
            <b>Recorded category in 2025:</b> {safe_text(observed)}<br>
            <b>Suggested category for 2026:</b> {safe_text(risk)}<br>
            <b>Chances:</b> {' · '.join(probabilities)}<br>
            <b>Certainty:</b> {f'{confidence:.1%}' if pd.notna(confidence) else 'Not available'}<br>
            <b>Check needed:</b> {safe_text(confidence_flag)}
          </div>
        """
    else:
        spills = row.get(
            "total_counted_spills_in_period",
            row.get("counted_spills", row.get("counted_spills_using_12_24h_count_method", "Not available")),
        )
        duration = row.get(
            "total_spill_duration_hours_in_period",
            row.get("total_duration_hours", "Not available"),
        )
        place_spills = row.get("place_total_counted_spills", np.nan)
        place_high = row.get("place_high_risk_locations", np.nan)
        place_medium = row.get("place_medium_risk_locations", np.nan)
        place_low = row.get("place_low_risk_locations", np.nan)
        place_summary = ""
        if any(pd.notna(value) for value in [place_high, place_medium, place_low]):
            place_summary = (
                f"<br><b>Risk around this town/city:</b> High {value_text(place_high)}, "
                f"Medium {value_text(place_medium)}, Low {value_text(place_low)}"
            )
        evidence_rows = f"""
          <div style="margin-top:8px;padding:9px;background:#EAF6F0;border-radius:9px;">
            <b>Risk shown on this map:</b> {safe_text(risk)}<br>
            <b>Years covered:</b> {safe_text(years_observed)}<br>
            <b>Counted spills for this outlet:</b> {value_text(spills)}<br>
            <b>Recorded duration:</b> {value_text(duration, 1, ' hours')}<br>
            <div style="margin:6px 0 3px"><b>Town/city counted spills by year</b></div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px">{annual_spills}</div>
            <b>Town/city total:</b> {value_text(place_spills)}
            {place_summary}
          </div>
        """

    return f"""
      <div style="font-family:Verdana,Arial,sans-serif;width:330px;color:#173D3A;line-height:1.58;font-size:13px;">
        <div style="font-size:16px;font-weight:800;margin:-1px -1px 8px;padding:9px 10px;
                    border-radius:8px;background:#E9F4F8;">{safe_text(site)}</div>
        <b>Town/city:</b> {safe_text(place)}<br>
        <b>Place match:</b> {relationship} · {distance}<br>
        <b>Water company:</b> {safe_text(company)}<br>
        <b>Receiving water:</b> {safe_text(receiving)}<br>
        <b>Catchment:</b> {safe_text(catchment)}<br>
        <b>Risk history:</b> {safe_text(risk_history)}<br>
        <b>Permit reference:</b> {safe_text(permit)}<br>
        <b>Grid reference:</b> {safe_text(grid)}
        {evidence_rows}
        <div style="margin-top:7px;font-size:11px;color:#5D7772;">
          Coordinates: {float(row['latitude']):.5f}, {float(row['longitude']):.5f}<br>
          National Grid X/Y: {x_coordinate}, {y_coordinate}
        </div>
      </div>
    """


def add_colab_map_panels(
    water_map: folium.Map,
    plotting: pd.DataFrame,
    risk_column: str,
    prediction: bool,
) -> None:
    """Add the same directory-and-ranking experience used by the Colab map."""
    if plotting.empty:
        return

    def plain(value, fallback="Not available"):
        if value is None or pd.isna(value) or str(value).strip() == "":
            return fallback
        return str(value).strip()

    place_column = first_existing(plotting, ["official_place_name", "town_or_city"])
    company_column = first_existing(plotting, ["water_company_name", "company"])
    site_column = first_existing(
        plotting,
        ["site_name", "source_site_name_ea_consents_database"],
    )
    if not place_column or not company_column:
        return

    company_ranking = make_risk_ranking(plotting, risk_column, company_column)
    company_trends = load_table("company_spill_trends")
    trend_lookup = {}
    if not company_trends.empty and {
        "water_company_name",
        "reporting_year",
        "counted_spills",
    }.issubset(company_trends.columns):
        for company_name, company_rows in company_trends.groupby("water_company_name"):
            company_rows = company_rows.copy()
            company_rows["reporting_year"] = pd.to_numeric(
                company_rows["reporting_year"], errors="coerce"
            )
            yearly = []
            for year in (2023, 2024, 2025):
                match = company_rows.loc[company_rows["reporting_year"].eq(year)]
                count_value = (
                    pd.to_numeric(match["counted_spills"], errors="coerce").iloc[0]
                    if not match.empty
                    else np.nan
                )
                duration_value = (
                    pd.to_numeric(match["spill_duration_hours"], errors="coerce").iloc[0]
                    if not match.empty and "spill_duration_hours" in match.columns
                    else np.nan
                )
                yearly.append(
                    {
                        "year": year,
                        "count": None if pd.isna(count_value) else round(float(count_value), 1),
                        "duration": None if pd.isna(duration_value) else round(float(duration_value), 1),
                    }
                )
            trend_lookup[str(company_name)] = yearly

    ranking_rows = []
    for _, row in company_ranking.head(12).iterrows():
        company_name = str(row[company_column])
        ranking_rows.append(
            f"""
            <button type="button" class="edm-map-rank edm-company-trend-button"
                    data-company="{html.escape(company_name, quote=True)}"
                    aria-label="Show the 2023 to 2025 spill trend for {html.escape(company_name, quote=True)}">
              <span class="edm-map-rank-number">{int(row['Rank'])}</span>
              <b>{html.escape(company_name)}</b>
              <div><span class="risk-high">&#9650; {int(row.get('High', 0)):,}</span>
              <span class="risk-medium">&#9670; {int(row.get('Medium', 0)):,}</span>
              <span class="risk-low">&#9679; {int(row.get('Low', 0)):,}</span></div>
              <small>View 2023–2025 spill trend</small>
            </button>
            """
        )

    risk_counts = plotting[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
    map_title = "Predicted 2026 risk" if prediction else "Observed spill risk"
    period_text = "Forecast - not a confirmed event" if prediction else "Recorded 2023-2025 evidence"
    place_detail_label = "Forecast status" if prediction else "Recorded spills"
    panels = f"""
    <style>
      .edm-map-panel {{position:fixed;top:12px;z-index:9999;width:300px;max-height:86vh;
        overflow:auto;padding:12px;background:rgba(251,253,249,.96);color:#173D3A;
        border:1px solid #AFCFC6;border-radius:14px;box-shadow:0 7px 25px rgba(28,77,70,.19);
        font:12px/1.38 'Atkinson Hyperlegible',Verdana,Arial,sans-serif;}}
      #edm-map-left {{left:12px;height:calc(86vh - 24px);overflow:hidden;display:flex;flex-direction:column;}}
      #edm-map-right {{right:12px;top:12px;bottom:auto;}}
      .edm-map-title {{margin:-12px -12px 8px;padding:10px 12px;border-radius:13px 13px 0 0;
        color:#173D3A;background:linear-gradient(120deg,#CFEAE3,#DDEFF4);font-size:16px;font-weight:800;}}
      .edm-map-period {{margin:5px 0 8px;padding:5px 7px;border-radius:8px;background:#FFFFFF;
        color:#446862;font-size:11px;font-weight:700;}}
      .edm-map-legend {{display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin:7px 0;}}
      .edm-map-legend div {{padding:6px 3px;border-radius:8px;text-align:center;background:#FFFFFF;font-weight:700;}}
      .edm-map-panel label {{display:block;margin:5px 0 2px;font-weight:700;color:#365F5B;}}
      .edm-map-panel input,.edm-map-panel select {{width:100%;box-sizing:border-box;padding:7px 8px;
        border:1px solid #B7CEC8;border-radius:8px;background:#FFFFFF;color:#173D3A;font-size:12px;}}
      .edm-map-filter-row {{display:grid;grid-template-columns:1fr 1fr;gap:6px;}}
      #edm-place-count {{padding:6px 1px 3px;font-weight:700;}}
      #edm-place-results {{flex:1;min-height:230px;overflow:auto;margin-top:2px;padding-right:2px;
        border-top:1px solid #D4E5DF;}}
      .edm-place-letter {{position:sticky;top:0;padding:3px 7px;background:#DDEFF4;color:#245B61;
        font-weight:800;border-radius:6px;}}
      .edm-place-button {{display:block;width:100%;margin:5px 0;padding:7px 8px;text-align:left;
        color:#173D3A;background:#FFFFFF;border:1px solid #D1E1DC;border-left:6px solid var(--place-risk);
        border-radius:9px;cursor:pointer;font-size:12px;}}
      .edm-place-button:hover,.edm-place-button:focus {{background:#EDF8F5;transform:translateX(2px);}}
      .edm-place-name {{display:block;font-size:13px;font-weight:800;margin-bottom:2px;}}
      .edm-place-detail {{color:#5D7772;font-size:11px;}}
      .edm-map-rank {{display:block;width:100%;margin:5px 0;padding:7px;text-align:left;color:#173D3A;
        border:1px solid #D5E5E0;border-left:5px solid #68AFC2;border-radius:9px;background:#FFFFFF;
        cursor:pointer;font:12px/1.38 'Atkinson Hyperlegible',Verdana,Arial,sans-serif;}}
      .edm-map-rank:hover,.edm-map-rank:focus {{background:#EAF6F0;transform:translateX(-2px);
        box-shadow:0 4px 10px rgba(35,89,81,.12);}}
      .edm-map-rank-number {{display:inline-flex;align-items:center;justify-content:center;width:23px;
        height:23px;margin-right:5px;border-radius:50%;background:#DDEFF4;color:#245B61;font-weight:800;}}
      .edm-map-rank div {{margin:3px 0 0 29px;font-size:11px;word-spacing:5px;}}
      .edm-map-rank small {{display:block;margin:4px 0 0 29px;color:#47716A;font-weight:700;}}
      #edm-company-trend {{display:none;margin:7px 0 9px;padding:9px;border:1px solid #BFD8D0;
        border-radius:11px;background:linear-gradient(160deg,#F7FCFA,#EAF6F0);}}
      #edm-company-trend h4 {{margin:0 0 3px;font-size:13px;color:#173D3A;}}
      .edm-trend-status {{display:inline-block;margin:2px 0 7px;padding:3px 7px;border-radius:999px;
        background:#FFFFFF;color:#365F5B;font-size:11px;font-weight:800;}}
      .edm-trend-bars {{display:flex;align-items:flex-end;justify-content:space-around;height:126px;
        padding:7px 3px 0;border-bottom:1px solid #9BBDB4;}}
      .edm-trend-year {{display:flex;flex-direction:column;align-items:center;justify-content:flex-end;
        width:30%;height:100%;font-size:10px;color:#365F5B;}}
      .edm-trend-value {{margin-bottom:3px;font-weight:800;color:#173D3A;}}
      .edm-trend-bar {{width:34px;min-height:3px;border-radius:7px 7px 2px 2px;
        background:linear-gradient(180deg,#68AFC2,#4A9C7D);}}
      .edm-trend-year-label {{margin-top:4px;font-weight:800;}}
      .edm-trend-duration {{margin-top:7px;padding:6px;border-radius:7px;background:#FFFFFF;
        color:#52716C;font-size:10px;line-height:1.55;}}
      .risk-high {{color:#A84B4B;font-weight:800;}} .risk-medium {{color:#93611D;font-weight:800;}}
      .risk-low {{color:#357A63;font-weight:800;}}
      @media(max-width:1000px) {{.edm-map-panel{{width:235px;max-height:86vh;}}
        #edm-map-left{{height:calc(86vh - 24px);}}
        #edm-map-right{{top:12px;bottom:auto;}}}}
    </style>
    <aside id="edm-map-left" class="edm-map-panel" aria-label="Town and city directory">
      <div class="edm-map-title">{map_title}</div>
      <div class="edm-map-period">{period_text}</div>
      <div class="edm-map-legend">
        <div style="color:#357A63;">&#9679; Low<br>{int(risk_counts['Low']):,}</div>
        <div style="color:#93611D;">&#9670; Medium<br>{int(risk_counts['Medium']):,}</div>
        <div style="color:#A84B4B;">&#9650; High<br>{int(risk_counts['High']):,}</div>
      </div>
      <label for="edm-place-search">Find a town or city</label>
      <input id="edm-place-search" type="search" placeholder="Type a name or browse below">
      <div class="edm-map-filter-row">
        <div><label for="edm-risk-filter">Risk</label>
        <select id="edm-risk-filter"><option value="">All risks</option><option>High</option><option>Medium</option><option>Low</option></select></div>
        <div><label for="edm-company-filter">Water company</label>
        <select id="edm-company-filter"><option value="">All companies</option></select></div>
      </div>
      <div id="edm-place-count" class="edm-place-detail">Loading the complete place list...</div>
      <div id="edm-place-results"></div>
    </aside>
    <aside id="edm-map-right" class="edm-map-panel" aria-label="Water company ranking">
      <div class="edm-map-title">Water-company ranking</div>
      <div class="edm-map-period">High-risk locations first</div>
      <div class="edm-place-detail" style="margin-bottom:6px;">Select a company to view its recorded spill trend.</div>
      <div id="edm-company-trend" aria-live="polite"></div>
      {''.join(ranking_rows)}
    </aside>
    """
    water_map.get_root().html.add_child(folium.Element(panels))

    directory = []
    for _, row in plotting.iterrows():
        risk = plain(row.get(risk_column), "Uncategorised")
        directory.append(
            {
                "lat": round(float(row["latitude"]), 6),
                "lon": round(float(row["longitude"]), 6),
                "risk": risk,
                "colour": RISK_COLOURS.get(risk, "#78909C"),
                "company": plain(row.get(company_column), "Unknown company"),
                "place": plain(row.get(place_column), "Place unavailable"),
                "site": plain(row.get(site_column), "Spill outlet") if site_column else "Spill outlet",
                "spills": (
                    "Model-generated 2026 risk"
                    if prediction
                    else value_text(
                        row.get(
                            "place_total_counted_spills",
                            row.get("total_counted_spills_in_period"),
                        )
                    )
                ),
            }
        )
    directory_json = json.dumps(directory, ensure_ascii=True, separators=(",", ":")).replace("</", "<\\/")
    company_trend_json = json.dumps(
        trend_lookup,
        ensure_ascii=True,
        separators=(",", ":"),
    ).replace("</", "<\\/")
    map_name = water_map.get_name()
    script = f"""
    var edmSites={directory_json};
    var edmCompanyTrends={company_trend_json};
    var edmMap={map_name};
    var edmFocusMarker=null;
    function edmEscape(value){{var n=document.createElement('div');n.textContent=value||'';return n.innerHTML;}}
    function edmNumber(value,decimals){{
      if(value===null||value===undefined||Number.isNaN(Number(value)))return 'Not reported';
      return Number(value).toLocaleString(undefined,{{minimumFractionDigits:decimals,maximumFractionDigits:decimals}});
    }}
    function edmShowCompanyTrend(company){{
      var root=document.getElementById('edm-company-trend');
      var rows=edmCompanyTrends[company];
      root.style.display='block';
      if(!rows||!rows.length){{
        root.innerHTML='<h4>'+edmEscape(company)+'</h4><div class="edm-place-detail">The annual spill-count export is not available for this company.</div>';
        root.scrollIntoView({{block:'nearest',behavior:'smooth'}});
        return;
      }}
      var valid=rows.filter(function(row){{return row.count!==null;}});
      var maximum=Math.max.apply(null,valid.map(function(row){{return Number(row.count);}}).concat([1]));
      var status='Annual direction unavailable';
      if(valid.length>1){{
        var change=Number(valid[valid.length-1].count)-Number(valid[0].count);
        status=change>0?'Increased by '+edmNumber(change,0)+' counted spills':
          (change<0?'Decreased by '+edmNumber(Math.abs(change),0)+' counted spills':'No overall change');
      }}
      var bars=rows.map(function(row){{
        var height=row.count===null?3:Math.max(3,Math.round(100*Number(row.count)/maximum));
        return '<div class="edm-trend-year"><span class="edm-trend-value">'+edmNumber(row.count,0)+
          '</span><span class="edm-trend-bar" style="height:'+height+'%"></span><span class="edm-trend-year-label">'+
          row.year+'</span></div>';
      }}).join('');
      var durations=rows.map(function(row){{return '<b>'+row.year+':</b> '+edmNumber(row.duration,1)+' hours';}}).join(' &nbsp; ');
      root.innerHTML='<h4>'+edmEscape(company)+'</h4><div class="edm-place-detail">Recorded counted spills</div>'+ 
        '<div class="edm-trend-status">'+edmEscape(status)+'</div><div class="edm-trend-bars">'+bars+'</div>'+ 
        '<div class="edm-trend-duration"><b>Recorded duration</b><br>'+durations+'</div>'+ 
        '<div class="edm-place-detail" style="margin-top:5px">The 2023–2025 trend is recorded evidence; any 2026 category remains a forecast.</div>';
      root.scrollIntoView({{block:'nearest',behavior:'smooth'}});
    }}
    function edmBuildPlaces(){{
      var query=document.getElementById('edm-place-search').value.toLowerCase().trim();
      var risk=document.getElementById('edm-risk-filter').value;
      var company=document.getElementById('edm-company-filter').value;
      var groups=new Map();
      edmSites.forEach(function(site){{
        if((risk&&site.risk!==risk)||(company&&site.company!==company))return;
        if(query&&site.place.toLowerCase().indexOf(query)===-1)return;
        if(!groups.has(site.place))groups.set(site.place,{{name:site.place,sites:[],companies:new Set(),high:0,medium:0,low:0,spills:site.spills}});
        var place=groups.get(site.place);place.sites.push(site);place.companies.add(site.company);
        if(site.risk==='High')place.high++;else if(site.risk==='Medium')place.medium++;else if(site.risk==='Low')place.low++;
      }});
      return Array.from(groups.values()).sort(function(a,b){{return a.name.localeCompare(b.name);}});
    }}
    function edmFocusPlace(place){{
      var coords=place.sites.map(function(s){{return[s.lat,s.lon];}});
      if(coords.length===1)edmMap.setView(coords[0],14);else edmMap.fitBounds(L.latLngBounds(coords).pad(.16),{{maxZoom:13}});
      if(edmFocusMarker)edmMap.removeLayer(edmFocusMarker);
      var centre=coords.reduce(function(t,c){{t[0]+=c[0];t[1]+=c[1];return t;}},[0,0]);
      centre=[centre[0]/coords.length,centre[1]/coords.length];
      edmFocusMarker=L.circleMarker(centre,{{radius:10,color:'#173D3A',weight:3,fillColor:'#DDEFF4',fillOpacity:.95}}).addTo(edmMap);
      var popup='<div style="font:12px/1.45 Verdana;color:#173D3A;min-width:245px"><b style="font-size:14px">'+edmEscape(place.name)+'</b><br>'+coords.length.toLocaleString()+' mapped outlets<br><b>Risk:</b> High '+place.high+', Medium '+place.medium+', Low '+place.low+'<br><b>Companies:</b> '+edmEscape(Array.from(place.companies).sort().join(', '))+'<br><b>{place_detail_label}:</b> '+edmEscape(place.spills)+'</div>';
      edmFocusMarker.bindPopup(popup,{{maxWidth:330}}).openPopup();
    }}
    function edmRenderPlaces(){{
      var places=edmBuildPlaces();document.getElementById('edm-place-count').textContent=places.length.toLocaleString()+' places shown';
      var root=document.getElementById('edm-place-results');root.replaceChildren();var previous='';
      if(!places.length){{
        root.innerHTML='<div style="margin:8px 0;padding:10px;border-radius:9px;background:#FFF0DD;color:#704C1D">No matching town or city. Clear the search or change the filters.</div>';
        return;
      }}
      places.forEach(function(place){{
        var letter=(place.name.charAt(0)||'#').toUpperCase();if(letter!==previous){{var h=document.createElement('div');h.className='edm-place-letter';h.textContent=letter;root.appendChild(h);previous=letter;}}
        var button=document.createElement('button');button.type='button';button.className='edm-place-button';
        button.style.setProperty('--place-risk',place.high?'#D66565':(place.medium?'#E2A45C':'#4A9C7D'));
        button.innerHTML='<span class="edm-place-name">'+edmEscape(place.name)+'</span><span class="edm-place-detail">&#9650; '+place.high+' &nbsp; &#9670; '+place.medium+' &nbsp; &#9679; '+place.low+' · '+place.sites.length+' outlets</span>';
        button.addEventListener('click',function(){{edmFocusPlace(place);}});root.appendChild(button);
      }});
    }}
    var companySelect=document.getElementById('edm-company-filter');
    Array.from(new Set(edmSites.map(function(s){{return s.company;}}))).sort().forEach(function(company){{var o=document.createElement('option');o.value=company;o.textContent=company;companySelect.appendChild(o);}});
    document.getElementById('edm-place-search').addEventListener('input',edmRenderPlaces);
    document.getElementById('edm-risk-filter').addEventListener('change',edmRenderPlaces);
    document.getElementById('edm-company-filter').addEventListener('change',edmRenderPlaces);
    document.querySelectorAll('.edm-company-trend-button').forEach(function(button){{
      button.addEventListener('click',function(){{edmShowCompanyTrend(button.dataset.company);}});
    }});
    window.setTimeout(edmRenderPlaces,0);
    """
    water_map.get_root().script.add_child(folium.Element(script))


def build_folium_map(
    frame: pd.DataFrame,
    risk_column: str,
    prediction: bool,
    display_style: str,
) -> folium.Map:
    plotting = frame.copy()
    plotting["latitude"] = pd.to_numeric(plotting["latitude"], errors="coerce")
    plotting["longitude"] = pd.to_numeric(plotting["longitude"], errors="coerce")
    plotting = plotting.dropna(subset=["latitude", "longitude"])
    plotting = plotting.loc[plotting[risk_column].isin(RISK_ORDER)].copy()

    centre = [52.85, -1.45]
    if not plotting.empty:
        centre = [float(plotting["latitude"].median()), float(plotting["longitude"].median())]

    water_map = folium.Map(
        location=centre,
        zoom_start=6,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer(
        tiles="CartoDB positron",
        name="Pastel light map",
        show=True,
        control=True,
    ).add_to(water_map)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Detailed street map",
        show=False,
        control=True,
    ).add_to(water_map)

    if display_style == "Risk concentration":
        heat_values = plotting[["latitude", "longitude", risk_column]].copy()
        heat_values["weight"] = heat_values[risk_column].map(
            {"Low": 1.0, "Medium": 2.0, "High": 3.5}
        )
        HeatMap(
            heat_values[["latitude", "longitude", "weight"]].to_numpy().tolist(),
            name="Spill-risk concentration",
            radius=13,
            blur=17,
            min_opacity=0.28,
            gradient={0.20: "#B7DDE5", 0.48: "#7BC6B5", 0.72: "#E8C77C", 1.0: "#D66565"},
        ).add_to(water_map)
    else:
        callback = """
        function (row) {
          const colours = {Low: '#4A9C7D', Medium: '#E2A45C', High: '#D66565'};
          const marker = L.circleMarker([row[0], row[1]], {
            radius: 6.5,
            color: '#FFFFFF',
            weight: 1.4,
            fillColor: colours[row[2]] || '#78909C',
            fillOpacity: 0.92
          });
          marker.bindPopup(row[3], {maxWidth: 365});
          marker.bindTooltip(row[4], {direction: 'top', opacity: 0.96});
          return marker;
        }
        """
        for risk in ["High", "Medium", "Low"]:
            risk_rows = plotting.loc[plotting[risk_column].eq(risk)]
            if risk_rows.empty:
                continue
            cluster_data = []
            for _, row in risk_rows.iterrows():
                place = row.get("official_place_name", row.get("town_or_city", "Unknown place"))
                site = row.get("site_name", "Spill outlet")
                tooltip = f"{RISK_SYMBOLS[risk]} {risk} risk · {site} · {place}"
                cluster_data.append(
                    [
                        float(row["latitude"]),
                        float(row["longitude"]),
                        risk,
                        popup_for_row(row, risk_column, prediction),
                        safe_text(tooltip),
                    ]
                )
            layer = folium.FeatureGroup(
                name=f"{RISK_SYMBOLS[risk]} {risk} risk · {len(risk_rows):,} locations",
                show=True,
            )
            FastMarkerCluster(
                data=cluster_data,
                callback=callback,
                options={"maxClusterRadius": 35, "disableClusteringAtZoom": 11},
            ).add_to(layer)
            layer.add_to(water_map)

    Fullscreen(position="topright", title="Open full-screen map", title_cancel="Exit full screen").add_to(water_map)
    MeasureControl(position="topright", primary_length_unit="kilometers").add_to(water_map)
    MiniMap(toggle_display=True, minimized=True, position="bottomright").add_to(water_map)
    folium.LayerControl(collapsed=True, position="topright").add_to(water_map)

    if display_style == "Clustered spill locations":
        add_colab_map_panels(
            water_map,
            plotting,
            risk_column,
            prediction,
        )
    else:
        risk_counts = plotting[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
        legend = f"""
        <div style="position:fixed;left:18px;bottom:28px;z-index:9999;
                    background:rgba(255,255,255,.94);border:1px solid #BFD6CF;
                    border-radius:12px;padding:10px 13px;color:#173D3A;
                    box-shadow:0 5px 18px rgba(34,82,75,.16);font:12px Arial;">
          <div style="font-weight:700;margin-bottom:6px;">Risk category</div>
          <div><span style="color:#4A9C7D;font-size:17px;">●</span> Low · {int(risk_counts['Low']):,}</div>
          <div><span style="color:#E2A45C;font-size:16px;">◆</span> Medium · {int(risk_counts['Medium']):,}</div>
          <div><span style="color:#D66565;font-size:15px;">▲</span> High · {int(risk_counts['High']):,}</div>
        </div>
        """
        water_map.get_root().html.add_child(folium.Element(legend))

    if len(plotting) > 1:
        water_map.fit_bounds(
            [
                [float(plotting["latitude"].min()), float(plotting["longitude"].min())],
                [float(plotting["latitude"].max()), float(plotting["longitude"].max())],
            ],
            padding=(18, 18),
        )

    return water_map


def filter_map(frame: pd.DataFrame, risk_column: str, prediction: bool) -> tuple[pd.DataFrame, str]:
    filtered = frame.copy()
    place_column = first_existing(filtered, ["official_place_name", "town_or_city"])
    company_column = first_existing(filtered, ["water_company_name", "company"])

    filter_columns = st.columns([1.05, 1.05, 1.05, 1.35])
    with filter_columns[0]:
        risk_choices = st.multiselect(
            "Show these risk levels",
            RISK_ORDER,
            default=RISK_ORDER,
            key=f"{'forecast' if prediction else 'observed'}_risk",
        )
    with filter_columns[1]:
        company_options = ["All companies"] + available_values(filtered, company_column) if company_column else ["All companies"]
        company = st.selectbox("Choose a water company", company_options, key=f"{'forecast' if prediction else 'observed'}_company")
    with filter_columns[2]:
        place_options = ["All towns/cities"] + available_values(filtered, place_column) if place_column else ["All towns/cities"]
        place = st.selectbox("Choose a town or city", place_options, key=f"{'forecast' if prediction else 'observed'}_place")
    with filter_columns[3]:
        search = st.text_input(
            "Search for a site, permit or river",
            key=f"{'forecast' if prediction else 'observed'}_search",
        ).strip()

    if risk_choices:
        filtered = filtered.loc[filtered[risk_column].isin(risk_choices)]
    else:
        filtered = filtered.iloc[0:0]
    if company_column and company != "All companies":
        filtered = filtered.loc[filtered[company_column].astype(str).eq(company)]
    if place_column and place != "All towns/cities":
        filtered = filtered.loc[filtered[place_column].astype(str).eq(place)]
    if search:
        searchable = [
            column for column in [
                "site_name", "source_site_name_ea_consents_database", "permit_reference",
                "receiving_water", "source_receiving_water", "catchment", "catchment_name",
                "parsed_grid_reference",
            ] if column in filtered.columns
        ]
        match = pd.Series(False, index=filtered.index)
        for column in searchable:
            match |= filtered[column].astype("string").str.contains(search, case=False, regex=False, na=False)
        filtered = filtered.loc[match]

    if not prediction and "reporting_year" in filtered.columns:
        years = sorted(pd.to_numeric(filtered["reporting_year"], errors="coerce").dropna().astype(int).unique().tolist())
        if years:
            selected_years = st.multiselect("Observed reporting year", years, default=years, key="observed_years")
            filtered = filtered.loc[pd.to_numeric(filtered["reporting_year"], errors="coerce").isin(selected_years)]

    return filtered, place_column or "town_or_city"


# =============================================================================
# MODEL PROBABILITY HELPERS
# =============================================================================

def aligned_probabilities(model, values):
    raw = np.asarray(model.predict_proba(values), dtype=float)
    aligned = np.zeros((len(values), 3), dtype=float)
    for position, class_code in enumerate(model.classes_):
        aligned[:, int(class_code)] = raw[:, position]
    return aligned


def probability_logit(probabilities):
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def apply_calibrators(raw, calibrators):
    calibrated = np.zeros_like(raw, dtype=float)
    for class_code, calibrator in enumerate(calibrators):
        if calibrator is None:
            calibrated[:, class_code] = raw[:, class_code]
        else:
            calibrated[:, class_code] = calibrator.predict_proba(
                probability_logit(raw[:, class_code])
            )[:, 1]
    totals = calibrated.sum(axis=1, keepdims=True)
    invalid = ~np.isfinite(totals[:, 0]) | (totals[:, 0] <= 0)
    calibrated[invalid] = raw[invalid]
    totals = calibrated.sum(axis=1, keepdims=True)
    return calibrated / totals


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

st.sidebar.html(
    """
    <div class="edm-brand">
      <div style="display:flex;align-items:center;">
        <span class="edm-brand-mark">💧</span>
        <div><div style="font-size:1.18rem;font-weight:800;">EDM Observatory</div>
        <div style="font-size:.78rem;color:#5D7772;">England · evidence · forecast</div></div>
      </div>
    </div>
    """,
)

st.sidebar.markdown("**Reading options**")
larger_text = st.sidebar.toggle("Use larger writing", value=False)
reduce_motion = st.sidebar.toggle("Stop moving illustrations", value=False)

if larger_text:
    st.html(
        """
        <style>
          html, body, [class*="css"] { font-size: 19px !important; }
          .edm-metric-note, .edm-rank-detail { font-size: .92rem !important; }
        </style>
        """
    )

if reduce_motion:
    st.html(
        """
        <style>
          *, *::before, *::after {
            animation: none !important;
            transition: none !important;
          }
        </style>
        """
    )

PAGES = [
    "🌤️ Start here",
    "🌧️ Explore the map",
    "⛈️ Priority locations",
    "🌦️ Places and companies",
    "🌈 2026 predictions",
    "☁️💧 Check one location",
    "🌊 About the evidence",
]

page_label = st.sidebar.radio("Choose a page", PAGES)
page = page_label.split(" ", 1)[1]

# Every area is visible in a compact top navigation row. The sidebar remains
# available for optional reading controls and is collapsed by default.
st.html(
    """
    <div style="display:flex;align-items:center;gap:.65rem;margin:.1rem 0 .25rem;">
      <span class="edm-brand-mark" style="width:36px;height:36px;font-size:19px;margin:0;">🌧️</span>
      <div><b style="font-size:1.02rem;">EDM Water &amp; Spill-Risk Observatory</b>
      <div style="font-size:.76rem;color:#5D7772;">England · recorded evidence · 2026 forecast</div></div>
    </div>
    """
)
page_label = st.radio(
    "Dashboard sections",
    PAGES,
    horizontal=True,
    label_visibility="collapsed",
    key="main_navigation",
)
page = page_label.split(" ", 1)[1]

st.sidebar.markdown("---")
st.sidebar.html(
    """
    <div class="edm-access-note">
      <b style="color:#173D3A;">What does “2026 forecast” mean?</b><br>
      It is the system's best estimate from earlier information. It is not a known
      result and does not prove that a spill will happen.
    </div>
    """,
)


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

if page == "Start here":
    render_hero()
    render_page_cards()

    observed = load_table("observed_locations")
    forecast = load_table("forecast_map_points")
    observed_kpis = load_table("observed_kpis")
    forecast_kpis = load_table("forecast_kpis")

    observed_high = int(observed.get("period_risk_category", pd.Series(dtype=str)).eq("High").sum())
    predicted_high = int(forecast.get("predicted_2026_risk", pd.Series(dtype=str)).eq("High").sum())
    review_count = int(forecast.get("confidence_flag", pd.Series(dtype=str)).ne("Higher confidence").sum()) if "confidence_flag" in forecast.columns else 0

    metric_cards(
        [
            {"label": "Locations on the map", "value": value_text(len(observed)), "note": "Recorded information from 2023–2025", "accent": "#A8D8D0"},
            {"label": "Highest-concern locations", "value": value_text(observed_high), "note": "Places currently shown as High risk", "accent": "#E9A7A7"},
            {"label": "Locations with a 2026 estimate", "value": value_text(len(forecast)), "note": "Known sites the system could assess", "accent": "#B7DDE5"},
            {"label": "Estimates needing a closer look", "value": value_text(review_count), "note": "The answer was less certain", "accent": "#F1D39D"},
        ]
    )

    banner(
        "<b>Recorded 2023–2025</b> means information already supplied for those years. "
        "<b>2026 forecast</b> means what the system estimates may happen next. "
        "A forecast is never presented as a known event.",
        icon="🛟",
        background=PALE_AMBER,
        edge="#D59A3C",
    )

    section_header("Risk at a glance", "Recorded evidence and forecasts always remain separate.")
    render_risk_guide()

    left, right = st.columns([1, 1])
    with left:
        if not observed.empty and "period_risk_category" in observed.columns:
            st.plotly_chart(
                risk_donut(observed, "period_risk_category", "Observed location categories"),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    with right:
        if not forecast.empty and "predicted_2026_risk" in forecast.columns:
            st.plotly_chart(
                risk_donut(forecast, "predicted_2026_risk", "Predicted 2026 categories"),
                use_container_width=True,
                config={"displayModeBar": False},
            )

    section_header(
        "How a combined sewer works",
        "Hover over the illustrated stages. Normal flow goes to treatment; heavy rain can use the overflow route.",
    )
    render_sewer_story()


# =============================================================================
# PAGE 2 — COMBINED OBSERVED/PREDICTED MAP EXPERIENCE
# =============================================================================

elif page == "Explore the map":
    section_header(
        "Explore spill locations across England",
        "Start with every mapped outlet, then choose a cluster, town, water company or risk level.",
    )

    layer = st.radio(
        "What would you like to see?",
        ["Recorded 2023–2025 (what happened)", "2026 forecast (what may happen)"],
        horizontal=True,
    )
    display_style = st.radio(
        "How should the map look?",
        ["Clustered spill locations", "Risk concentration"],
        horizontal=True,
    )

    prediction = layer.startswith("2026")
    table_name = "forecast_map_points" if prediction else "observed_locations"
    risk_column = "predicted_2026_risk" if prediction else "period_risk_category"
    map_data = load_table(table_name)

    if map_data.empty:
        st.error("The map information is unavailable. Please try again later.")
    elif risk_column not in map_data.columns:
        st.error("The map cannot find the risk category needed for this view.")
    else:
        if prediction:
            banner(
                "<b>2026 forecast:</b> these colours show what the system estimates may happen. "
                "They do not show events that have already happened in 2026.",
                icon="🔮",
                background=PALE_AMBER,
                edge="#D59A3C",
            )
        else:
            banner(
                "<b>Recorded information:</b> every marker is a mapped discharge outlet using the supplied 2023–2025 records.",
                icon="🌊",
                background=PALE_MINT,
                edge="#4A9C7D",
            )

        render_risk_guide()
        filtered, place_column = filter_map(map_data, risk_column, prediction)
        risk_counts = filtered[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
        metric_cards(
            [
                {"label": "Locations currently shown", "value": value_text(len(filtered)), "note": "Change the choices above to narrow the map", "accent": "#B7DDE5"},
                {"label": "Low", "value": value_text(risk_counts["Low"]), "note": "● lower concern", "accent": "#A8D8D0"},
                {"label": "Medium", "value": value_text(risk_counts["Medium"]), "note": "◆ closer attention", "accent": "#F1D39D"},
                {"label": "High", "value": value_text(risk_counts["High"]), "note": "▲ priority review", "accent": "#E9A7A7"},
            ]
        )

        if filtered.empty:
            st.warning("No locations match those choices. Remove one or more filters and try again.")
        else:
            company_column = first_existing(filtered, ["water_company_name", "company"])
            place_ranking = make_risk_ranking(filtered, risk_column, place_column)
            company_ranking = make_risk_ranking(filtered, risk_column, company_column)

            # The full-width map contains the same place directory and water-
            # company ranking used by the detailed Colab map.
            with st.spinner("Drawing the interactive map and grouping nearby locations..."):
                map_object = build_folium_map(
                    filtered,
                    risk_column,
                    prediction,
                    display_style,
                )

            st_folium(
                map_object,
                height=900,
                use_container_width=True,
                returned_objects=[],
                key=f"edm_{'forecast' if prediction else 'recorded'}_{display_style}",
            )
            st.caption(
                "Select a numbered cluster to zoom in. Select a coloured marker "
                "for its site, place, company and risk details."
            )

            # Compact lists are retained as an accessible alternative.
            with st.expander("Compact rankings for the current map view"):
                place_tab, company_tab = st.tabs(["Places", "Water companies"])
                with place_tab:
                    render_rank_list(
                        place_ranking,
                        place_column,
                        "No town or city ranking is available for these choices.",
                        limit=10,
                    )
                with company_tab:
                    render_rank_list(
                        company_ranking,
                        company_column,
                        "No water-company ranking is available for these choices.",
                        limit=10,
                    )

            with st.expander("See the records behind this map"):
                st.dataframe(filtered.head(5000), use_container_width=True, hide_index=True)
                download_table(filtered, "filtered_2026_predictions.csv" if prediction else "filtered_observed_locations.csv")


# =============================================================================
# PAGE 3 — HIGH-RISK PRIORITY LOCATIONS
# =============================================================================

elif page == "Priority locations":
    section_header(
        "High-risk locations requiring priority review",
        "See the exact towns, mapped outlets and water companies linked to the High category.",
    )

    priority_view = st.radio(
        "Choose the evidence",
        ["Recorded 2023–2025", "Predicted 2026"],
        horizontal=True,
        key="priority_evidence_view",
    )
    priority_prediction = priority_view.startswith("Predicted")
    priority_table = "forecast_map_points" if priority_prediction else "observed_locations"
    priority_risk_column = "predicted_2026_risk" if priority_prediction else "period_risk_category"
    priority_data = load_table(priority_table)

    if priority_data.empty or priority_risk_column not in priority_data.columns:
        st.error("The priority-location information is unavailable. Please try again later.")
    else:
        priority_data = priority_data.loc[
            priority_data[priority_risk_column].astype(str).eq("High")
        ].copy()
        priority_place = first_existing(priority_data, ["official_place_name", "town_or_city"])
        priority_company = first_existing(priority_data, ["water_company_name", "company"])
        priority_site = first_existing(
            priority_data,
            ["site_name", "source_site_name_ea_consents_database"],
        )

        if priority_prediction:
            banner(
                "<b>Predicted 2026:</b> these are model-generated High categories, not confirmed events.",
                icon="🔮",
                background=PALE_AMBER,
                edge="#D59A3C",
            )
        else:
            banner(
                "<b>Recorded priority review:</b> High means the strongest concern category in this dashboard. "
                "It is not proof of environmental harm or an emergency declaration.",
                icon="🚨",
                background="#FBE8E8",
                edge="#D66565",
            )

        if not priority_place or not priority_company:
            st.error("Town/city or water-company fields are missing from the priority data.")
        elif priority_data.empty:
            st.info("No High-risk locations are available for this selection.")
        else:
            filter_one, filter_two, filter_three = st.columns([1, 1, 1.35])
            with filter_one:
                selected_priority_company = st.selectbox(
                    "Water company",
                    ["All companies"] + available_values(priority_data, priority_company),
                    key="priority_company_filter",
                )
            with filter_two:
                selected_priority_place = st.selectbox(
                    "Town or city",
                    ["All towns/cities"] + available_values(priority_data, priority_place),
                    key="priority_place_filter",
                )
            with filter_three:
                priority_search = st.text_input(
                    "Find a town, outlet, company or receiving water",
                    key="priority_search",
                ).strip()

            priority_filtered = priority_data.copy()
            if selected_priority_company != "All companies":
                priority_filtered = priority_filtered.loc[
                    priority_filtered[priority_company].astype(str).eq(selected_priority_company)
                ]
            if selected_priority_place != "All towns/cities":
                priority_filtered = priority_filtered.loc[
                    priority_filtered[priority_place].astype(str).eq(selected_priority_place)
                ]
            if priority_search:
                searchable_columns = [
                    column
                    for column in [
                        priority_place,
                        priority_company,
                        priority_site,
                        "receiving_water",
                        "source_receiving_water",
                        "permit_reference",
                    ]
                    if column and column in priority_filtered.columns
                ]
                priority_match = pd.Series(False, index=priority_filtered.index)
                for column in searchable_columns:
                    priority_match |= priority_filtered[column].astype("string").str.contains(
                        priority_search,
                        case=False,
                        regex=False,
                        na=False,
                    )
                priority_filtered = priority_filtered.loc[priority_match]

            if priority_filtered.empty:
                st.warning("No High-risk locations match these choices. Clear a filter and try again.")
            else:
                place_summary = (
                    priority_filtered.groupby(
                        [priority_place, priority_company],
                        dropna=False,
                    )
                    .size()
                    .reset_index(name="High-risk mapped outlets")
                    .sort_values(
                        ["High-risk mapped outlets", priority_place],
                        ascending=[False, True],
                    )
                    .reset_index(drop=True)
                )
                place_summary.insert(0, "Priority rank", np.arange(1, len(place_summary) + 1))

                company_summary = (
                    priority_filtered.groupby(priority_company, dropna=False)
                    .agg(
                        **{
                            "High-risk mapped outlets": (priority_risk_column, "size"),
                            "Towns/cities represented": (priority_place, "nunique"),
                        }
                    )
                    .reset_index()
                    .sort_values("High-risk mapped outlets", ascending=False)
                    .reset_index(drop=True)
                )
                company_summary.insert(0, "Company rank", np.arange(1, len(company_summary) + 1))

                top_place = str(place_summary.iloc[0][priority_place]) if not place_summary.empty else "Not available"
                metric_cards(
                    [
                        {
                            "label": "High-risk mapped outlets",
                            "value": value_text(len(priority_filtered)),
                            "note": priority_view,
                            "accent": "#E9A7A7",
                        },
                        {
                            "label": "Towns and cities",
                            "value": value_text(priority_filtered[priority_place].nunique()),
                            "note": "Places with at least one High location",
                            "accent": "#F1D39D",
                        },
                        {
                            "label": "Water companies",
                            "value": value_text(priority_filtered[priority_company].nunique()),
                            "note": "Companies represented in this view",
                            "accent": "#B7DDE5",
                        },
                        {
                            "label": "Highest-ranked place",
                            "value": top_place,
                            "note": "Ranked by High-risk mapped outlets",
                            "accent": "#A8D8D0",
                        },
                    ]
                )

                place_tab, company_tab, outlet_tab = st.tabs(
                    ["Towns and cities", "Water companies", "Exact mapped outlets"]
                )

                with place_tab:
                    chart_data = place_summary.head(20).sort_values("High-risk mapped outlets")
                    chart_data = chart_data.copy()
                    chart_data["Place and company"] = (
                        chart_data[priority_place].astype(str)
                        + " · "
                        + chart_data[priority_company].astype(str)
                    )
                    place_figure = px.bar(
                        chart_data,
                        x="High-risk mapped outlets",
                        y="Place and company",
                        orientation="h",
                        color="High-risk mapped outlets",
                        color_continuous_scale=["#F9DEDE", "#E9A7A7", "#C85D5D"],
                        text="High-risk mapped outlets",
                        title="Towns and cities with the most High-risk mapped outlets",
                    )
                    place_figure.update_traces(textposition="outside")
                    place_figure.update_layout(coloraxis_showscale=False)
                    place_figure.update_yaxes(title="")
                    st.plotly_chart(
                        plot_style(place_figure, max(480, 34 * len(chart_data))),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.dataframe(place_summary, use_container_width=True, hide_index=True)

                with company_tab:
                    company_figure = px.bar(
                        company_summary.sort_values("High-risk mapped outlets"),
                        x="High-risk mapped outlets",
                        y=priority_company,
                        orientation="h",
                        color="High-risk mapped outlets",
                        color_continuous_scale=["#DDEFF4", "#9FCAD5", "#4E9FB6"],
                        text="High-risk mapped outlets",
                        title="Water companies linked to High-risk mapped outlets",
                    )
                    company_figure.update_traces(textposition="outside")
                    company_figure.update_layout(coloraxis_showscale=False)
                    company_figure.update_yaxes(title="")
                    st.plotly_chart(
                        plot_style(company_figure, max(430, 45 * len(company_summary))),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.dataframe(company_summary, use_container_width=True, hide_index=True)

                with outlet_tab:
                    outlet_columns = [
                        column
                        for column in [
                            priority_site,
                            priority_place,
                            priority_company,
                            "receiving_water",
                            "source_receiving_water",
                            "permit_reference",
                            priority_risk_column,
                            "total_counted_spills_in_period",
                            "total_spill_duration_hours_in_period",
                            "prediction_confidence",
                            "latitude",
                            "longitude",
                        ]
                        if column and column in priority_filtered.columns
                    ]
                    outlet_records = priority_filtered[outlet_columns].copy()
                    st.dataframe(outlet_records, use_container_width=True, hide_index=True)
                    download_table(
                        outlet_records,
                        "predicted_2026_high_risk_locations.csv"
                        if priority_prediction
                        else "recorded_high_risk_locations.csv",
                    )


# =============================================================================
# PAGE 4 — PLACES, COMPANIES AND CHANGE
# =============================================================================

elif page == "Places and companies":
    section_header(
        "See how places and water companies compare",
        "Use the charts to compare recorded risk, changes over time and the 2026 forecast.",
    )

    company_tab, town_tab, change_tab = st.tabs(
        ["Water companies", "Towns and cities", "2025 → 2026 change"]
    )

    with company_tab:
        rankings = load_table("company_rankings")
        annual = load_table("annual_company_trends")
        if rankings.empty:
            st.info("The company-ranking export is unavailable.")
        else:
            metric_choice = st.radio(
                "How would you like to rank the companies?",
                ["Number of High-risk locations", "Share of locations that are Medium or High"],
                horizontal=True,
            )
            if metric_choice.startswith("Number"):
                value_column = "high_risk_unique_locations"
                x_title = "Unique observed High-risk locations"
                colour = RISK_COLOURS["High"]
            else:
                value_column = "medium_or_high_risk_percent"
                x_title = "Categorised locations that are Medium or High (%)"
                colour = RISK_COLOURS["Medium"]

            if {"water_company_name", value_column}.issubset(rankings.columns):
                plot = rankings.dropna(subset=[value_column]).sort_values(value_column)
                figure = px.bar(
                    plot,
                    x=value_column,
                    y="water_company_name",
                    orientation="h",
                    text=value_column,
                    color_discrete_sequence=[colour],
                    title=x_title,
                )
                figure.update_traces(texttemplate="%{text:.1f}" if "percent" in value_column else "%{text:,.0f}", textposition="outside")
                figure.update_xaxes(title=x_title)
                figure.update_yaxes(title="")
                st.plotly_chart(plot_style(figure, max(500, 48 * len(plot))), use_container_width=True)

            with st.expander("Company ranking table"):
                st.dataframe(rankings, use_container_width=True, hide_index=True)

        if not annual.empty and "water_company_name" in annual.columns:
            company_options = available_values(annual, "water_company_name")
            if company_options:
                company = st.selectbox("Explore one company's annual trend", company_options)
                rows = annual.loc[annual["water_company_name"].astype(str).eq(company)].copy()
                risk_columns = [
                    column for column in [
                        "low_risk_unique_locations", "medium_risk_unique_locations", "high_risk_unique_locations"
                    ] if column in rows.columns
                ]
                if risk_columns and "reporting_year" in rows.columns:
                    long = rows.melt(id_vars="reporting_year", value_vars=risk_columns, var_name="Risk", value_name="Locations")
                    long["Risk"] = long["Risk"].str.replace("_risk_unique_locations", "", regex=False).str.title()
                    figure = px.line(
                        long, x="reporting_year", y="Locations", color="Risk", markers=True,
                        color_discrete_map=RISK_COLOURS, title=f"Observed annual risk profile · {company}"
                    )
                    figure.update_xaxes(dtick=1, title="Reporting year")
                    st.plotly_chart(plot_style(figure, 440), use_container_width=True)

    with town_tab:
        towns = load_table("town_trends")
        if towns.empty or "official_place_name" not in towns.columns:
            st.info("The town/city trend export is unavailable.")
        else:
            place_options = available_values(towns, "official_place_name")
            place = st.selectbox("Select a town or city", place_options)
            row = towns.loc[towns["official_place_name"].astype(str).eq(place)].iloc[0]
            metric_cards(
                [
                    {"label": "Place", "value": place, "note": str(row.get("water_companies", "Company not recorded")), "accent": "#B7DDE5"},
                    {"label": "2023 → 2025 direction", "value": str(row.get("trend_2023_to_2025", "Not available")), "note": "Observed counted-spill direction", "accent": "#A8D8D0"},
                    {"label": "Risk history", "value": str(row.get("town_risk_transition", "Not available")), "note": "Highest annual mapped risk", "accent": "#F1D39D"},
                    {"label": "2025 counted spills", "value": value_text(row.get("counted_spills_2025")), "note": "Recorded evidence—not volume", "accent": "#E9A7A7"},
                ]
            )
            trend = pd.DataFrame(
                {
                    "Year": [2023, 2024, 2025],
                    "Counted spills": [row.get(f"counted_spills_{year}") for year in [2023, 2024, 2025]],
                    "Duration hours": [row.get(f"duration_hours_{year}") for year in [2023, 2024, 2025]],
                }
            )
            left, right = st.columns(2)
            with left:
                figure = px.area(trend, x="Year", y="Counted spills", markers=True, title=f"Counted spills · {place}")
                figure.update_traces(line_color="#4A9C7D", fillcolor="rgba(74,156,125,0.20)")
                figure.update_xaxes(dtick=1)
                st.plotly_chart(plot_style(figure, 400), use_container_width=True)
            with right:
                figure = px.area(trend, x="Year", y="Duration hours", markers=True, title=f"Recorded duration · {place}")
                figure.update_traces(line_color="#68AFC2", fillcolor="rgba(104,175,194,0.20)")
                figure.update_xaxes(dtick=1)
                st.plotly_chart(plot_style(figure, 400), use_container_width=True)
            with st.expander("Search the full town/city table"):
                st.dataframe(towns, use_container_width=True, hide_index=True)

    with change_tab:
        comparison = load_table("observed_vs_predicted")
        companies = load_table("observed_predicted_company_comparison")
        banner(
            "The left axis is the <b>observed 2025</b> category; the bottom axis is the "
            "<b>predicted 2026</b> category. Forecasts remain decision-support outputs.",
            icon="↔️",
            background=PALE_LAVENDER,
            edge="#8F79A8",
        )
        if comparison.empty:
            st.info("The observed-versus-predicted export is unavailable.")
        else:
            matrix = pd.crosstab(
                comparison["observed_2025_risk"], comparison["predicted_2026_risk"]
            ).reindex(index=RISK_ORDER, columns=RISK_ORDER, fill_value=0)
            figure = go.Figure(
                go.Heatmap(
                    z=matrix.to_numpy(), x=matrix.columns, y=matrix.index,
                    text=matrix.to_numpy(), texttemplate="%{text:,}",
                    colorscale=[[0, "#F4FAF7"], [0.35, "#C5E5DA"], [0.70, "#73B5A1"], [1, "#2F7569"]],
                    hovertemplate="Observed %{y}<br>Predicted %{x}<br>%{z:,} locations<extra></extra>",
                )
            )
            figure.update_layout(title="Observed 2025 → predicted 2026 transition matrix", xaxis_title="Predicted 2026", yaxis_title="Observed 2025")
            st.plotly_chart(plot_style(figure, 520), use_container_width=True)
        if not companies.empty:
            with st.expander("Company-level observed and predicted comparison"):
                st.dataframe(companies, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 5 — 2026 PREDICTIONS AND AFFECTED LOCATIONS
# =============================================================================

elif page == "2026 predictions":
    section_header(
        "Predicted 2026 spill risks and affected locations",
        "Explore the forecast by risk, town or city, water company and exact mapped outlet.",
    )
    banner(
        "<b>2026 forecast:</b> these are model-generated risk estimates for planning and review. "
        "They are not confirmed 2026 spill events.",
        icon="🔮",
        background=PALE_AMBER,
        edge="#D59A3C",
    )

    forecast = load_table("forecast_map_points")
    forecast_risk = "predicted_2026_risk"

    if forecast.empty or forecast_risk not in forecast.columns:
        st.error("The 2026 prediction information is unavailable. Please try again later.")
    else:
        forecast_place = first_existing(forecast, ["official_place_name", "town_or_city"])
        forecast_company = first_existing(forecast, ["water_company_name", "company"])
        forecast_site = first_existing(
            forecast,
            ["site_name", "source_site_name_ea_consents_database"],
        )

        filter_columns = st.columns([1.05, 1.05, 1.05, 1.35])
        with filter_columns[0]:
            forecast_risks = st.multiselect(
                "Predicted risk",
                RISK_ORDER,
                default=RISK_ORDER,
                key="forecast_page_risks",
            )
        with filter_columns[1]:
            forecast_company_choice = st.selectbox(
                "Water company",
                ["All companies"] + available_values(forecast, forecast_company)
                if forecast_company
                else ["All companies"],
                key="forecast_page_company",
            )
        with filter_columns[2]:
            forecast_place_choice = st.selectbox(
                "Town or city",
                ["All towns/cities"] + available_values(forecast, forecast_place)
                if forecast_place
                else ["All towns/cities"],
                key="forecast_page_place",
            )
        with filter_columns[3]:
            forecast_search = st.text_input(
                "Find a location, company or receiving water",
                key="forecast_page_search",
            ).strip()

        forecast_filtered = forecast.loc[
            forecast[forecast_risk].isin(forecast_risks)
        ].copy() if forecast_risks else forecast.iloc[0:0].copy()

        if forecast_company and forecast_company_choice != "All companies":
            forecast_filtered = forecast_filtered.loc[
                forecast_filtered[forecast_company].astype(str).eq(forecast_company_choice)
            ]
        if forecast_place and forecast_place_choice != "All towns/cities":
            forecast_filtered = forecast_filtered.loc[
                forecast_filtered[forecast_place].astype(str).eq(forecast_place_choice)
            ]
        if forecast_search:
            search_columns = [
                column
                for column in [
                    forecast_place,
                    forecast_company,
                    forecast_site,
                    "receiving_water",
                    "source_receiving_water",
                    "permit_reference",
                ]
                if column and column in forecast_filtered.columns
            ]
            forecast_match = pd.Series(False, index=forecast_filtered.index)
            for column in search_columns:
                forecast_match |= forecast_filtered[column].astype("string").str.contains(
                    forecast_search,
                    case=False,
                    regex=False,
                    na=False,
                )
            forecast_filtered = forecast_filtered.loc[forecast_match]

        if forecast_filtered.empty:
            st.warning("No predicted locations match these choices. Clear a filter and try again.")
        else:
            forecast_counts = (
                forecast_filtered[forecast_risk]
                .value_counts()
                .reindex(RISK_ORDER, fill_value=0)
            )
            metric_cards(
                [
                    {
                        "label": "Predicted locations",
                        "value": value_text(len(forecast_filtered)),
                        "note": "Mapped outlets in this view",
                        "accent": "#B7DDE5",
                    },
                    {
                        "label": "Predicted High",
                        "value": value_text(forecast_counts["High"]),
                        "note": "Priority review category",
                        "accent": "#E9A7A7",
                    },
                    {
                        "label": "Predicted Medium",
                        "value": value_text(forecast_counts["Medium"]),
                        "note": "Closer-attention category",
                        "accent": "#F1D39D",
                    },
                    {
                        "label": "Affected towns/cities",
                        "value": value_text(
                            forecast_filtered[forecast_place].nunique()
                            if forecast_place
                            else np.nan
                        ),
                        "note": "Places represented in the forecast",
                        "accent": "#A8D8D0",
                    },
                ]
            )

            map_tab, place_tab, company_tab, location_tab = st.tabs(
                [
                    "Interactive forecast map",
                    "Affected towns and cities",
                    "Water companies",
                    "Exact predicted locations",
                ]
            )

            with map_tab:
                with st.spinner("Drawing the 2026 prediction map..."):
                    forecast_map = build_folium_map(
                        forecast_filtered,
                        forecast_risk,
                        True,
                        "Clustered spill locations",
                    )
                st_folium(
                    forecast_map,
                    height=860,
                    use_container_width=True,
                    returned_objects=[],
                    key="dedicated_2026_prediction_map",
                )
                st.caption(
                    "Select a cluster to zoom in, then select a marker for the predicted "
                    "risk, probabilities, town/city and water company."
                )

            with place_tab:
                if forecast_place:
                    forecast_place_ranking = make_risk_ranking(
                        forecast_filtered,
                        forecast_risk,
                        forecast_place,
                    )
                    place_chart = forecast_place_ranking.head(25).copy()
                    place_long = place_chart.melt(
                        id_vars=[forecast_place],
                        value_vars=RISK_ORDER,
                        var_name="Predicted risk",
                        value_name="Mapped locations",
                    )
                    place_figure = px.bar(
                        place_long,
                        x="Mapped locations",
                        y=forecast_place,
                        color="Predicted risk",
                        orientation="h",
                        barmode="stack",
                        color_discrete_map=RISK_COLOURS,
                        category_orders={"Predicted risk": RISK_ORDER},
                        title="Affected towns and cities ranked by predicted risk",
                    )
                    place_figure.update_yaxes(
                        title="",
                        categoryorder="array",
                        categoryarray=place_chart[forecast_place].iloc[::-1].tolist(),
                    )
                    st.plotly_chart(
                        plot_style(place_figure, max(520, 29 * len(place_chart))),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.dataframe(
                        forecast_place_ranking,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Town/city information is unavailable for these predictions.")

            with company_tab:
                if forecast_company:
                    forecast_company_ranking = make_risk_ranking(
                        forecast_filtered,
                        forecast_risk,
                        forecast_company,
                    )
                    company_chart = forecast_company_ranking.copy()
                    company_long = company_chart.melt(
                        id_vars=[forecast_company],
                        value_vars=RISK_ORDER,
                        var_name="Predicted risk",
                        value_name="Mapped locations",
                    )
                    company_figure = px.bar(
                        company_long,
                        x="Mapped locations",
                        y=forecast_company,
                        color="Predicted risk",
                        orientation="h",
                        barmode="stack",
                        color_discrete_map=RISK_COLOURS,
                        category_orders={"Predicted risk": RISK_ORDER},
                        title="Water companies ranked by predicted 2026 risk",
                    )
                    company_figure.update_yaxes(
                        title="",
                        categoryorder="array",
                        categoryarray=company_chart[forecast_company].iloc[::-1].tolist(),
                    )
                    st.plotly_chart(
                        plot_style(company_figure, max(450, 44 * len(company_chart))),
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    st.dataframe(
                        forecast_company_ranking,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("Water-company information is unavailable for these predictions.")

            with location_tab:
                prediction_columns = [
                    column
                    for column in [
                        forecast_site,
                        forecast_place,
                        forecast_company,
                        "receiving_water",
                        "source_receiving_water",
                        forecast_risk,
                        "probability_low",
                        "probability_medium",
                        "probability_high",
                        "prediction_confidence",
                        "confidence_flag",
                        "permit_reference",
                        "latitude",
                        "longitude",
                    ]
                    if column and column in forecast_filtered.columns
                ]
                prediction_records = forecast_filtered[prediction_columns].copy()
                risk_sort = {"High": 0, "Medium": 1, "Low": 2}
                prediction_records["_risk_order"] = prediction_records[forecast_risk].map(risk_sort)
                prediction_records = prediction_records.sort_values(
                    ["_risk_order", forecast_place]
                    if forecast_place in prediction_records.columns
                    else ["_risk_order"]
                ).drop(columns="_risk_order")
                st.dataframe(
                    prediction_records,
                    use_container_width=True,
                    hide_index=True,
                )
                download_table(prediction_records, "predicted_2026_affected_locations.csv")


# =============================================================================
# PAGE 5 — INDIVIDUAL PREDICTION
# =============================================================================

elif page == "Check one location":
    section_header(
        "Check one location",
        "Enter the information known for the previous year to see the system's suggested 2026 risk category.",
    )
    banner(
        "This is an estimate for investigation and planning. It does not confirm that a future spill will occur.",
        icon="⚠️",
        background=PALE_AMBER,
        edge="#D59A3C",
    )

    try:
        bundle = load_model_bundle()
        metadata = load_input_metadata()
    except Exception as error:
        st.error(f"The saved prediction model could not be loaded: {error}")
        st.stop()

    st.caption("The same prediction method used for the public 2026 map is used here.")
    values = {}

    with st.form("prediction_form"):
        st.subheader("1. Information from the previous year")
        numeric_columns = metadata.get("numeric_columns", [])
        numeric_widgets = st.columns(2)
        for position, column in enumerate(numeric_columns):
            default = float(metadata.get("numeric_defaults", {}).get(column, 0.0) or 0.0)
            with numeric_widgets[position % 2]:
                values[column] = st.number_input(pretty(column), value=default, format="%.3f")

        st.subheader("2. About the location")
        categorical_columns = metadata.get("categorical_columns", [])
        categorical_widgets = st.columns(2)
        for position, column in enumerate(categorical_columns):
            options = metadata.get("categorical_options", {}).get(column, ["__MISSING__"]) or ["__MISSING__"]
            display_options = ["Not recorded" if option == "__MISSING__" else option for option in options]
            with categorical_widgets[position % 2]:
                selected = st.selectbox(pretty(column), display_options)
                values[column] = "__MISSING__" if selected == "Not recorded" else selected

        submitted = st.form_submit_button("Show the estimated 2026 risk", type="primary", use_container_width=True)

    if submitted:
        raw_input = pd.DataFrame([values])
        for column in categorical_columns:
            raw_input[column] = raw_input[column].astype("string").fillna("__MISSING__").astype(str)
        transformed = np.asarray(bundle["preprocessor"].transform(raw_input), dtype=np.float32)
        transformed[~np.isfinite(transformed)] = np.nan
        raw_probabilities = aligned_probabilities(bundle["model"], transformed)
        probabilities = apply_calibrators(raw_probabilities, bundle["probability_calibrators"])[0]
        code = int(np.argmax(probabilities))
        labels = bundle.get("risk_code_to_label", {0: "Low", 1: "Medium", 2: "High"})
        prediction = labels.get(code, labels.get(str(code), str(code)))
        ordered = np.sort(probabilities)
        margin = float(ordered[-1] - ordered[-2])
        confidence = float(probabilities.max())

        metric_cards(
            [
                {"label": "Predicted 2026 risk", "value": str(prediction), "note": "Model-generated category", "accent": {"Low": "#A8D8D0", "Medium": "#F1D39D", "High": "#E9A7A7"}.get(str(prediction), "#B7DDE5")},
                {"label": "Low probability", "value": f"{probabilities[0]:.1%}", "note": "Calibrated probability", "accent": "#A8D8D0"},
                {"label": "Medium probability", "value": f"{probabilities[1]:.1%}", "note": "Calibrated probability", "accent": "#F1D39D"},
                {"label": "High probability", "value": f"{probabilities[2]:.1%}", "note": "Calibrated probability", "accent": "#E9A7A7"},
            ]
        )
        probability_figure = go.Figure(
            go.Bar(
                x=RISK_ORDER,
                y=probabilities,
                marker_color=[RISK_COLOURS[label] for label in RISK_ORDER],
                text=[f"{value:.1%}" for value in probabilities],
                textposition="outside",
            )
        )
        probability_figure.update_yaxes(range=[0, 1], tickformat=".0%", title="Calibrated probability")
        probability_figure.update_layout(title="Probability profile", showlegend=False)
        st.plotly_chart(plot_style(probability_figure, 410), use_container_width=True, config={"displayModeBar": False})

        if confidence < 0.60 or margin < 0.15:
            banner(
                f"<b>Review required:</b> maximum probability is {confidence:.1%} and the leading margin is {margin:.1%}.",
                icon="🔍",
                background=PALE_AMBER,
                edge="#D59A3C",
            )
        else:
            banner(
                f"This result meets the dashboard's higher-confidence rule (confidence {confidence:.1%}; margin {margin:.1%}).",
                icon="✅",
                background=PALE_MINT,
                edge="#4A9C7D",
            )


# =============================================================================
# PAGE 6 — WATER QUALITY, AUDIT, SEARCH AND LIMITATIONS
# =============================================================================

else:
    section_header(
        "About the evidence",
        "Explore nearby water measurements, find a record and understand what this website can and cannot show.",
    )

    water_tab, audit_tab, search_tab, method_tab = st.tabs(
        ["Water and rivers", "How the data was checked", "Find a record", "How it works"]
    )

    with water_tab:
        water_kpis = load_table("water_quality_kpis")
        quality = load_table("water_quality_records")
        coverage = load_table("water_quality_coverage")

        banner(
            "Nearby Environment Agency monitoring records provide environmental context. "
            "Geographic proximity does not prove that an EDM outlet caused a measured result.",
            icon="🌱",
            background=PALE_MINT,
            edge="#4A9C7D",
        )
        if not water_kpis.empty and {"KPI", "Value"}.issubset(water_kpis.columns):
            cards = []
            for position, (_, row) in enumerate(water_kpis.head(4).iterrows()):
                cards.append({"label": str(row["KPI"]), "value": value_text(row["Value"]), "note": str(row.get("Meaning", "Official nearby-station evidence")), "accent": ["#A8D8D0", "#B7DDE5", "#CDBDDE", "#F1D39D"][position % 4]})
            metric_cards(cards)

        if not coverage.empty:
            with st.expander("Water-quality coverage by company"):
                st.dataframe(coverage, use_container_width=True, hide_index=True)

        if quality.empty:
            st.info("No public water-quality export is available.")
        else:
            filters = st.columns(2)
            filtered = quality.copy()
            with filters[0]:
                company_options = ["All companies"] + available_values(filtered, "company")
                company = st.selectbox("Company", company_options, key="water_company")
            with filters[1]:
                parameter_options = ["All parameters"] + available_values(filtered, "project_parameter_name")
                parameter = st.selectbox("Measured parameter", parameter_options)
            if company != "All companies" and "company" in filtered.columns:
                filtered = filtered.loc[filtered["company"].astype(str).eq(company)]
            if parameter != "All parameters" and "project_parameter_name" in filtered.columns:
                filtered = filtered.loc[filtered["project_parameter_name"].astype(str).eq(parameter)]
            st.dataframe(filtered.head(5000), use_container_width=True, hide_index=True)
            download_table(filtered, "filtered_water_quality_evidence.csv")

    with audit_tab:
        audit_tables = [
            ("Target-leakage audit", "leakage_audit"),
            ("Temporal matching audit", "temporal_matching_audit"),
            ("Model-data usability", "model_data_quality"),
            ("Missing measurements", "missing_measurement_audit"),
            ("Coordinate quality", "coordinate_quality"),
        ]
        for title, name in audit_tables:
            with st.expander(title, expanded=title == "Target-leakage audit"):
                frame = load_table(name)
                if frame.empty:
                    st.info(f"{title} is unavailable.")
                else:
                    st.dataframe(frame, use_container_width=True, hide_index=True)

    with search_tab:
        query = st.text_input(
            "Search company, town/city, site, receiving water or permit reference",
            placeholder="For example: Leeds, Thames Water, River Avon…",
        ).strip()
        if query:
            search_tables = [
                ("Observed locations", "observed_locations"),
                ("Forecast locations", "forecast_map_points"),
                ("Town/city trends", "town_trends"),
                ("Company rankings", "company_rankings"),
            ]
            results_found = False
            for title, name in search_tables:
                frame = load_table(name)
                if frame.empty:
                    continue
                text_columns = frame.select_dtypes(include=["object", "string"]).columns
                match = pd.Series(False, index=frame.index)
                for column in text_columns:
                    match |= frame[column].astype("string").str.contains(query, case=False, regex=False, na=False)
                result = frame.loc[match]
                if not result.empty:
                    results_found = True
                    st.subheader(f"{title} · {len(result):,} matches")
                    st.dataframe(result.head(500), use_container_width=True, hide_index=True)
            if not results_found:
                st.warning("No verified dashboard records matched that search.")

    with method_tab:
        st.html(
            """
            <div class="edm-journey">
              <div class="edm-journey-step"><span class="edm-journey-number">1</span><h4>Recorded information</h4><p>The map starts with cleaned records supplied for 2023–2025.</p></div>
              <div class="edm-journey-step"><span class="edm-journey-number">2</span><h4>Year-by-year checks</h4><p>Earlier years are used to estimate the following year, so future information is not used too early.</p></div>
              <div class="edm-journey-step"><span class="edm-journey-number">3</span><h4>Clear results</h4><p>Every map and ranking labels recorded information separately from forecasts.</p></div>
            </div>
            """,
        )
        banner(
            "<b>Interpretation boundary:</b> this prototype supports investigation and prioritisation. "
            "It does not measure sewage volume, prove ecological damage, establish legal liability or confirm a predicted event.",
            icon="🛟",
            background=PALE_AMBER,
            edge="#D59A3C",
        )
        st.markdown(
            """
            - Observed Low, Medium and High categories are historical evidence.
            - Predicted 2026 categories are calibrated machine-learning outputs.
            - Spill duration is not sewage volume.
            - Town names are geographic reference points, not proof that every discharge occurred inside a town boundary.
            - Nearby water-quality observations show geographic association, not causation.
            - The 2025 cohort has informed model assessment; genuinely new later data is required for final external testing.
            """
        )


st.html(
    """
    <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(55,120,110,.18);
                color:#5D7772;font-size:.78rem;text-align:center;">
      England EDM Water &amp; Spill-Risk Observatory · verified evidence, transparent forecasts and responsible interpretation
    </div>
    """,
)
