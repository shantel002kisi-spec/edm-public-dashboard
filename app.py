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
    initial_sidebar_state="expanded",
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
        font-family: "Inter", "Segoe UI", Arial, sans-serif;
        color: var(--edm-ink);
      }

      .stApp {
        background:
          radial-gradient(circle at 86% 4%, rgba(178, 220, 226, 0.35), transparent 24rem),
          radial-gradient(circle at 8% 92%, rgba(190, 224, 204, 0.38), transparent 25rem),
          linear-gradient(145deg, #FBFDF9 0%, #F3FAF7 48%, #F5F9FC 100%);
      }

      .block-container {
        max-width: 1450px;
        padding-top: 1.1rem;
        padding-bottom: 4rem;
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
        background: rgba(255,255,255,0.48);
        border: 1px solid rgba(62, 127, 117, 0.15);
        border-radius: 12px;
        padding: 0.42rem 0.62rem;
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
        grid-template-columns: minmax(0, 1.6fr) minmax(260px, 0.7fr);
        gap: 1.4rem;
        align-items: center;
        padding: 1.7rem 2rem;
        margin: 0.1rem 0 1.2rem;
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
        font-size: clamp(2rem, 4.2vw, 3.65rem);
        line-height: 1.03;
        letter-spacing: -0.045em;
      }

      .edm-hero p {
        max-width: 850px;
        margin: 0.8rem 0 0;
        color: var(--edm-muted);
        font-size: 1.07rem;
        line-height: 1.65;
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
        min-height: 185px;
        display: flex;
        align-items: center;
        justify-content: center;
      }

      .edm-section-header {
        margin: 0.2rem 0 0.9rem;
        padding: 1rem 1.2rem;
        border-left: 7px solid #4A9C7D;
        border-radius: 0 16px 16px 0;
        background: linear-gradient(90deg, rgba(234,246,240,0.95), rgba(255,255,255,0.50));
      }

      .edm-section-header h2 {
        margin: 0;
        color: var(--edm-ink);
        font-size: 1.75rem;
        line-height: 1.15;
      }

      .edm-section-header p {
        margin: 0.38rem 0 0;
        color: var(--edm-muted);
        line-height: 1.5;
      }

      .edm-metric-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 0.7rem 0 1.2rem;
      }

      .edm-metric-card {
        position: relative;
        overflow: hidden;
        min-height: 132px;
        padding: 1rem 1.05rem;
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
        font-size: 2rem;
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

      .edm-banner {
        display: flex;
        gap: 0.75rem;
        align-items: flex-start;
        padding: 0.9rem 1rem;
        margin: 0.6rem 0 1rem;
        border-radius: 14px;
        color: var(--edm-ink);
        background: var(--banner-bg, #E9F4F8);
        border-left: 6px solid var(--banner-edge, #68AFC2);
        line-height: 1.55;
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

      @media (max-width: 1050px) {
        .edm-metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .edm-hero { grid-template-columns: 1fr; }
        .edm-water-art { min-height: 135px; }
      }

      @media (max-width: 680px) {
        .edm-metric-grid, .edm-journey { grid-template-columns: 1fr; }
        .edm-hero { padding: 1.25rem; border-radius: 18px; }
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
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
    st.markdown(
        """
        <section class="edm-hero">
          <div>
            <div class="edm-kicker">💧 England environmental intelligence</div>
            <h1>EDM Water &amp; Spill-Risk Observatory</h1>
            <p>
              Explore verified 2023–2025 evidence, compare places and water companies,
              and examine transparent 2026 machine-learning forecasts. Observations and
              predictions remain visibly separated throughout the system.
            </p>
          </div>
          <div class="edm-water-art" aria-label="Water, river and leaf illustration">
            <svg viewBox="0 0 360 220" width="100%" role="img" aria-label="Pastel water and environment illustration">
              <defs>
                <linearGradient id="waterDrop" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0" stop-color="#93D3DE"/>
                  <stop offset="1" stop-color="#4A9C7D"/>
                </linearGradient>
              </defs>
              <circle cx="185" cy="112" r="92" fill="#FFFFFF" opacity="0.55"/>
              <path d="M180 28 C150 74 128 101 128 134 C128 170 153 192 184 192 C216 192 241 169 241 135 C241 101 215 72 180 28Z" fill="url(#waterDrop)" opacity="0.94"/>
              <path d="M154 140 C173 150 198 150 217 138" fill="none" stroke="#EAF7F4" stroke-width="8" stroke-linecap="round"/>
              <path d="M47 169 C84 145 111 149 140 169 C168 188 201 190 236 169 C266 151 294 151 330 169" fill="none" stroke="#68AFC2" stroke-width="9" stroke-linecap="round" opacity="0.72"/>
              <path d="M41 193 C81 171 115 176 143 193 C173 211 205 211 236 192 C271 171 300 174 334 193" fill="none" stroke="#A8D8D0" stroke-width="7" stroke-linecap="round"/>
              <path d="M82 75 C57 58 43 66 45 88 C68 91 79 84 82 75Z" fill="#8DC6A7"/>
              <path d="M81 75 C85 50 99 44 115 60 C105 78 94 83 81 75Z" fill="#B5DDBE"/>
              <path d="M282 93 C301 68 318 71 323 91 C306 105 293 104 282 93Z" fill="#8DC6A7"/>
              <path d="M282 93 C276 69 260 62 247 80 C256 98 269 102 282 93Z" fill="#C2E3C8"/>
            </svg>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="edm-section-header">
          <h2>{html.escape(title)}</h2>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def banner(text: str, icon="ℹ️", background=PALE_BLUE, edge="#68AFC2"):
    st.markdown(
        f"""
        <div class="edm-banner" style="--banner-bg:{background};--banner-edge:{edge};">
          <span aria-hidden="true">{icon}</span><div>{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
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
    st.markdown('<div class="edm-metric-grid">' + "".join(pieces) + "</div>", unsafe_allow_html=True)


def plot_style(figure: go.Figure, height=480):
    figure.update_layout(
        height=height,
        margin=dict(l=28, r=20, t=65, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.45)",
        font=dict(color=INK, size=13),
        title_font=dict(color=INK, size=18),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#FFFFFF", font_color=INK),
    )
    figure.update_xaxes(gridcolor="rgba(68,120,110,0.12)", zeroline=False)
    figure.update_yaxes(gridcolor="rgba(68,120,110,0.12)", zeroline=False)
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
    place = row.get("official_place_name", row.get("town_or_city", "Not available"))
    site = row.get("site_name", row.get("source_site_name_ea_consents_database", "Not available"))
    company = row.get("water_company_name", "Not available")
    receiving = row.get("receiving_water", row.get("source_receiving_water", "Not available"))
    catchment = row.get("catchment_name", row.get("catchment", "Not available"))
    grid = row.get("parsed_grid_reference", "Not available")
    risk = row.get(risk_column, "Not available")

    evidence_rows = ""
    if prediction:
        observed = row.get("observed_2025_risk", row.get("actual_2025_risk_label", "Not available"))
        confidence = pd.to_numeric(pd.Series([row.get("prediction_confidence")]), errors="coerce").iloc[0]
        confidence_flag = row.get("confidence_flag", "Not available")
        probabilities = []
        for label, column in [("Low", "probability_low"), ("Medium", "probability_medium"), ("High", "probability_high")]:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            probabilities.append(f"{label}: {value:.1%}" if pd.notna(value) else f"{label}: not available")
        evidence_rows = f"""
          <div style="margin-top:7px;padding:7px;background:#FFF3DD;border-radius:7px;">
            <b>Observed 2025:</b> {safe_text(observed)}<br>
            <b>Predicted 2026:</b> {safe_text(risk)}<br>
            <b>Probability:</b> {' · '.join(probabilities)}<br>
            <b>Confidence:</b> {f'{confidence:.1%}' if pd.notna(confidence) else 'Not available'}<br>
            <b>Review flag:</b> {safe_text(confidence_flag)}
          </div>
        """
    else:
        year = row.get("reporting_year", "2023–2025 combined period")
        spills = row.get(
            "total_counted_spills_in_period",
            row.get(
                "counted_spills",
                row.get("counted_spills_using_12_24h_count_method", "Not available"),
            ),
        )
        duration = row.get(
            "total_spill_duration_hours_in_period",
            row.get("total_duration_hours", "Not available"),
        )
        evidence_rows = f"""
          <div style="margin-top:7px;padding:7px;background:#EAF6F0;border-radius:7px;">
            <b>Observed risk:</b> {safe_text(risk)}<br>
            <b>Evidence period:</b> {safe_text(year)}<br>
            <b>Period counted spills:</b> {value_text(spills)}<br>
            <b>Period recorded duration:</b> {value_text(duration, 1, ' hours')}
          </div>
        """

    return f"""
      <div style="font-family:Arial,sans-serif;width:300px;color:#173D3A;line-height:1.42;">
        <div style="font-size:16px;font-weight:700;margin-bottom:5px;">{safe_text(site)}</div>
        <b>Town/city:</b> {safe_text(place)}<br>
        <b>Water company:</b> {safe_text(company)}<br>
        <b>Receiving water:</b> {safe_text(receiving)}<br>
        <b>Catchment:</b> {safe_text(catchment)}<br>
        <b>Grid reference:</b> {safe_text(grid)}
        {evidence_rows}
        <div style="margin-top:7px;font-size:11px;color:#5D7772;">
          Coordinates: {float(row['latitude']):.5f}, {float(row['longitude']):.5f}
        </div>
      </div>
    """


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

    if display_style == "Location density":
        HeatMap(
            plotting[["latitude", "longitude"]].to_numpy().tolist(),
            name="Filtered location density",
            radius=13,
            blur=17,
            min_opacity=0.28,
            gradient={0.20: "#B7DDE5", 0.48: "#7BC6B5", 0.72: "#E8C77C", 1.0: "#D66565"},
        ).add_to(water_map)
    else:
        cluster_data = []
        for _, row in plotting.iterrows():
            risk = str(row[risk_column])
            place = row.get("official_place_name", row.get("town_or_city", "Unknown place"))
            site = row.get("site_name", "EDM location")
            tooltip = f"{RISK_SYMBOLS.get(risk, '●')} {risk} · {site} · {place}"
            cluster_data.append(
                [
                    float(row["latitude"]),
                    float(row["longitude"]),
                    risk,
                    popup_for_row(row, risk_column, prediction),
                    safe_text(tooltip),
                ]
            )

        callback = """
        function (row) {
          const colours = {Low: '#4A9C7D', Medium: '#E2A45C', High: '#D66565'};
          const symbols = {Low: '●', Medium: '◆', High: '▲'};
          const marker = L.circleMarker([row[0], row[1]], {
            radius: 6.5,
            color: '#FFFFFF',
            weight: 1.4,
            fillColor: colours[row[2]] || '#78909C',
            fillOpacity: 0.90
          });
          marker.bindPopup(row[3], {maxWidth: 340});
          marker.bindTooltip(row[4], {direction: 'top', opacity: 0.96});
          return marker;
        }
        """
        FastMarkerCluster(
            data=cluster_data,
            callback=callback,
            name="Clustered EDM locations",
            show=True,
        ).add_to(water_map)

    Fullscreen(position="topright", title="Open full-screen map", title_cancel="Exit full screen").add_to(water_map)
    MeasureControl(position="topright", primary_length_unit="kilometers").add_to(water_map)
    MiniMap(toggle_display=True, minimized=True, position="bottomright").add_to(water_map)
    folium.LayerControl(collapsed=True, position="topright").add_to(water_map)

    legend = """
    <div style="position:fixed;left:18px;bottom:28px;z-index:9999;
                background:rgba(255,255,255,.94);border:1px solid #BFD6CF;
                border-radius:12px;padding:10px 13px;color:#173D3A;
                box-shadow:0 5px 18px rgba(34,82,75,.16);font:13px Arial;">
      <div style="font-weight:700;margin-bottom:6px;">Risk category</div>
      <div><span style="color:#4A9C7D;font-size:17px;">●</span> Low</div>
      <div><span style="color:#E2A45C;font-size:16px;">◆</span> Medium</div>
      <div><span style="color:#D66565;font-size:15px;">▲</span> High</div>
      <div style="margin-top:6px;font-size:11px;color:#5D7772;">Click a marker for evidence</div>
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
            "Risk category",
            RISK_ORDER,
            default=RISK_ORDER,
            key=f"{'forecast' if prediction else 'observed'}_risk",
        )
    with filter_columns[1]:
        company_options = ["All companies"] + available_values(filtered, company_column) if company_column else ["All companies"]
        company = st.selectbox("Water company", company_options, key=f"{'forecast' if prediction else 'observed'}_company")
    with filter_columns[2]:
        place_options = ["All towns/cities"] + available_values(filtered, place_column) if place_column else ["All towns/cities"]
        place = st.selectbox("Town or city", place_options, key=f"{'forecast' if prediction else 'observed'}_place")
    with filter_columns[3]:
        search = st.text_input(
            "Search site, permit or receiving water",
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

st.sidebar.markdown(
    """
    <div class="edm-brand">
      <div style="display:flex;align-items:center;">
        <span class="edm-brand-mark">💧</span>
        <div><div style="font-size:1.18rem;font-weight:800;">EDM Observatory</div>
        <div style="font-size:.78rem;color:#5D7772;">England · evidence · forecast</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = [
            "🏡 Overview",
            "🗺️ Explore maps",
            "🏙️ Places & companies",
            "📊 Model performance",
            "🔎 Individual prediction",
            "🌿 Evidence & limitations",
            '🌧️ Rainfall and spills',
        ]

page_label = st.sidebar.radio("Navigate", PAGES)
page = page_label.split(" ", 1)[1]

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="font-size:.82rem;line-height:1.5;color:#5D7772;">
      <b style="color:#173D3A;">Responsible-use note</b><br>
      A 2026 category is a calibrated model forecast—not a confirmed future spill,
      pollution-volume estimate or statement of legal responsibility.
    </div>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

if page == "Overview":
    render_hero()

    observed = load_table("observed_locations")
    forecast = load_table("forecast_map_points")
    observed_kpis = load_table("observed_kpis")
    forecast_kpis = load_table("forecast_kpis")

    observed_high = int(observed.get("period_risk_category", pd.Series(dtype=str)).eq("High").sum())
    predicted_high = int(forecast.get("predicted_2026_risk", pd.Series(dtype=str)).eq("High").sum())
    review_count = int(forecast.get("confidence_flag", pd.Series(dtype=str)).ne("Higher confidence").sum()) if "confidence_flag" in forecast.columns else 0

    metric_cards(
        [
            {"label": "Mapped observed locations", "value": value_text(len(observed)), "note": "Verified 2023–2025 evidence", "accent": "#A8D8D0"},
            {"label": "Observed High risk", "value": value_text(observed_high), "note": "Highest recorded category", "accent": "#E9A7A7"},
            {"label": "Forecast-eligible locations", "value": value_text(len(forecast)), "note": "Known sites with 2026 forecasts", "accent": "#B7DDE5"},
            {"label": "Predictions to review", "value": value_text(review_count), "note": "Low confidence or close probabilities", "accent": "#F1D39D"},
        ]
    )

    banner(
        "<b>Observed evidence</b> comes from verified historical labels. "
        "<b>Predicted 2026 risk</b> comes from the selected calibrated model. "
        "The interface uses different wording and contextual warnings so the two cannot be confused.",
        icon="🛟",
        background=PALE_AMBER,
        edge="#D59A3C",
    )

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

    section_header("How the observatory works", "A transparent route from verified evidence to public decision support.")
    st.markdown(
        """
        <div class="edm-journey">
          <div class="edm-journey-step"><span class="edm-journey-number">1</span>
            <h4>Observe</h4><p>Clean and validate genuine EDM site-year records, labels and locations.</p></div>
          <div class="edm-journey-step"><span class="edm-journey-number">2</span>
            <h4>Forecast</h4><p>Use earlier-year evidence to estimate the next year's Low, Medium and High probabilities.</p></div>
          <div class="edm-journey-step"><span class="edm-journey-number">3</span>
            <h4>Investigate responsibly</h4><p>Explore patterns, uncertainty and environmental context without treating predictions as facts.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 2 — COMBINED OBSERVED/PREDICTED MAP EXPERIENCE
# =============================================================================

elif page == "Explore maps":
    section_header(
        "Interactive environmental risk map",
        "Switch between verified observations and model forecasts, then filter, cluster, inspect or view density.",
    )

    layer = st.radio(
        "Evidence layer",
        ["Observed 2023–2025", "Predicted 2026"],
        horizontal=True,
    )
    display_style = st.radio(
        "Map display",
        ["Clustered sites", "Location density"],
        horizontal=True,
    )

    prediction = layer == "Predicted 2026"
    table_name = "forecast_map_points" if prediction else "observed_locations"
    risk_column = "predicted_2026_risk" if prediction else "period_risk_category"
    map_data = load_table(table_name)

    if map_data.empty:
        st.error(f"The required {table_name} export is unavailable.")
    elif risk_column not in map_data.columns:
        st.error(f"The required risk field '{risk_column}' is missing.")
    else:
        if prediction:
            banner(
                "<b>Forecast layer:</b> markers show calibrated 2026 probabilities for known 2025 sites. "
                "They are not observed 2026 events.",
                icon="🔮",
                background=PALE_AMBER,
                edge="#D59A3C",
            )
        else:
            banner(
                "<b>Observed layer:</b> markers show verified Low, Medium and High categories from the supplied historical records.",
                icon="🌊",
                background=PALE_MINT,
                edge="#4A9C7D",
            )

        filtered, _ = filter_map(map_data, risk_column, prediction)
        risk_counts = filtered[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
        metric_cards(
            [
                {"label": "Filtered locations", "value": value_text(len(filtered)), "note": layer, "accent": "#B7DDE5"},
                {"label": "Low", "value": value_text(risk_counts["Low"]), "note": "● lower-risk category", "accent": "#A8D8D0"},
                {"label": "Medium", "value": value_text(risk_counts["Medium"]), "note": "◆ medium-risk category", "accent": "#F1D39D"},
                {"label": "High", "value": value_text(risk_counts["High"]), "note": "▲ high-risk category", "accent": "#E9A7A7"},
            ]
        )

        if filtered.empty:
            st.warning("No locations match the selected filters.")
        else:
            with st.spinner("Preparing the interactive map…"):
                map_object = build_folium_map(filtered, risk_column, prediction, display_style)
            st_folium(
                map_object,
                height=720,
                use_container_width=True,
                returned_objects=[],
                key=f"edm_{'forecast' if prediction else 'observed'}_{display_style}",
            )
            st.caption(
                "Cluster numbers show how many filtered locations overlap. Zoom to separate them; "
                "click a marker for site evidence, risk wording, coordinates and forecast probabilities where applicable."
            )
            with st.expander("View and export the filtered evidence table"):
                st.dataframe(filtered.head(5000), use_container_width=True, hide_index=True)
                download_table(filtered, "filtered_2026_predictions.csv" if prediction else "filtered_observed_locations.csv")


# =============================================================================
# PAGE 3 — PLACES, COMPANIES AND CHANGE
# =============================================================================

elif page == "Places & companies":
    section_header(
        "Places and water-company evidence",
        "Compare absolute burden, proportional burden, annual trends and observed-to-predicted change.",
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
                "Ranking measure",
                ["Absolute High-risk burden", "Proportional Medium-or-High burden"],
                horizontal=True,
            )
            if metric_choice.startswith("Absolute"):
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
# PAGE 4 — MODEL PERFORMANCE
# =============================================================================

elif page == "Model performance":
    section_header(
        "Model performance and validation",
        "Training-period cross-validation and chronological 2025 validation are presented separately.",
    )
    banner(
        "Macro F1 is the primary selection metric because it gives equal importance to Low, Medium and High. "
        "Balanced accuracy addresses class imbalance; High-risk recall measures sensitivity to genuine High-risk cases.",
        icon="🧠",
        background=PALE_LAVENDER,
        edge="#8F79A8",
    )

    models = load_table("model_comparison")
    metrics = load_table("validation_metrics")
    report = load_table("classification_report")

    if not models.empty:
        model_column = first_existing(models, ["Model", "model"])
        metric_columns = [
            column for column in models.columns
            if any(term in str(column).casefold() for term in ["macro f1", "balanced accuracy", "high-risk recall"])
            and pd.api.types.is_numeric_dtype(models[column])
        ]
        if model_column and metric_columns:
            long = models.melt(id_vars=model_column, value_vars=metric_columns, var_name="Metric", value_name="Score")
            figure = px.bar(
                long, x=model_column, y="Score", color="Metric", barmode="group",
                title="Training-period candidate-model comparison",
                color_discrete_sequence=["#70B7A5", "#8CBBD0", "#C4A8D5"],
            )
            figure.update_yaxes(range=[0, 1], tickformat=".0%")
            st.plotly_chart(plot_style(figure, 500), use_container_width=True)
        st.dataframe(models, use_container_width=True, hide_index=True)
    else:
        st.info("The model-comparison export is unavailable.")

    metric_tab, class_tab = st.tabs(["2025 validation metrics", "Class-level results"])
    with metric_tab:
        if not metrics.empty:
            st.dataframe(metrics.rename(columns=lambda value: str(value).replace("Test", "2025 validation")), use_container_width=True, hide_index=True)
        else:
            st.info("The validation-metric export is unavailable.")
    with class_tab:
        if not report.empty:
            st.dataframe(report, use_container_width=True, hide_index=True)
        else:
            st.info("The classification-report export is unavailable.")

    st.markdown(
        """
        <div class="edm-journey">
          <div class="edm-journey-step"><span class="edm-journey-number">1</span><h4>Train</h4><p>Fit preprocessing and candidate models on earlier-year transitions only.</p></div>
          <div class="edm-journey-step"><span class="edm-journey-number">2</span><h4>Validate</h4><p>Evaluate the selected configuration chronologically on 2025 outcomes.</p></div>
          <div class="edm-journey-step"><span class="edm-journey-number">3</span><h4>Forecast</h4><p>Refit on eligible history and generate calibrated probabilities for 2026.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 5 — INDIVIDUAL PREDICTION
# =============================================================================

elif page == "Individual prediction":
    section_header(
        "Individual 2026 risk prediction",
        "Enter previous-year information using the saved model's fitted preprocessing and calibrated probability pipeline.",
    )
    banner(
        "This calculator produces a model prediction for investigation and prioritisation. "
        "It does not confirm that a future event will occur.",
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

    st.caption(f"Selected model: {bundle.get('selected_model_name', 'Not recorded')}")
    values = {}

    with st.form("prediction_form"):
        st.subheader("Previous-year measurements")
        numeric_columns = metadata.get("numeric_columns", [])
        numeric_widgets = st.columns(2)
        for position, column in enumerate(numeric_columns):
            default = float(metadata.get("numeric_defaults", {}).get(column, 0.0) or 0.0)
            with numeric_widgets[position % 2]:
                values[column] = st.number_input(pretty(column), value=default, format="%.3f")

        st.subheader("Site characteristics")
        categorical_columns = metadata.get("categorical_columns", [])
        categorical_widgets = st.columns(2)
        for position, column in enumerate(categorical_columns):
            options = metadata.get("categorical_options", {}).get(column, ["__MISSING__"]) or ["__MISSING__"]
            display_options = ["Not recorded" if option == "__MISSING__" else option for option in options]
            with categorical_widgets[position % 2]:
                selected = st.selectbox(pretty(column), display_options)
                values[column] = "__MISSING__" if selected == "Not recorded" else selected

        submitted = st.form_submit_button("Calculate calibrated prediction", type="primary", use_container_width=True)

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

elif page == "Rainfall and spills":
    from pathlib import Path as _RainfallPath

    import pandas as _rainfall_pd
    import plotly.express as _rainfall_px

    _rainfall_directory = (
        _RainfallPath(__file__).resolve().parent
        / "data"
    )

    _annual_path = (
        _rainfall_directory
        / "rainfall_annual_2021_2025.csv.gz"
    )

    _monthly_path = (
        _rainfall_directory
        / "rainfall_monthly_2021_2025.csv.gz"
    )

    st.title(
        "Official rainfall and recorded spills"
    )

    st.caption(
        "Observed Met Office regional rainfall for 2021–2025. "
        "Rainfall provides supporting evidence of weather exposure, "
        "but does not prove why an individual outlet discharged."
    )

    if not _annual_path.exists():
        st.error(
            "The annual rainfall file is missing. "
            "Run Rainfall Cells 1 and 2 again."
        )

    elif not _monthly_path.exists():
        st.error(
            "The monthly rainfall file is missing. "
            "Run Rainfall Cells 1 and 2 again."
        )

    else:
        _annual = _rainfall_pd.read_csv(
            _annual_path
        )

        _monthly = _rainfall_pd.read_csv(
            _monthly_path
        )

        _annual["year"] = (
            _rainfall_pd.to_numeric(
                _annual["year"],
                errors="coerce",
            )
        )

        _monthly["year"] = (
            _rainfall_pd.to_numeric(
                _monthly["year"],
                errors="coerce",
            )
        )

        _regions = sorted(
            _annual["region"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        if not _regions:
            st.error(
                "No regional rainfall records were found."
            )

        else:
            _default_region = (
                _regions.index(
                    "England and Wales"
                )
                if "England and Wales"
                in _regions
                else 0
            )

            _selected_region = (
                st.selectbox(
                    "Choose an official rainfall region",
                    _regions,
                    index=_default_region,
                    key="official_rainfall_region",
                )
            )

            _annual_view = (
                _annual.loc[
                    _annual["region"]
                    .astype(str)
                    .eq(_selected_region)
                ]
                .sort_values("year")
                .copy()
            )

            _monthly_view = (
                _monthly.loc[
                    _monthly["region"]
                    .astype(str)
                    .eq(_selected_region)
                ]
                .copy()
            )

            if not _annual_view.empty:

                _latest = (
                    _annual_view.iloc[-1]
                )

                _latest_year = int(
                    _latest["year"]
                )

                (
                    _card1,
                    _card2,
                    _card3,
                    _card4,
                ) = st.columns(4)

                _card1.metric(
                    f"{_latest_year} rainfall",
                    (
                        f"{_latest['annual_rainfall_mm']:,.0f} mm"
                    ),
                )

                _card2.metric(
                    "Wet days",
                    f"{_latest['wet_days']:,.0f}",
                )

                _card3.metric(
                    "Heavy-rain days",
                    (
                        f"{_latest['heavy_rain_days']:,.0f}"
                    ),
                )

                _card4.metric(
                    "Wettest day",
                    (
                        f"{_latest['maximum_daily_rainfall_mm']:,.1f} mm"
                    ),
                )

                _annual_figure = (
                    _rainfall_px.bar(
                        _annual_view,
                        x="year",
                        y="annual_rainfall_mm",
                        color="annual_rainfall_mm",
                        text_auto=".0f",
                        color_continuous_scale=[
                            "#E8F5F2",
                            "#BDE0FE",
                            "#6C8EBF",
                        ],
                        title=(
                            "Annual rainfall — "
                            + _selected_region
                        ),
                        labels={
                            "year": "Year",
                            "annual_rainfall_mm":
                                "Rainfall (mm)",
                        },
                    )
                )

                _annual_figure.update_layout(
                    template="plotly_white",
                    coloraxis_showscale=False,
                )

                st.plotly_chart(
                    _annual_figure,
                    use_container_width=True,
                )

            if not _monthly_view.empty:

                _monthly_view["month"] = (
                    _rainfall_pd.to_numeric(
                        _monthly_view["month"],
                        errors="coerce",
                    )
                )

                _monthly_view["period"] = (
                    _rainfall_pd.to_datetime(
                        _monthly_view["year"]
                        .astype("Int64")
                        .astype(str)
                        + "-"
                        + _monthly_view["month"]
                        .astype("Int64")
                        .astype(str)
                        + "-01",
                        errors="coerce",
                    )
                )

                _monthly_view = (
                    _monthly_view
                    .dropna(
                        subset=[
                            "period",
                            "monthly_rainfall_mm",
                        ]
                    )
                    .sort_values("period")
                )

                _monthly_figure = (
                    _rainfall_px.area(
                        _monthly_view,
                        x="period",
                        y="monthly_rainfall_mm",
                        color_discrete_sequence=[
                            "#78B7C5"
                        ],
                        title=(
                            "Monthly rainfall pattern — "
                            + _selected_region
                        ),
                        labels={
                            "period": "Month",
                            "monthly_rainfall_mm":
                                "Rainfall (mm)",
                        },
                    )
                )

                _monthly_figure.update_layout(
                    template="plotly_white",
                )

                st.plotly_chart(
                    _monthly_figure,
                    use_container_width=True,
                )

            st.info(
                "Wetter periods may increase hydraulic pressure "
                "on combined sewer systems. This comparison is "
                "descriptive and does not establish the cause of "
                "a particular outlet's discharge."
            )

            st.markdown(
                "Source: [Met Office HadUKP daily precipitation]"
                "(https://www.metoffice.gov.uk/hadobs/"
                "hadukp/data/download.html)"
            )


# =============================================================================
# PAGE 6 — WATER QUALITY, AUDIT, SEARCH AND LIMITATIONS
# =============================================================================

else:
    section_header(
        "Environmental evidence, audit and limitations",
        "Inspect nearby-station observations, data-quality evidence, searchable records and responsible-use boundaries.",
    )

    water_tab, audit_tab, search_tab, method_tab = st.tabs(
        ["Water quality", "Data-quality audit", "Evidence search", "Method & limitations"]
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
        st.markdown(
            """
            <div class="edm-journey">
              <div class="edm-journey-step"><span class="edm-journey-number">1</span><h4>Verified observations</h4><p>Observed categories come from cleaned historical records and remain distinguishable from forecasts.</p></div>
              <div class="edm-journey-step"><span class="edm-journey-number">2</span><h4>Chronological modelling</h4><p>Earlier-year measurements predict the following year's category; later outcomes assess performance.</p></div>
              <div class="edm-journey-step"><span class="edm-journey-number">3</span><h4>Traceable presentation</h4><p>Maps, rankings, probabilities, uncertainty flags and audit tables retain source context.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
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


st.markdown(
    """
    <div style="margin-top:2.5rem;padding-top:1rem;border-top:1px solid rgba(55,120,110,.18);
                color:#5D7772;font-size:.78rem;text-align:center;">
      England EDM Water &amp; Spill-Risk Observatory · verified evidence, transparent forecasts and responsible interpretation
    </div>
    """,
    unsafe_allow_html=True,
)
