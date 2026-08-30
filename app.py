from __future__ import annotations

import base64
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


DASHBOARD_RELEASE = "2026-08-30-full-2021-2025-dashboard-v16"

OBSERVED_YEARS = tuple(range(2021, 2026))
BASELINE_YEARS = tuple(year for year in OBSERVED_YEARS if year < 2025)
OBSERVED_PERIOD = "2021–2025"
BASELINE_PERIOD = "2021–2024"


# =============================================================================
# PAGE AND PROJECT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Sewage Overflow Insights",
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

      /* Keep Streamlit's Share controls visible without allowing the fixed
         toolbar to overlap the title at the top of any dashboard page. */
      header[data-testid="stHeader"] {
        height: 2.85rem !important;
        min-height: 2.85rem !important;
        background: rgba(251, 253, 249, 0.96) !important;
        border-bottom: 1px solid rgba(55, 120, 110, 0.10);
      }

      header[data-testid="stHeader"] [data-testid="stToolbar"] {
        min-height: 2.85rem !important;
        height: 2.85rem !important;
        align-items: center !important;
      }

      .block-container {
        width: 100%;
        max-width: 1900px;
        padding-top: 3.35rem;
        padding-left: 1rem;
        padding-right: 1rem;
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
        grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
        gap: 1.4rem;
        align-items: center;
        min-height: 345px;
        padding: 2rem 2.2rem;
        margin: 0.15rem 0 1.15rem;
        border: 1px solid rgba(61, 129, 118, 0.22);
        border-radius: 30px;
        background:
          radial-gradient(circle at 9% 16%, rgba(255,255,255,.92), transparent 15rem),
          radial-gradient(circle at 82% 14%, rgba(255,239,190,.55), transparent 13rem),
          linear-gradient(125deg, rgba(224,245,235,.99), rgba(220,241,248,.98) 58%, rgba(235,239,249,.98));
        box-shadow: 0 22px 55px rgba(37,92,84,.13);
      }

      .edm-hero::before,
      .edm-hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        pointer-events: none;
      }

      .edm-hero::before {
        width: 170px;
        height: 170px;
        left: -62px;
        bottom: -74px;
        border: 24px solid rgba(105,185,177,.13);
      }

      .edm-hero::after {
        width: 120px;
        height: 120px;
        right: 32%;
        top: -75px;
        border: 18px solid rgba(104,175,194,.12);
      }

      .edm-hero h1 {
        margin: 0;
        max-width: 880px;
        color: var(--edm-ink);
        font-size: clamp(2.15rem, 4vw, 4rem);
        line-height: 1.02;
        letter-spacing: 0;
      }

      .edm-hero p {
        max-width: 850px;
        margin: 0.8rem 0 0;
        color: var(--edm-muted);
        font-size: 1.08rem;
        line-height: 1.55;
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
        min-height: 330px;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: .5rem;
        border-radius: 28px;
        background: rgba(255,255,255,.48);
        border: 1px solid rgba(84,145,137,.18);
        box-shadow: inset 0 0 35px rgba(255,255,255,.48), 0 12px 30px rgba(49,104,96,.10);
      }

      .edm-sewer-diagram {
        position: relative;
        width: 100%;
        height: 320px;
        overflow: hidden;
        border: 1px solid rgba(53,112,104,.18);
        border-radius: 22px;
        background: linear-gradient(to bottom, #EAF8FC 0 48%, #E7DDC9 48% 100%);
        color: #173D3A;
      }

      .edm-diagram-title {
        position: absolute;
        z-index: 8;
        top: 12px;
        left: 16px;
        right: 16px;
        padding: 7px 10px;
        border-radius: 12px;
        background: rgba(255,255,255,.88);
        color: #204E49;
        font-size: .78rem;
        font-weight: 850;
        text-align: center;
        box-shadow: 0 4px 12px rgba(41,96,89,.08);
      }

      .edm-hero-raindrop {
        position: absolute;
        z-index: 4;
        top: 56px;
        color: #4FA9CE;
        font-size: 18px;
        line-height: 1;
        animation: edmRain 1.6s ease-in-out infinite;
      }

      .edm-hero-raindrop.r1 { left: 10%; }
      .edm-hero-raindrop.r2 { left: 20%; animation-delay: .35s; }
      .edm-hero-raindrop.r3 { left: 34%; animation-delay: .7s; }
      .edm-hero-raindrop.r4 { left: 48%; animation-delay: .2s; }
      .edm-hero-raindrop.r5 { left: 61%; animation-delay: .8s; }

      .edm-house {
        position: absolute;
        z-index: 5;
        left: 8%;
        top: 94px;
        width: 84px;
        height: 54px;
        border: 3px solid #47766F;
        border-radius: 5px 5px 2px 2px;
        background: #FFF8E8;
      }

      .edm-house::before {
        content: "";
        position: absolute;
        left: 7px;
        top: -34px;
        width: 62px;
        height: 62px;
        border-top: 3px solid #47766F;
        border-left: 3px solid #47766F;
        background: #D7A99D;
        transform: rotate(45deg);
        z-index: -1;
      }

      .edm-house::after {
        content: "Wastewater";
        position: absolute;
        left: 6px;
        bottom: 6px;
        color: #315F5A;
        font-size: 10px;
        font-weight: 750;
      }

      .edm-door {
        position: absolute;
        right: 9px;
        bottom: 0;
        width: 18px;
        height: 30px;
        border: 2px solid #47766F;
        background: #DCEDE5;
      }

      .edm-road {
        position: absolute;
        z-index: 3;
        top: 152px;
        left: 0;
        right: 0;
        height: 36px;
        border-top: 5px solid #94B58D;
        border-bottom: 4px solid #71817E;
        background: #AAB7B5;
      }

      .edm-road::after {
        content: "RAIN + WASTEWATER ENTER ONE PIPE";
        position: absolute;
        left: 34%;
        top: 8px;
        color: #FFFFFF;
        font-size: 9px;
        font-weight: 850;
        letter-spacing: .05em;
      }

      .edm-drain {
        position: absolute;
        z-index: 6;
        left: 46%;
        top: 158px;
        width: 38px;
        height: 12px;
        border-radius: 3px;
        background: repeating-linear-gradient(90deg, #334E4B 0 3px, #8CA09D 3px 7px);
      }

      .edm-house-connector,
      .edm-drain-connector {
        position: absolute;
        z-index: 2;
        width: 13px;
        border: 3px solid #5B716D;
        background: #9C6A48;
      }

      .edm-house-connector { left: 24%; top: 138px; height: 91px; }
      .edm-drain-connector { left: 49%; top: 166px; height: 63px; background: #69ADBE; }

      .edm-main-pipe {
        position: absolute;
        z-index: 5;
        left: 8%;
        top: 216px;
        width: 66%;
        height: 46px;
        overflow: hidden;
        border: 5px solid #5A6C69;
        border-radius: 24px;
        background: #EEF2EF;
        box-shadow: 0 5px 10px rgba(62,77,73,.12);
      }

      .edm-main-flow {
        position: absolute;
        left: 4px;
        right: 4px;
        bottom: 4px;
        height: 17px;
        border-radius: 12px;
        background: repeating-linear-gradient(110deg, #8B5A3C 0 20px, #A56B42 20px 38px);
        animation: edmRiver 3s linear infinite;
      }

      .edm-main-pipe-label {
        position: absolute;
        z-index: 7;
        left: 19%;
        top: 229px;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 900;
        letter-spacing: .04em;
        text-transform: uppercase;
      }

      .edm-overflow-chamber {
        position: absolute;
        z-index: 7;
        left: 68%;
        top: 200px;
        width: 48px;
        height: 76px;
        border: 4px solid #4E6863;
        border-radius: 8px 8px 18px 18px;
        background: linear-gradient(to bottom, #EDF4F1 0 42%, #8B5A3C 42% 100%);
      }

      .edm-overflow-chamber span {
        position: absolute;
        left: 50%;
        top: -18px;
        width: 92px;
        transform: translateX(-50%);
        color: #315F5A;
        font-size: 9px;
        font-weight: 850;
        text-align: center;
      }

      .edm-overflow-pipe {
        position: absolute;
        z-index: 6;
        left: 74%;
        top: 243px;
        width: 99px;
        height: 20px;
        border: 4px solid #5A6C69;
        border-radius: 12px;
        background: #8B5A3C;
        transform: rotate(17deg);
        transform-origin: left center;
      }

      .edm-discharge-label {
        position: absolute;
        z-index: 9;
        right: 5px;
        top: 218px;
        padding: 4px 7px;
        border-radius: 8px;
        background: #FFF5EA;
        color: #7A452C;
        font-size: 9px;
        font-weight: 900;
      }

      .edm-river {
        position: absolute;
        z-index: 4;
        right: -18px;
        bottom: -20px;
        width: 150px;
        height: 75px;
        border: 5px solid rgba(62,145,167,.35);
        border-radius: 55% 45% 0 0;
        background: repeating-linear-gradient(165deg, #8FD0DE 0 15px, #65B4C8 15px 28px);
      }

      .edm-river span {
        position: absolute;
        left: 28px;
        top: 9px;
        color: #FFFFFF;
        font-size: 10px;
        font-weight: 900;
      }

      .edm-treatment-route {
        position: absolute;
        z-index: 4;
        left: 18%;
        top: 257px;
        width: 12px;
        height: 24px;
        border: 3px solid #4D7770;
        background: #73B6A4;
      }

      .edm-treatment {
        position: absolute;
        z-index: 6;
        left: 7%;
        bottom: 7px;
        width: 125px;
        padding: 7px 8px;
        border: 2px solid #4C8176;
        border-radius: 11px;
        background: #DDF0E4;
        color: #285B54;
        font-size: 10px;
        font-weight: 850;
        text-align: center;
      }

      /* Static, label-free combined-sewer illustration used in the hero. */
      .edm-simple-sewer-art {
        position: relative;
        width: 100%;
        height: 310px;
        overflow: hidden;
        border: 1px solid rgba(53,112,104,.18);
        border-radius: 22px;
        background: linear-gradient(to bottom, #EAF8FC 0 46%, #E7DDC9 46% 100%);
      }

      .edm-simple-house {
        position: absolute;
        z-index: 5;
        width: 78px;
        height: 58px;
        border: 3px solid #54746F;
        border-radius: 5px 5px 2px 2px;
        background: #FFF8E8;
      }

      .edm-simple-house::before {
        content: "";
        position: absolute;
        left: 7px;
        top: -31px;
        width: 58px;
        height: 58px;
        border-top: 3px solid #54746F;
        border-left: 3px solid #54746F;
        background: #CFA99E;
        transform: rotate(45deg);
        z-index: -1;
      }

      .edm-simple-house-a { left: 8%; top: 75px; }
      .edm-simple-house-b { left: 35%; top: 91px; transform: scale(.82); }

      .edm-simple-window {
        position: absolute;
        left: 10px;
        top: 15px;
        width: 19px;
        height: 18px;
        border: 2px solid #54746F;
        background: #C9E7ED;
      }

      .edm-simple-door {
        position: absolute;
        right: 9px;
        bottom: 0;
        width: 19px;
        height: 32px;
        border: 2px solid #54746F;
        background: #DCEDE5;
      }

      .edm-simple-ground {
        position: absolute;
        z-index: 3;
        top: 144px;
        left: 0;
        right: 0;
        height: 34px;
        border-top: 5px solid #94B58D;
        border-bottom: 4px solid #768885;
        background: #AEBBB8;
      }

      .edm-simple-drain {
        position: absolute;
        z-index: 7;
        left: 54%;
        top: 150px;
        width: 40px;
        height: 12px;
        border-radius: 3px;
        background: repeating-linear-gradient(90deg, #344D4A 0 3px, #91A39F 3px 7px);
      }

      .edm-simple-connector {
        position: absolute;
        z-index: 4;
        width: 15px;
        border: 4px solid #586B68;
        background: #DDE6E2;
      }

      .edm-simple-connector-a { left: 22%; top: 128px; height: 94px; }
      .edm-simple-connector-b { left: 44%; top: 137px; height: 85px; }
      .edm-simple-connector-c { left: 57%; top: 160px; height: 62px; }

      .edm-static-main-pipe {
        position: absolute;
        z-index: 6;
        left: 7%;
        top: 210px;
        width: 68%;
        height: 48px;
        overflow: hidden;
        border: 6px solid #586B68;
        border-radius: 26px;
        background: #ECF1EF;
        box-shadow: 0 5px 10px rgba(62,77,73,.13);
      }

      .edm-static-main-water {
        position: absolute;
        left: 5px;
        right: 5px;
        bottom: 5px;
        height: 13px;
        border-radius: 9px;
        background: #9CCFD8;
      }

      .edm-static-chamber {
        position: absolute;
        z-index: 8;
        left: 69%;
        top: 197px;
        width: 46px;
        height: 73px;
        border: 5px solid #526A66;
        border-radius: 9px 9px 19px 19px;
        background: linear-gradient(to bottom, #EDF3F1 0 52%, #A7C9C5 52% 100%);
      }

      .edm-static-outfall-pipe {
        position: absolute;
        z-index: 7;
        left: 75%;
        top: 238px;
        width: 118px;
        height: 26px;
        overflow: hidden;
        border: 5px solid #586B68;
        border-radius: 14px;
        background: #E7ECEA;
        transform: rotate(16deg);
        transform-origin: left center;
      }

      .edm-outfall-stain {
        position: absolute;
        right: -4px;
        bottom: 3px;
        width: 65%;
        height: 11px;
        border-radius: 8px;
        background: linear-gradient(90deg, rgba(139,90,60,.18), #8B5A3C 72%);
      }

      .edm-static-receiving-water {
        position: absolute;
        z-index: 5;
        right: -20px;
        bottom: -20px;
        width: 170px;
        height: 80px;
        overflow: hidden;
        border: 5px solid rgba(62,145,167,.34);
        border-radius: 60% 40% 0 0;
        background: repeating-linear-gradient(165deg, #A8DDE5 0 18px, #79C3D1 18px 34px);
      }

      .edm-water-stain {
        position: absolute;
        left: 8px;
        top: 7px;
        width: 74px;
        height: 34px;
        border-radius: 50%;
        background: radial-gradient(ellipse at left, rgba(139,90,60,.62), rgba(139,90,60,.20) 55%, transparent 76%);
      }

      .edm-hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: .55rem;
        margin-top: 1.1rem;
      }

      .edm-hero-badges span {
        display: inline-flex;
        align-items: center;
        padding: .42rem .68rem;
        border: 1px solid rgba(51,117,108,.16);
        border-radius: 999px;
        background: rgba(255,255,255,.72);
        color: #315F5A;
        font-size: .78rem;
        font-weight: 760;
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

      .edm-quality-guide {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }

      .edm-quality-flow {
        display: grid;
        grid-template-columns: 1fr auto 1fr auto 1fr;
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
        min-height: 94px;
        padding: 1rem 1.1rem;
        border-radius: 19px;
        background: linear-gradient(145deg, rgba(255,255,255,.94), var(--risk-tint));
        border-top: 8px solid var(--risk-colour);
        box-shadow: 0 10px 25px rgba(38,91,84,.08);
      }

      .edm-risk-guide b {
        display: block;
        margin-bottom: .2rem;
        color: var(--edm-ink);
        font-size: 1.05rem;
      }

      .edm-home-chart-note {
        margin: .35rem 0 .85rem;
        padding: .78rem 1rem;
        border: 1px solid rgba(55,120,110,.14);
        border-radius: 15px;
        color: #416B66;
        background: linear-gradient(110deg, rgba(235,248,242,.88), rgba(233,245,249,.88));
        font-size: .9rem;
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

      /* The dashboard has one navigation menu: a compact vertical menu in
         Streamlit's sidebar. These rules override the horizontal radio style
         used by filters inside the main page. */
      section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        gap: .34rem !important;
        padding: .3rem 0 !important;
        margin: .15rem 0 .65rem !important;
        border: 0 !important;
        border-radius: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
      }

      section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label {
        flex: 0 0 auto !important;
        width: 100% !important;
        min-height: 40px !important;
        justify-content: flex-start !important;
        padding: .42rem .55rem !important;
        margin: 0 !important;
        border: 1px solid rgba(71,139,164,.24) !important;
        border-radius: 11px !important;
        background: linear-gradient(135deg, #E7F5FA, #D8ECF5) !important;
        box-shadow: 0 3px 9px rgba(58,123,148,.07) !important;
        font-size: .84rem !important;
        font-weight: 720 !important;
        text-align: left !important;
        transform: none !important;
      }

      section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label:hover {
        background: linear-gradient(135deg, #DCEFF7, #CBE6F1) !important;
        border-color: rgba(55,126,153,.42) !important;
      }

      section[data-testid="stSidebar"] div[data-testid="stRadio"] > div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(135deg, #CFEAF4, #BFDDEA) !important;
        border-color: rgba(48,118,145,.48) !important;
        box-shadow: 0 4px 12px rgba(48,118,145,.13) !important;
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

      iframe[title="streamlit_folium.st_folium"] {
        width: 100% !important;
        border: 1px solid rgba(48,112,103,.20) !important;
        border-radius: 24px !important;
        box-shadow: 0 18px 42px rgba(34,82,75,.14) !important;
        background: linear-gradient(145deg,#DDF2F5,#E8F3EA) !important;
      }

      @media (max-width: 1050px) {
        .block-container { padding-right: 1rem; }
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
        .edm-quality-guide, .edm-quality-flow { grid-template-columns: 1fr !important; }
        .edm-quality-flow > div:nth-child(2),
        .edm-quality-flow > div:nth-child(4) { transform: rotate(90deg); }
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
          <div style="--risk-colour:#4A9C7D;--risk-tint:#E7F4EC;"><b>💧 &#9679; Low</b>Lower concern</div>
          <div style="--risk-colour:#E2A45C;--risk-tint:#FFF1D8;"><b>💧 &#9670; Medium</b>Closer attention</div>
          <div style="--risk-colour:#D66565;--risk-tint:#FBE5E6;"><b>💧 &#9650; High</b>Priority review</div>
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
    sewer_figure_path = (
        ROOT / "assets" / "combined_sewer_overflow_professional_v1.png"
    )
    if sewer_figure_path.exists():
        encoded_figure = base64.b64encode(
            sewer_figure_path.read_bytes()
        ).decode("ascii")
        sewer_figure_html = f"""
          <figure style="margin:0; width:100%;">
            <img
              src="data:image/png;base64,{encoded_figure}"
              alt="Professional diagram explaining how rainfall and household wastewater enter a combined sewer, with normal flow to treatment and a storm-overflow route to a receiving river during heavy rainfall."
              style="display:block; width:100%; height:auto; border-radius:22px; border:1px solid rgba(53,112,104,.18);"
            />
          </figure>
        """
    else:
        sewer_figure_html = """
          <div role="status" style="padding:2rem; border-radius:22px; background:#EAF6F0; color:#173D3A;">
            Combined sewer process figure is temporarily unavailable.
          </div>
        """

    st.html(
        f"""
        <section class="edm-hero">
          <div>
            <div class="edm-kicker">💧 England and Wales</div>
            <h1>Sewage Overflow Insights</h1>
            <p>
              Explore mapped discharge outlets, receiving waters and recorded 2021–2025 risk,
              then view the separate, clearly labelled 2026 forecast.
            </p>
            <div class="edm-hero-badges" aria-label="Dashboard highlights">
              <span>Mapped receiving waters</span>
              <span>Exact discharge outlets</span>
              <span>Separate 2026 forecast</span>
            </div>
          </div>
          <div class="edm-water-art">
            {sewer_figure_html}
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
            <div class="edm-page-icon">💧</div><h3>Interactive maps</h3>
            <p>Recorded outlets and 2026 estimates.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#FBE4E4;">
            <div class="edm-page-icon">💧</div><h3>Priority locations</h3>
            <p>High-risk places, outlets and companies.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#E5F3EA;">
            <div class="edm-page-icon">💧</div><h3>Places &amp; companies</h3>
            <p>Simple rankings and yearly patterns.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#FFF1D8;">
            <div class="edm-page-icon">💧</div><h3>Improvements &amp; changes</h3>
            <p>See where counted spills rose or fell.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#F0EAF6;">
            <div class="edm-page-icon">💧</div><h3>2026 predictions</h3>
            <p>Forecast risks and affected locations.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#FFF0DD;">
            <div class="edm-page-icon">💧</div><h3>Find a location</h3>
            <p>Search a site and view its probabilities.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#E7F4F6;">
            <div class="edm-page-icon">🌧️</div><h3>Rainfall and spills</h3>
            <p>Explore official 2021–2025 regional rainfall measurements.</p>
          </div>
          <div class="edm-page-card" style="--page-tint:#EDF3DE;">
            <div class="edm-page-icon">💧</div><h3>Evidence</h3>
            <p>Sources, quality checks and limitations.</p>
          </div>
        </div>
        """
    )
    destinations = [
        ("Open map", "Explore the map"),
        ("Priority list", "Priority locations"),
        ("Compare", "Places and companies"),
        ("Changes", "Improvements and changes"),
        ("2026 forecast", "2026 predictions"),
        ("Rainfall", "Rainfall and spills"),
        ("Find a site", "Check one location"),
        ("Evidence", "About the evidence"),
    ]
    for row_start in range(0, len(destinations), 4):
        row_destinations = destinations[row_start:row_start + 4]
        for column, (button_text, destination) in zip(
            st.columns(4), row_destinations
        ):
            with column:
                st.button(
                    button_text,
                    key=f"home_{destination}",
                    use_container_width=True,
                    on_click=lambda target=destination: st.session_state.update(
                        sidebar_navigation=target
                    ),
                )


def render_sewer_story():
    """Show the professional combined-sewer process figure."""
    sewer_figure_path = (
        ROOT / "assets" / "combined_sewer_overflow_professional_v1.png"
    )
    if sewer_figure_path.exists():
        st.image(
            str(sewer_figure_path),
            caption=(
                "Illustrative combined-sewer process. The diagram explains the "
                "system generally and is not evidence about a particular site."
            ),
            use_container_width=True,
        )
    else:
        st.warning(
            "The combined sewer process figure is unavailable. "
            "Check that assets/combined_sewer_overflow_professional_v1.png "
            "was deployed with app.py."
        )
    return

    # Retained below only as a legacy fallback reference; it is not rendered.
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


def mapped_annual_spill_totals(
    frame: pd.DataFrame,
    place_column: str | None,
) -> dict[int, float]:
    """Return annual counted spills without repeating town/city totals per outlet."""
    years = OBSERVED_YEARS
    annual_columns = [f"place_counted_spills_{year}" for year in years]
    if (
        frame.empty
        or not place_column
        or place_column not in frame.columns
        or not all(column in frame.columns for column in annual_columns)
    ):
        return {year: np.nan for year in years}

    # Each mapped outlet carries its town/city's annual total. Keep one row per
    # place before adding the totals so the same spills are never counted once
    # for every outlet in that place.
    place_totals = (
        frame[[place_column, *annual_columns]]
        .dropna(subset=[place_column])
        .drop_duplicates(subset=[place_column])
    )
    return {
        year: pd.to_numeric(
            place_totals[f"place_counted_spills_{year}"],
            errors="coerce",
        ).sum(min_count=1)
        for year in years
    }


def normalised_record_key(value) -> str:
    """Create a stable key for IDs read from compressed CSV files."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if pd.notna(numeric) and float(numeric).is_integer():
        return str(int(numeric))
    return text.casefold()


def water_quality_site_key(company, site) -> str:
    company_key = "" if company is None or pd.isna(company) else str(company).strip().casefold()
    site_key = "" if site is None or pd.isna(site) else str(site).strip().casefold()
    return f"{company_key}||{site_key}" if company_key and site_key else ""


def water_quality_popup_panel(records: pd.DataFrame) -> str:
    """Summarise linked 2025 nearby-station observations for one outlet."""
    if records.empty:
        return ""

    parameter_column = first_existing(
        records,
        ["project_parameter_name", "determinand_name", "parameter"],
    )
    result_column = first_existing(
        records,
        ["result_as_reported", "result_numeric", "exact_numeric_result"],
    )
    if not parameter_column or not result_column:
        return ""

    working = records.copy()
    date_column = first_existing(
        working,
        ["measurement_datetime", "measurement_date", "sample_date"],
    )
    if date_column:
        working["_measurement_date"] = pd.to_datetime(
            working[date_column],
            errors="coerce",
        )
        dated = working.loc[working["_measurement_date"].dt.year.eq(2025)]
        if not dated.empty:
            working = dated
    elif "measurement_year" in working.columns:
        year_values = pd.to_numeric(working["measurement_year"], errors="coerce")
        dated = working.loc[year_values.eq(2025)]
        if not dated.empty:
            working = dated
        working["_measurement_date"] = pd.NaT
    else:
        working["_measurement_date"] = pd.NaT

    working = working.loc[working[parameter_column].notna()].copy()
    if working.empty:
        return ""

    station_column = first_existing(
        working,
        ["sampling_point_name", "monitoring_station_name", "sampling_point_id"],
    )
    distance_column = first_existing(
        working,
        ["station_distance_km", "monitoring_station_distance_km"],
    )
    unit_column = first_existing(working, ["reported_unit", "unit"])
    observation_column = first_existing(working, ["observation_id", "measurement_id"])

    station_text = "Nearby Environment Agency monitoring station"
    if station_column:
        station_values = (
            working[station_column]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda values: values.ne("")]
            .unique()
            .tolist()
        )
        if station_values:
            station_text = ", ".join(station_values[:2])
            if len(station_values) > 2:
                station_text += f" and {len(station_values) - 2} more"

    distance_text = ""
    if distance_column:
        distances = pd.to_numeric(working[distance_column], errors="coerce").dropna()
        if not distances.empty:
            distance_text = f" · nearest station {float(distances.min()):.2f} km from outlet"

    indicator_rows = []
    for parameter, parameter_records in working.groupby(parameter_column, dropna=False):
        parameter_records = parameter_records.sort_values(
            "_measurement_date",
            ascending=False,
            na_position="last",
        )
        reported = parameter_records.loc[
            parameter_records[result_column].notna()
            & parameter_records[result_column].astype(str).str.strip().ne("")
        ]
        latest = reported.iloc[0] if not reported.empty else parameter_records.iloc[0]
        result = safe_text(latest.get(result_column))
        unit = safe_text(latest.get(unit_column), "") if unit_column else ""
        measured_on = latest.get("_measurement_date")
        date_text = (
            pd.Timestamp(measured_on).strftime("%d %b %Y")
            if pd.notna(measured_on)
            else "2025 date not reported"
        )
        observation_count = (
            int(parameter_records[observation_column].nunique())
            if observation_column
            else int(len(parameter_records))
        )
        value_with_unit = f"{result} {unit}".strip()
        indicator_rows.append(
            "<div style='padding:6px 7px;margin:4px 0;background:#FFFFFF;"
            "border:1px solid #C8DDD7;border-radius:7px'>"
            f"<b>{safe_text(parameter)}</b><br>"
            f"Latest reported result: {value_with_unit}<br>"
            f"<span style='font-size:11px;color:#5D7772'>{date_text} · "
            f"{observation_count:,} measurement{'s' if observation_count != 1 else ''}</span>"
            "</div>"
        )

    return f"""
      <div style="margin-top:8px;padding:9px;background:#E9F4F8;border-radius:9px;">
        <b>2025 nearby water-quality measurements</b><br>
        <span style="font-size:11px;color:#456F73">{safe_text(station_text)}{distance_text}</span>
        <div style="max-height:220px;overflow:auto;margin-top:5px">{''.join(indicator_rows)}</div>
        <div style="margin-top:6px;font-size:10px;color:#5D7772">
          These are measurements at a nearby monitoring station. Proximity does not prove
          that this outlet caused the result. Dissolved oxygen is a water-quality indicator,
          not itself a pollutant.
        </div>
      </div>
    """


@st.cache_data(show_spinner=False)
def water_quality_popup_lookups() -> dict[str, dict[str, str]]:
    """Build fast popup lookups from the public 2025 water-quality export."""
    quality = load_table("water_quality_records")
    empty = {"by_location": {}, "by_site": {}}
    if quality.empty:
        return empty

    by_location: dict[str, str] = {}
    if "location_id" in quality.columns:
        quality["_location_key"] = quality["location_id"].map(normalised_record_key)
        for key, records in quality.loc[quality["_location_key"].ne("")].groupby("_location_key"):
            panel = water_quality_popup_panel(records)
            if panel:
                by_location[str(key)] = panel

    company_column = first_existing(quality, ["company", "water_company_name"])
    site_column = first_existing(
        quality,
        ["site_name", "source_site_name_ea_consents_database"],
    )
    by_site: dict[str, str] = {}
    if company_column and site_column:
        quality["_site_key"] = [
            water_quality_site_key(company, site)
            for company, site in zip(quality[company_column], quality[site_column])
        ]
        for key, records in quality.loc[quality["_site_key"].ne("")].groupby("_site_key"):
            panel = water_quality_popup_panel(records)
            if panel:
                by_site[str(key)] = panel

    return {"by_location": by_location, "by_site": by_site}


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


def risk_donut(
    frame: pd.DataFrame,
    risk_column: str,
    title: str,
    centre_label: str,
) -> go.Figure:
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
            hole=0.61,
            sort=False,
            direction="clockwise",
            rotation=-92,
            marker=dict(
                colors=[RISK_COLOURS[label] for label in counts.index],
                line=dict(color="#FBFDF9", width=4),
            ),
            textinfo="label+percent",
            textfont=dict(size=14, color=INK),
            insidetextorientation="horizontal",
            pull=[0, 0.012, 0.035],
            hovertemplate=(
                "%{label} risk category<br>"
                "%{value:,} discharge outlets<br>"
                "%{percent}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title=title,
        annotations=[
            dict(
                text=(
                    f"<b>{int(counts.sum()):,}</b><br>"
                    f"<span style='font-size:12px'>{centre_label}</span>"
                ),
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color=INK),
            )
        ],
        showlegend=False,
    )
    return plot_style(figure, height=455)


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
    years_observed = row.get("years_observed", OBSERVED_PERIOD)
    risk_history = row.get("risk_history", "Not available")
    risk = row.get(risk_column, "Not available")
    relationship = safe_text(row.get("official_place_relationship"))
    distance = value_text(row.get("distance_to_official_place_km"), 2, " km")
    x_coordinate = value_text(row.get("easting_x"), 1)
    y_coordinate = value_text(row.get("northing_y"), 1)
    annual_boxes = []
    for year in OBSERVED_YEARS:
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
            for year in OBSERVED_YEARS:
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
                    aria-label="Show the 2021 to 2025 spill trend for {html.escape(company_name, quote=True)}">
              <span class="edm-map-rank-number">{int(row['Rank'])}</span>
              <b>{html.escape(company_name)}</b>
              <div><span class="risk-high">&#9650; {int(row.get('High', 0)):,}</span>
              <span class="risk-medium">&#9670; {int(row.get('Medium', 0)):,}</span>
              <span class="risk-low">&#9679; {int(row.get('Low', 0)):,}</span></div>
              <small>View 2021–2025 spill trend</small>
            </button>
            """
        )

    risk_counts = plotting[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
    map_title = "Predicted 2026 risk" if prediction else "Observed spill risk"
    period_text = "Forecast - not a confirmed event" if prediction else "Recorded 2021–2025 evidence"
    place_detail_label = "Forecast status" if prediction else "Recorded spills"
    panels = f"""
    <style>
      .edm-map-panel {{position:fixed;top:12px;z-index:9999;width:300px;max-height:86vh;
        overflow:auto;padding:12px;background:rgba(251,253,249,.96);color:#173D3A;
        border:1px solid #AFCFC6;border-radius:14px;box-shadow:0 7px 25px rgba(28,77,70,.19);
        font:12px/1.38 'Atkinson Hyperlegible',Verdana,Arial,sans-serif;
        transition:transform .28s ease,opacity .22s ease;}}
      #edm-map-left {{left:12px;height:calc(86vh - 24px);overflow:hidden;display:flex;flex-direction:column;}}
      #edm-map-right {{right:12px;top:12px;bottom:auto;}}
      .edm-map-title {{margin:-12px -12px 8px;padding:10px 12px;border-radius:13px 13px 0 0;
        color:#173D3A;background:linear-gradient(120deg,#CFEAE3,#DDEFF4);font-size:16px;font-weight:800;
        display:flex;align-items:center;justify-content:space-between;gap:8px;}}
      .edm-panel-close {{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;
        flex:0 0 28px;padding:0;border:1px solid rgba(40,100,93,.20);border-radius:50%;cursor:pointer;
        background:rgba(255,255,255,.78);color:#245B61;font-size:18px;font-weight:800;line-height:1;}}
      .edm-panel-close:hover,.edm-panel-close:focus {{background:#FFFFFF;box-shadow:0 3px 9px rgba(28,77,70,.14);}}
      #edm-panel-toggle {{display:none;position:fixed;top:12px;left:12px;z-index:10001;padding:9px 13px;
        border:1px solid #8FBDB2;border-radius:999px;background:rgba(251,253,249,.97);color:#173D3A;
        box-shadow:0 6px 18px rgba(28,77,70,.20);cursor:pointer;font:700 12px/1.2 'Atkinson Hyperlegible',Verdana,Arial,sans-serif;}}
      #edm-panel-toggle:hover,#edm-panel-toggle:focus {{background:#EAF6F0;}}
      body.edm-panels-hidden #edm-map-left {{transform:translateX(calc(-100% - 24px));opacity:0;pointer-events:none;}}
      body.edm-panels-hidden #edm-map-right {{transform:translateX(calc(100% + 24px));opacity:0;pointer-events:none;}}
      body.edm-panels-hidden #edm-panel-toggle {{display:inline-flex;align-items:center;gap:6px;}}
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
    <button id="edm-panel-toggle" type="button" aria-label="Show the map filters and rankings">
      &#9776; Show filters and rankings
    </button>
    <aside id="edm-map-left" class="edm-map-panel" aria-label="Town and city directory">
      <div class="edm-map-title"><span>{map_title}</span>
      <button class="edm-panel-close" type="button" data-edm-hide-panels aria-label="Hide the map panels">&times;</button></div>
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
      <div class="edm-map-title"><span>Water-company ranking</span>
      <button class="edm-panel-close" type="button" data-edm-hide-panels aria-label="Hide the map panels">&times;</button></div>
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
    var edmPanelsManuallyHidden=false;
    function edmHidePanels(manual){{
      document.body.classList.add('edm-panels-hidden');
      if(manual)edmPanelsManuallyHidden=true;
    }}
    function edmShowPanels(){{
      edmPanelsManuallyHidden=false;
      document.body.classList.remove('edm-panels-hidden');
    }}
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
        '<div class="edm-place-detail" style="margin-top:5px">The 2021–2025 trend is recorded evidence; any 2026 category remains a forecast.</div>';
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
    document.querySelectorAll('[data-edm-hide-panels]').forEach(function(button){{
      button.addEventListener('click',function(){{edmHidePanels(true);}});
    }});
    document.getElementById('edm-panel-toggle').addEventListener('click',edmShowPanels);
    edmMap.on('popupopen',function(){{edmHidePanels(false);}});
    edmMap.on('popupclose',function(){{
      if(!edmPanelsManuallyHidden)window.setTimeout(edmShowPanels,120);
    }});
    window.setTimeout(edmRenderPlaces,0);
    """
    # streamlit-folium does not reliably emit custom code placed in
    # ``root.script``. Add a real script element and wait until Leaflet loads.
    water_map.get_root().html.add_child(
        folium.Element(
            "<script>\n"
            "window.addEventListener('load', function(){\n"
            + script
            + "\n});\n"
            "</script>"
        )
    )


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
        tiles="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="Pastel rivers and places",
        show=True,
        control=True,
        subdomains="abcd",
        max_zoom=20,
    ).add_to(water_map)
    folium.TileLayer(
        tiles="CartoDB positron",
        name="Soft contrast map",
        show=False,
        control=True,
    ).add_to(water_map)
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Detailed street map",
        show=False,
        control=True,
    ).add_to(water_map)

    water_map.get_root().header.add_child(
        folium.Element(
            """
            <style>
              .leaflet-container {
                background: linear-gradient(145deg,#CFEAF1 0%,#E1F2EE 55%,#EFF6E8 100%) !important;
                font-family: "Atkinson Hyperlegible", Verdana, Arial, sans-serif !important;
              }
              .leaflet-tile-pane {
                filter: saturate(1.14) contrast(.97) brightness(1.025);
              }
              .leaflet-control-zoom a,
              .leaflet-control-layers,
              .leaflet-control-scale-line {
                border-color: rgba(39,106,97,.28) !important;
                color: #173D3A !important;
                background: rgba(251,253,249,.94) !important;
              }
              .marker-cluster-small,
              .marker-cluster-medium,
              .marker-cluster-large {
                background-color: rgba(104,175,194,.30) !important;
              }
              .marker-cluster div {
                background: linear-gradient(145deg,#74C6BE,#5BA6C1) !important;
                color: #FFFFFF !important;
                box-shadow: 0 5px 14px rgba(28,83,78,.24);
              }
              .leaflet-popup-content-wrapper,
              .leaflet-popup-tip {
                background: rgba(251,253,249,.98) !important;
              }
            </style>
            """
        )
    )

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
        <div><div style="font-size:1.12rem;font-weight:800;line-height:1.2;">Sewage Overflow<br>Insights</div>
        <div style="font-size:.76rem;color:#5D7772;margin-top:.18rem;">England and Wales · evidence · forecast</div></div>
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
    "Start here",
    "Explore the map",
    "Priority locations",
    "Places and companies",
    "Improvements and changes",
    "2026 predictions",
    "Rainfall and spills",
    "Check one location",
    "About the evidence",
]

st.sidebar.markdown("---")
page_label = st.sidebar.radio(
    "Choose a section",
    PAGES,
    key="sidebar_navigation",
)
st.sidebar.html(
    """
    <div class="edm-access-note">
      <b style="color:#173D3A;">2026 means forecast</b><br>
      It is an estimate, not a confirmed future spill.
    </div>
    """
)

page = page_label


# =============================================================================
# PAGE 1 — OVERVIEW
# =============================================================================

if page == "Start here":
    render_hero()
    section_header("Risk categories", "")
    render_risk_guide()

    section_header(
        "Spill-risk percentages",
        "Compare the recorded evidence with the separately labelled 2026 forecast.",
    )
    st.html(
        """
        <div class="edm-home-chart-note">
          Each percentage is the share of <b>mapped discharge outlets</b> in a risk category.
          Recorded 2021–2025 evidence and predicted 2026 risk are deliberately kept separate.
        </div>
        """
    )

    observed_overview = load_table("observed_locations")
    forecast_overview = load_table("forecast_map_points")
    overview_charts = []
    if (
        not observed_overview.empty
        and "period_risk_category" in observed_overview.columns
    ):
        overview_charts.append(
            (
                observed_overview,
                "period_risk_category",
                "Recorded risk categories for mapped discharge outlets, 2021–2025",
                "outlets classified",
                "home_observed_risk_share",
            )
        )
    if (
        not forecast_overview.empty
        and "predicted_2026_risk" in forecast_overview.columns
    ):
        overview_charts.append(
            (
                forecast_overview,
                "predicted_2026_risk",
                "Predicted risk categories for mapped discharge outlets, 2026",
                "outlets forecast",
                "home_predicted_risk_share",
            )
        )

    if overview_charts:
        for chart_column, chart_details in zip(
            st.columns(len(overview_charts)),
            overview_charts,
        ):
            (
                chart_frame,
                chart_risk,
                chart_title,
                chart_centre_label,
                chart_key,
            ) = chart_details
            with chart_column:
                st.plotly_chart(
                    risk_donut(
                        chart_frame,
                        chart_risk,
                        chart_title,
                        chart_centre_label,
                    ),
                    use_container_width=True,
                    key=chart_key,
                    config={"displayModeBar": False},
                )
    else:
        st.info("The risk-percentage charts will appear when the dashboard data is available.")


# =============================================================================
# PAGE 2 — COMBINED OBSERVED/PREDICTED MAP EXPERIENCE
# =============================================================================

elif page == "Explore the map":
    st.html(
        """
        <style>
          .block-container {
            width:100% !important;
            max-width:none !important;
            padding-left:.55rem !important;
            padding-right:.55rem !important;
          }
        </style>
        """
    )
    section_header(
        "Explore spill locations across England",
        "Start with every mapped outlet, then choose a cluster, town, water company or risk level.",
    )

    layer = st.radio(
        "What would you like to see?",
        ["Recorded 2021–2025 (what happened)", "2026 forecast (what may happen)"],
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
                "<b>Recorded information:</b> every marker is a mapped discharge outlet using the supplied 2021–2025 records.",
                icon="💧",
                background=PALE_MINT,
                edge="#4A9C7D",
            )

        render_risk_guide()
        filtered, place_column = filter_map(map_data, risk_column, prediction)
        risk_counts = filtered[risk_column].value_counts().reindex(RISK_ORDER, fill_value=0)
        if prediction:
            metric_cards(
                [
                    {
                        "label": "2026 forecast outlets shown",
                        "value": value_text(len(filtered)),
                        "note": "Mapped receiving-water outlets · not confirmed spills",
                        "accent": "#B7DDE5",
                    },
                    {
                        "label": "2026 predicted Low outlets",
                        "value": value_text(risk_counts["Low"]),
                        "note": "Forecast receiving-water locations",
                        "accent": "#A8D8D0",
                    },
                    {
                        "label": "2026 predicted Medium outlets",
                        "value": value_text(risk_counts["Medium"]),
                        "note": "Forecast receiving-water locations",
                        "accent": "#F1D39D",
                    },
                    {
                        "label": "2026 predicted High outlets",
                        "value": value_text(risk_counts["High"]),
                        "note": "Forecast receiving-water locations",
                        "accent": "#E9A7A7",
                    },
                ]
            )
        else:
            annual_spills = mapped_annual_spill_totals(filtered, place_column)
            recorded_cards = [
                {
                    "label": "Receiving-water outlets shown",
                    "value": value_text(len(filtered)),
                    "note": "Mapped outlets — this is not a spill count",
                    "accent": "#B7DDE5",
                }
            ]
            year_accents = ["#C6DFEA", "#B9DCCF", "#A8D8D0", "#E8CD6A", "#F1D39D"]
            recorded_cards.extend(
                {
                    "label": f"{year} counted spills",
                    "value": value_text(annual_spills[year]),
                    "note": "Recorded across the receiving-water locations shown",
                    "accent": accent,
                }
                for year, accent in zip(OBSERVED_YEARS, year_accents)
            )
            metric_cards(recorded_cards)

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
                height=1020,
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
        ["Recorded 2021–2025", "Predicted 2026"],
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
            direction_value = row.get(
                "trend_2021_to_2025",
                row.get("trend_2023_to_2025", "Not available"),
            )
            metric_cards(
                [
                    {"label": "Place", "value": place, "note": str(row.get("water_companies", "Company not recorded")), "accent": "#B7DDE5"},
                    {"label": "2021 → 2025 direction", "value": str(direction_value), "note": "Observed counted-spill direction", "accent": "#A8D8D0"},
                    {"label": "Risk history", "value": str(row.get("town_risk_transition", "Not available")), "note": "Highest annual mapped risk", "accent": "#F1D39D"},
                    {"label": "2025 counted spills", "value": value_text(row.get("counted_spills_2025")), "note": "Recorded evidence—not volume", "accent": "#E9A7A7"},
                ]
            )
            trend = pd.DataFrame(
                {
                    "Year": list(OBSERVED_YEARS),
                    "Counted spills": [row.get(f"counted_spills_{year}") for year in OBSERVED_YEARS],
                    "Duration hours": [row.get(f"duration_hours_{year}") for year in OBSERVED_YEARS],
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
# PAGE 5 — IMPROVEMENTS AND RECORDED CHANGES
# =============================================================================

elif page == "Improvements and changes":
    section_header(
        "Where did recorded spills and risk improve?",
        "Compare water companies, then explore every available town, city and receiving-water outlet.",
    )
    banner(
        "A decrease means the recorded 2025 result is lower than the corresponding "
        "2021–2024 average for that company or location. Missing years remain missing; "
        "they are not replaced with artificial values. These figures describe recorded "
        "change and do not prove why it happened.",
        icon="↕",
        background="#EDF7F3",
        edge="#62A887",
    )

    company_change_tab, town_change_tab, water_change_tab = st.tabs(
        [
            "Water-company improvement",
            "Towns and cities",
            "Receiving-water locations",
        ]
    )

    with company_change_tab:
        company_changes = load_table("company_improvement_results")
        if (
            not company_changes.empty
            and (
                "baseline_period" not in company_changes.columns
                or not company_changes["baseline_period"]
                .astype(str)
                .eq("2021-2024")
                .all()
            )
        ):
            # Never display the obsolete 2023–2024 comparison under a
            # 2021–2025 heading. Recalculate from the five-year annual table.
            company_changes = pd.DataFrame()

        # A transparent fallback keeps the page usable during deployment. The
        # common-location Colab export replaces this descriptive annual result.
        if company_changes.empty:
            annual_fallback = load_table("annual_company_trends")
            fallback_rows = []
            required_risk_columns = {
                "water_company_name",
                "reporting_year",
                "low_risk_unique_locations",
                "medium_risk_unique_locations",
                "high_risk_unique_locations",
            }
            if required_risk_columns.issubset(annual_fallback.columns):
                annual_fallback = annual_fallback.copy()
                annual_fallback["reporting_year"] = pd.to_numeric(
                    annual_fallback["reporting_year"], errors="coerce"
                )
                for company_name, company_rows in annual_fallback.groupby(
                    "water_company_name"
                ):
                    yearly_percentages = {}
                    yearly_totals = {}
                    for year in OBSERVED_YEARS:
                        row_match = company_rows.loc[
                            company_rows["reporting_year"].eq(year)
                        ]
                        if row_match.empty:
                            continue
                        row = row_match.iloc[0]
                        low = pd.to_numeric(row.get("low_risk_unique_locations"), errors="coerce")
                        medium = pd.to_numeric(row.get("medium_risk_unique_locations"), errors="coerce")
                        high = pd.to_numeric(row.get("high_risk_unique_locations"), errors="coerce")
                        total = low + medium + high
                        if pd.notna(total) and total > 0:
                            yearly_percentages[year] = (medium + high) / total * 100
                            yearly_totals[year] = total
                    if all(year in yearly_percentages for year in OBSERVED_YEARS):
                        baseline = np.mean(
                            [yearly_percentages[year] for year in BASELINE_YEARS]
                        )
                        fallback_rows.append(
                            {
                                "water_company_name": company_name,
                                "common_locations": yearly_totals[2025],
                                "baseline_medium_high_percent": baseline,
                                "medium_high_percent_2025": yearly_percentages[2025],
                                "risk_improvement_percentage_points": (
                                    baseline - yearly_percentages[2025]
                                ),
                            }
                        )
            company_changes = pd.DataFrame(fallback_rows)
            if not company_changes.empty:
                st.caption(
                    "Temporary descriptive view. Re-run the Colab deployment to use the exact common-location comparison."
                )

        required_company_columns = {
            "water_company_name",
            "baseline_medium_high_percent",
            "medium_high_percent_2025",
            "risk_improvement_percentage_points",
        }
        if company_changes.empty or not required_company_columns.issubset(
            company_changes.columns
        ):
            st.info(
                "Run the company-improvement Colab cell and the updated dashboard installer to publish this ranking."
            )
        else:
            company_changes = company_changes.copy()
            for column in [
                "baseline_medium_high_percent",
                "medium_high_percent_2025",
                "risk_improvement_percentage_points",
                "common_locations",
            ]:
                if column in company_changes.columns:
                    company_changes[column] = pd.to_numeric(
                        company_changes[column], errors="coerce"
                    )
            company_changes = company_changes.dropna(
                subset=["risk_improvement_percentage_points"]
            ).sort_values("risk_improvement_percentage_points")

            best_company = company_changes.iloc[-1]
            smallest_company = company_changes.iloc[0]
            metric_cards(
                [
                    {
                        "label": "Greatest improvement",
                        "value": str(best_company["water_company_name"]),
                        "note": f"{np.ceil(best_company['risk_improvement_percentage_points']):.0f} percentage-point reduction",
                        "accent": "#62A887",
                    },
                    {
                        "label": "Smallest improvement",
                        "value": str(smallest_company["water_company_name"]),
                        "note": f"{np.ceil(smallest_company['risk_improvement_percentage_points']):.0f} percentage-point change",
                        "accent": "#EBA35B",
                    },
                    {
                        "label": "Companies compared",
                        "value": f"{len(company_changes):,}",
                        "note": "Ranked using Medium/High-risk share",
                        "accent": "#E8CD6A",
                    },
                ]
            )

            maximum_value = company_changes[
                "risk_improvement_percentage_points"
            ].max()
            minimum_value = company_changes[
                "risk_improvement_percentage_points"
            ].min()

            def company_bar_colour(value):
                if value < 0:
                    return "#D97A76"
                if np.isclose(value, maximum_value):
                    return "#62A887"
                if np.isclose(value, minimum_value):
                    return "#EBA35B"
                return "#E8CD6A"

            company_changes["bar_colour"] = company_changes[
                "risk_improvement_percentage_points"
            ].map(company_bar_colour)
            company_changes["display_change"] = company_changes[
                "risk_improvement_percentage_points"
            ].map(lambda value: f"{int(np.ceil(value)):+d} points")
            company_changes["popup"] = company_changes.apply(
                lambda row: (
                    f"<b>{row['water_company_name']}</b><br>"
                    "Medium/High-risk locations before 2025: "
                    f"{int(np.ceil(row['baseline_medium_high_percent']))}%<br>"
                    "Medium/High-risk locations in 2025: "
                    f"{int(np.ceil(row['medium_high_percent_2025']))}%<br>"
                    "Improvement: "
                    f"{int(np.ceil(row['risk_improvement_percentage_points']))} percentage points"
                ),
                axis=1,
            )

            company_figure = go.Figure(
                go.Bar(
                    x=company_changes["risk_improvement_percentage_points"],
                    y=company_changes["water_company_name"],
                    orientation="h",
                    marker=dict(
                        color=company_changes["bar_colour"],
                        line=dict(color="#FFFFFF", width=1.5),
                    ),
                    text=company_changes["display_change"],
                    textposition="outside",
                    customdata=company_changes["popup"],
                    hovertemplate="%{customdata}<extra></extra>",
                )
            )
            company_figure.update_layout(
                title="Reduction in Medium/High-risk locations by 2025",
                xaxis_title="Improvement from the 2021–2024 average (percentage points)",
                yaxis_title="",
                height=max(560, 55 * len(company_changes)),
                margin=dict(l=175, r=100, t=85, b=85),
                showlegend=False,
            )
            company_figure.add_vline(x=0, line_color="#5D7772", line_width=1)
            st.plotly_chart(
                plot_style(company_figure, max(560, 55 * len(company_changes))),
                use_container_width=True,
            )
            st.caption(
                "Green = greatest improvement · Yellow = improvement · Orange = smallest improvement · Red = increase."
            )

            company_table = company_changes[
                [
                    "water_company_name",
                    "baseline_medium_high_percent",
                    "medium_high_percent_2025",
                    "risk_improvement_percentage_points",
                ]
            ].sort_values("risk_improvement_percentage_points", ascending=False)
            company_table.columns = [
                "Water company",
                "2021–2024 Medium/High average (%)",
                "2025 Medium/High (%)",
                "Improvement (percentage points)",
            ]
            st.dataframe(
                company_table.round(0), use_container_width=True, hide_index=True
            )

    with town_change_tab:
        town_changes = load_table("town_trends")
        required_town_columns = {"official_place_name"} | {
            f"counted_spills_{year}" for year in OBSERVED_YEARS
        }
        if town_changes.empty or not required_town_columns.issubset(
            town_changes.columns
        ):
            st.info("The town/city change export is unavailable.")
        else:
            town_changes = town_changes.copy()
            for column in [f"counted_spills_{year}" for year in OBSERVED_YEARS]:
                town_changes[column] = pd.to_numeric(
                    town_changes[column], errors="coerce"
                )
            town_changes["average_before_2025"] = town_changes[
                [f"counted_spills_{year}" for year in BASELINE_YEARS]
            ].mean(axis=1)
            town_changes["change_to_2025"] = (
                town_changes["counted_spills_2025"]
                - town_changes["average_before_2025"]
            )
            town_changes["change_percent"] = np.where(
                town_changes["average_before_2025"].gt(0),
                town_changes["change_to_2025"]
                / town_changes["average_before_2025"]
                * 100,
                np.nan,
            )
            town_changes["Direction"] = np.select(
                [
                    town_changes["change_to_2025"].lt(0),
                    town_changes["change_to_2025"].gt(0),
                ],
                ["Decreased", "Increased"],
                default="Stayed the same",
            )
            direction_filter = st.segmented_control(
                "Show towns and cities where counted spills:",
                ["All", "Decreased", "Increased", "Stayed the same"],
                default="All",
                key="town_change_direction",
            )
            town_options_frame = town_changes
            if direction_filter and direction_filter != "All":
                town_options_frame = town_changes.loc[
                    town_changes["Direction"].eq(direction_filter)
                ]
            town_options = available_values(
                town_options_frame, "official_place_name"
            )
            if not town_options:
                st.info("No towns or cities match that change category.")
            else:
                selected_town = st.selectbox(
                    "Choose any town or city",
                    town_options,
                    key="town_change_place",
                )
                town_row = town_options_frame.loc[
                    town_options_frame["official_place_name"]
                    .astype(str)
                    .eq(selected_town)
                ].iloc[0]
                town_direction = str(town_row["Direction"])
                town_colour = {
                    "Decreased": "#62A887",
                    "Increased": "#D97A76",
                    "Stayed the same": "#EBA35B",
                }[town_direction]
                if pd.notna(town_row["change_percent"]):
                    change_size = int(np.ceil(abs(town_row["change_percent"])))
                    town_change_text = (
                        f"{change_size}% fewer"
                        if town_direction == "Decreased"
                        else f"{change_size}% more"
                        if town_direction == "Increased"
                        else "No change"
                    )
                else:
                    town_change_text = town_direction
                metric_cards(
                    [
                        {
                            "label": "Town or city",
                            "value": selected_town,
                            "note": str(town_row.get("water_companies", "Company not recorded")),
                            "accent": "#B7DDE5",
                        },
                        {
                            "label": "Recorded direction",
                            "value": town_direction,
                            "note": town_change_text,
                            "accent": town_colour,
                        },
                        {
                            "label": "2025 counted spills",
                            "value": value_text(town_row["counted_spills_2025"]),
                            "note": "Counted events—not spill volume",
                            "accent": "#E8CD6A",
                        },
                    ]
                )
                town_figure_data = pd.DataFrame(
                    {
                        "Year": list(OBSERVED_YEARS),
                        "Counted spills": [
                            town_row[f"counted_spills_{year}"]
                            for year in OBSERVED_YEARS
                        ],
                    }
                )
                town_figure = go.Figure(
                    go.Scatter(
                        x=town_figure_data["Year"],
                        y=town_figure_data["Counted spills"],
                        mode="lines+markers+text",
                        text=town_figure_data["Counted spills"].map(
                            lambda value: value_text(value)
                        ),
                        textposition="top center",
                        line=dict(color=town_colour, width=6, shape="spline"),
                        marker=dict(
                            size=16,
                            color=["#C9DCE5", "#BED9DB", "#B8D8D1", "#E8CD6A", town_colour],
                        ),
                        fill="tozeroy",
                        fillcolor=f"{town_colour}22",
                        hovertemplate="%{x}: %{y:,.0f} counted spills<extra></extra>",
                    )
                )
                town_figure.update_layout(
                    title=f"Recorded counted-spill change · {selected_town}",
                    xaxis_title="Year",
                    yaxis_title="Counted spills",
                    height=470,
                    showlegend=False,
                )
                town_figure.update_xaxes(dtick=1)
                st.plotly_chart(
                    plot_style(town_figure, 470), use_container_width=True
                )

            town_change_table = town_changes[
                [
                    "official_place_name",
                    "water_companies",
                    *[f"counted_spills_{year}" for year in OBSERVED_YEARS],
                    "Direction",
                    "change_percent",
                ]
            ].copy()
            town_change_table.columns = [
                "Town or city",
                "Water company",
                *[str(year) for year in OBSERVED_YEARS],
                "Change",
                "Change (%)",
            ]
            town_change_table["Change (%)"] = town_change_table[
                "Change (%)"
            ].round(0)
            with st.expander("View every town and city"):
                st.dataframe(
                    town_change_table,
                    use_container_width=True,
                    hide_index=True,
                )

    with water_change_tab:
        water_changes = load_table("receiving_water_changes")
        required_water_columns = {
            "location_id",
            "water_company_name",
            "site_name",
            "receiving_water",
            "official_place_name",
            "spill_direction",
        } | {f"counted_spills_{year}" for year in OBSERVED_YEARS}
        if water_changes.empty or not required_water_columns.issubset(
            water_changes.columns
        ):
            st.info(
                "Run the updated company-improvement cell and dashboard installer to publish receiving-water changes."
            )
        else:
            water_changes = water_changes.copy()
            for column in [
                *[f"counted_spills_{year}" for year in OBSERVED_YEARS],
                "spill_change_percent",
            ]:
                if column in water_changes.columns:
                    water_changes[column] = pd.to_numeric(
                        water_changes[column], errors="coerce"
                    )
            water_changes["Display location"] = water_changes.apply(
                lambda row: (
                    f"{safe_text(row.get('receiving_water'), 'Receiving water not recorded')} — "
                    f"{safe_text(row.get('site_name'), 'Site not recorded')} — "
                    f"{safe_text(row.get('official_place_name'), 'Place not recorded')}"
                ),
                axis=1,
            )
            water_direction = st.segmented_control(
                "Show receiving-water locations where counted spills:",
                ["All", "Decreased", "Increased", "No change"],
                default="All",
                key="water_change_direction",
            )
            water_options_frame = water_changes
            if water_direction and water_direction != "All":
                water_options_frame = water_changes.loc[
                    water_changes["spill_direction"].astype(str).eq(water_direction)
                ]
            water_options = available_values(
                water_options_frame, "Display location"
            )
            if not water_options:
                st.info("No receiving-water locations match that change category.")
            else:
                selected_water = st.selectbox(
                    "Choose a receiving-water location",
                    water_options,
                    key="receiving_water_change_location",
                )
                water_row = water_options_frame.loc[
                    water_options_frame["Display location"]
                    .astype(str)
                    .eq(selected_water)
                ].iloc[0]
                water_result = str(water_row["spill_direction"])
                water_colour = {
                    "Decreased": "#62A887",
                    "Increased": "#D97A76",
                    "No change": "#EBA35B",
                }.get(water_result, "#68AFC2")
                water_change_value = pd.to_numeric(
                    water_row.get("spill_change_percent"), errors="coerce"
                )
                if pd.notna(water_change_value):
                    water_change_note = (
                        f"{int(np.ceil(abs(water_change_value)))}% "
                        + ("fewer" if water_change_value < 0 else "more")
                        if not np.isclose(water_change_value, 0)
                        else "No change"
                    )
                else:
                    water_change_note = "Percentage unavailable"
                metric_cards(
                    [
                        {
                            "label": "Receiving water",
                            "value": str(water_row["receiving_water"]),
                            "note": str(water_row["site_name"]),
                            "accent": "#B7DDE5",
                        },
                        {
                            "label": "Nearest town or city",
                            "value": str(water_row["official_place_name"]),
                            "note": str(water_row["water_company_name"]),
                            "accent": "#C9DDE8",
                        },
                        {
                            "label": "Recorded direction",
                            "value": water_result,
                            "note": water_change_note,
                            "accent": water_colour,
                        },
                    ]
                )
                water_figure_data = pd.DataFrame(
                    {
                        "Year": list(OBSERVED_YEARS),
                        "Counted spills": [
                            water_row[f"counted_spills_{year}"]
                            for year in OBSERVED_YEARS
                        ],
                    }
                )
                water_figure = go.Figure(
                    go.Bar(
                        x=water_figure_data["Year"],
                        y=water_figure_data["Counted spills"],
                        marker_color=["#C9DCE5", "#BED9DB", "#B8D8D1", "#E8CD6A", water_colour],
                        text=water_figure_data["Counted spills"].map(
                            lambda value: value_text(value)
                        ),
                        textposition="outside",
                        hovertemplate="%{x}: %{y:,.0f} counted spills<extra></extra>",
                    )
                )
                water_figure.update_layout(
                    title=f"Recorded counted spills · {water_row['receiving_water']}",
                    xaxis_title="Year",
                    yaxis_title="Counted spills",
                    height=470,
                    showlegend=False,
                )
                water_figure.update_xaxes(dtick=1)
                st.plotly_chart(
                    plot_style(water_figure, 470), use_container_width=True
                )

            water_table_columns = [
                "receiving_water",
                "site_name",
                "official_place_name",
                "water_company_name",
                *[f"counted_spills_{year}" for year in OBSERVED_YEARS],
                "spill_direction",
            ]
            with st.expander("View every receiving-water location"):
                st.dataframe(
                    water_changes[water_table_columns].rename(
                        columns={
                            "receiving_water": "Receiving water",
                            "site_name": "Outlet/site",
                            "official_place_name": "Nearest town or city",
                            "water_company_name": "Water company",
                            **{
                                f"counted_spills_{year}": str(year)
                                for year in OBSERVED_YEARS
                            },
                            "spill_direction": "Change",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# =============================================================================
# PAGE 6 — 2026 PREDICTIONS AND AFFECTED LOCATIONS
# =============================================================================

elif page == "2026 predictions":
    st.html(
        """
        <style>
          .block-container {
            width:100% !important;
            max-width:none !important;
            padding-left:.55rem !important;
            padding-right:.55rem !important;
          }
        </style>
        """
    )
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
                    height=980,
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
# PAGE 5 — OFFICIAL 2021–2025 RAINFALL MEASUREMENTS
# =============================================================================

elif page == "Rainfall and spills":
    st.html(
        """
        <section style="position:relative;overflow:hidden;margin:.15rem 0 1.1rem;
          padding:1.7rem 2rem;border:1px solid rgba(77,151,152,.22);border-radius:28px;
          background:linear-gradient(125deg,#E5F6F1 0%,#DFF2F7 55%,#EEEAF8 100%);
          box-shadow:0 20px 48px rgba(43,104,99,.13);">
          <div style="position:relative;z-index:2;max-width:850px">
            <div style="display:inline-block;padding:.35rem .8rem;border-radius:999px;
              background:rgba(255,255,255,.76);color:#326E70;font-weight:800;
              letter-spacing:.08em;">
              🌧️ OFFICIAL MET OFFICE RAINFALL EVIDENCE
            </div>
            <h1 style="margin:.75rem 0 .35rem;color:#173D3A;
              font-size:clamp(2rem,4vw,3.5rem);">
              Rainfall and recorded spills
            </h1>
            <p style="max-width:780px;margin:0;color:#4F706C;font-size:1.08rem;">
              Explore official regional rainfall measurements for 2021–2025 and use
              them as supporting weather evidence when interpreting recorded EDM spills.
            </p>
          </div>
          <svg aria-hidden="true" viewBox="0 0 520 180" style="position:absolute;
            right:-25px;bottom:-24px;width:min(38vw,520px);opacity:.73">
            <path d="M0 118 C75 72 135 158 212 112 C290 65 350 154 520 88
              L520 180 L0 180Z" fill="#91CDD4"/>
            <path d="M0 139 C85 96 145 173 230 132 C319 89 405 165 520 121"
              fill="none" stroke="#F8FFFF" stroke-width="12" stroke-linecap="round"/>
            <g fill="#6BAED6" opacity=".85">
              <path d="M374 42 C354 67 352 75 352 86 C352 101 362 111 376 111
                C390 111 400 101 400 86 C400 75 396 65 374 42Z"/>
              <path d="M432 24 C412 49 410 57 410 68 C410 83 420 93 434 93
                C448 93 458 83 458 68 C458 57 454 47 432 24Z"/>
            </g>
          </svg>
        </section>
        """
    )

    rainfall_annual = load_table("rainfall_annual_2021_2025")
    rainfall_monthly = load_table("rainfall_monthly_2021_2025")
    rainfall_daily = load_table("rainfall_daily_2021_2025")

    if rainfall_annual.empty or rainfall_monthly.empty:
        st.error(
            "The official rainfall dashboard files are unavailable. Run Rainfall "
            "Cells 1–2 and publish the three rainfall CSV files before using this page."
        )
    else:
        rainfall_annual = rainfall_annual.copy()
        rainfall_monthly = rainfall_monthly.copy()

        rainfall_annual["year"] = pd.to_numeric(
            rainfall_annual["year"], errors="coerce"
        )
        rainfall_monthly["year"] = pd.to_numeric(
            rainfall_monthly["year"], errors="coerce"
        )
        rainfall_monthly["month"] = pd.to_numeric(
            rainfall_monthly["month"], errors="coerce"
        )

        rainfall_regions = sorted(
            rainfall_annual["region"].dropna().astype(str).unique().tolist()
        )
        default_region = (
            rainfall_regions.index("England and Wales")
            if "England and Wales" in rainfall_regions
            else 0
        )
        selected_rainfall_region = st.selectbox(
            "Official rainfall region",
            rainfall_regions,
            index=default_region,
            key="full_dashboard_rainfall_region",
        )

        annual_view = rainfall_annual.loc[
            rainfall_annual["region"].astype(str).eq(selected_rainfall_region)
        ].sort_values("year")
        monthly_view = rainfall_monthly.loc[
            rainfall_monthly["region"].astype(str).eq(selected_rainfall_region)
        ].copy()

        latest = annual_view.iloc[-1]
        latest_year = int(latest["year"])
        metric_cards(
            [
                {
                    "label": f"{latest_year} rainfall",
                    "value": f"{latest['annual_rainfall_mm']:,.0f} mm",
                    "note": selected_rainfall_region,
                    "accent": "#68AFC2",
                },
                {
                    "label": "Wet days",
                    "value": value_text(latest["wet_days"]),
                    "note": "Days with at least 1 mm",
                    "accent": "#79BEAB",
                },
                {
                    "label": "Heavy-rain days",
                    "value": value_text(latest["heavy_rain_days"]),
                    "note": "Days with at least 10 mm",
                    "accent": "#8FBCE2",
                },
                {
                    "label": "Wettest day",
                    "value": f"{latest['maximum_daily_rainfall_mm']:,.1f} mm",
                    "note": "Maximum recorded daily total",
                    "accent": "#C8A8DD",
                },
            ]
        )

        section_header(
            "Annual rainfall pattern",
            "Compare official annual rainfall totals across the five observed years.",
        )
        annual_rainfall_figure = px.bar(
            annual_view,
            x="year",
            y="annual_rainfall_mm",
            color="annual_rainfall_mm",
            text_auto=".0f",
            color_continuous_scale=["#E8F5F2", "#BDE0FE", "#6C8EBF"],
            labels={"year": "Year", "annual_rainfall_mm": "Rainfall (mm)"},
            title=f"Annual rainfall — {selected_rainfall_region}",
        )
        annual_rainfall_figure.update_layout(coloraxis_showscale=False)
        st.plotly_chart(
            plot_style(annual_rainfall_figure, 520),
            use_container_width=True,
            key="full_dashboard_annual_rainfall",
            config={"displayModeBar": False},
        )

        monthly_view["period"] = pd.to_datetime(
            monthly_view["year"].astype("Int64").astype(str)
            + "-"
            + monthly_view["month"].astype("Int64").astype(str)
            + "-01",
            errors="coerce",
        )
        monthly_view = monthly_view.dropna(
            subset=["period", "monthly_rainfall_mm"]
        ).sort_values("period")

        section_header(
            "Monthly rainfall pattern",
            "View how rainfall varied within and between the observed years.",
        )
        monthly_rainfall_figure = px.area(
            monthly_view,
            x="period",
            y="monthly_rainfall_mm",
            color_discrete_sequence=["#68AFC2"],
            labels={"period": "Month", "monthly_rainfall_mm": "Rainfall (mm)"},
            title=f"Monthly rainfall — {selected_rainfall_region}",
        )
        st.plotly_chart(
            plot_style(monthly_rainfall_figure, 540),
            use_container_width=True,
            key="full_dashboard_monthly_rainfall",
            config={"displayModeBar": False},
        )

        section_header(
            "Heavy-rain exposure across regions",
            "Compare the number of days recording at least 10 mm of rainfall.",
        )
        regional_heavy_days = rainfall_annual.pivot(
            index="region",
            columns="year",
            values="heavy_rain_days",
        )
        heavy_rain_figure = px.imshow(
            regional_heavy_days,
            text_auto=".0f",
            aspect="auto",
            color_continuous_scale=["#FFF7E6", "#BDE0FE", "#6C8EBF"],
            labels={"x": "Year", "y": "Rainfall region", "color": "Days"},
            title="Heavy-rain days by official rainfall region",
        )
        st.plotly_chart(
            plot_style(heavy_rain_figure, 560),
            use_container_width=True,
            key="full_dashboard_heavy_rain",
            config={"displayModeBar": False},
        )

        banner(
            "<b>Interpretation:</b> wetter conditions can increase hydraulic pressure "
            "on combined sewer systems. These regional measurements provide weather "
            "context, but they do not prove why an individual outlet discharged and "
            "must not be treated as evidence of an infrastructure fault.",
            icon="i",
            background="#EDF7F3",
            edge="#62A887",
        )
        st.markdown(
            "Source: [Met Office HadUKP daily precipitation series]"
            "(https://www.metoffice.gov.uk/hadobs/hadukp/data/download.html)"
        )

        with st.expander("Download the official rainfall tables"):
            download_table(
                annual_view,
                "selected_region_annual_rainfall_2021_2025.csv",
            )
            download_table(
                monthly_view.drop(columns=["period"], errors="ignore"),
                "selected_region_monthly_rainfall_2021_2025.csv",
            )
            if not rainfall_daily.empty:
                daily_view = rainfall_daily.loc[
                    rainfall_daily["region"].astype(str).eq(selected_rainfall_region)
                ]
                download_table(
                    daily_view,
                    "selected_region_daily_rainfall_2021_2025.csv",
                )

    # Stop here so the removed legacy water-quality implementation below can
    # never execute. All other dashboard pages remain unchanged.
    st.stop()
    st.html(
        """
        <section style="position:relative;overflow:hidden;margin:.15rem 0 1.1rem;
          padding:1.7rem 2rem;border:1px solid rgba(77,151,152,.22);border-radius:28px;
          background:linear-gradient(125deg,#E5F6F1 0%,#DFF2F7 55%,#EEEAF8 100%);
          box-shadow:0 20px 48px rgba(43,104,99,.13);">
          <div style="position:relative;z-index:2;max-width:850px">
            <div style="display:inline-block;padding:.35rem .8rem;border-radius:999px;
              background:rgba(255,255,255,.76);color:#326E70;font-weight:800;letter-spacing:.08em;">
              💧 OFFICIAL 2025 MONITORING EVIDENCE
            </div>
            <h1 style="margin:.75rem 0 .35rem;color:#173D3A;font-size:clamp(2rem,4vw,3.5rem);">
              Water-quality measurements near selected outlets
            </h1>
            <p style="max-width:760px;margin:0;color:#4F706C;font-size:1.08rem;">
              Explore what nearby Environment Agency stations measured, when it was measured,
              and which mapped site the monitoring evidence was geographically linked to.
            </p>
          </div>
          <svg aria-hidden="true" viewBox="0 0 520 180" style="position:absolute;right:-25px;
            bottom:-24px;width:min(38vw,520px);opacity:.73">
            <path d="M0 118 C75 72 135 158 212 112 C290 65 350 154 520 88 L520 180 L0 180Z"
              fill="#91CDD4"/>
            <path d="M0 139 C85 96 145 173 230 132 C319 89 405 165 520 121"
              fill="none" stroke="#F8FFFF" stroke-width="12" stroke-linecap="round"/>
            <circle cx="355" cy="52" r="28" fill="#F9E7A8"/>
            <g fill="#FFFFFF" opacity=".9"><circle cx="410" cy="55" r="23"/>
              <circle cx="440" cy="42" r="31"/><circle cx="474" cy="58" r="22"/>
              <rect x="400" y="55" width="92" height="25" rx="13"/></g>
          </svg>
        </section>
        """
    )

    quality = load_table("water_quality_records")
    coverage = load_table("water_quality_coverage")
    combined_quality = load_table("water_quality_combined_screening")
    quality_profiles = load_table("water_quality_indicator_profiles")

    if not combined_quality.empty:
        combined_quality = combined_quality.copy()
        combined_quality["combined_relative_concern_score"] = pd.to_numeric(
            combined_quality["combined_relative_concern_score"],
            errors="coerce",
        )
        combined_quality["scored_indicators"] = pd.to_numeric(
            combined_quality["scored_indicators"],
            errors="coerce",
        )

        section_header(
            "Combined 2025 water-quality screening",
            "Connect the recorded indicators for each linked outlet and monitoring station, then identify places for investigation.",
        )
        banner(
            "<b>Relative screening—not a pollution verdict:</b> the score compares linked 2025 "
            "stations with one another. Higher ammonia, phosphate and BOD, and lower dissolved "
            "oxygen, raise the score. pH and temperature remain visible as context but are not "
            "forced into the score.",
            icon="i",
            background="#EDF7F3",
            edge="#62A887",
        )

        scored_quality = combined_quality.loc[
            combined_quality["scored_indicators"].ge(2)
            & combined_quality["combined_relative_concern_score"].notna()
        ].copy()
        priority_count = int(
            combined_quality["screening_band"]
            .astype(str)
            .eq("Higher relative concern")
            .sum()
        )
        metric_cards(
            [
                {
                    "label": "Linked outlet/station pairs",
                    "value": value_text(len(combined_quality)),
                    "note": "With 2025 monitoring evidence",
                    "accent": "#B7DDE5",
                },
                {
                    "label": "Comparable combined profiles",
                    "value": value_text(len(scored_quality)),
                    "note": "At least two directional indicators",
                    "accent": "#A8D8D0",
                },
                {
                    "label": "Higher relative concern",
                    "value": value_text(priority_count),
                    "note": "Priority for investigation—not confirmed harm",
                    "accent": "#D97A76",
                },
                {
                    "label": "Water companies represented",
                    "value": value_text(
                        combined_quality["water_company_name"].nunique()
                        if "water_company_name" in combined_quality.columns
                        else 0
                    ),
                    "note": "Geographically linked evidence",
                    "accent": "#CDBDDE",
                },
            ]
        )

        if scored_quality.empty:
            st.info(
                "No outlet/station pair has at least two comparable directional indicators. "
                "The individual measurements remain available below."
            )
        else:
            company_choices = ["All water companies"] + available_values(
                scored_quality,
                "water_company_name",
            )
            combined_company = st.selectbox(
                "Filter the combined screening by water company",
                company_choices,
                key="combined_quality_company",
            )
            combined_view = scored_quality.copy()
            if combined_company != "All water companies":
                combined_view = combined_view.loc[
                    combined_view["water_company_name"]
                    .astype(str)
                    .eq(combined_company)
                ]

            combined_view["display_location"] = combined_view.apply(
                lambda row: (
                    f"{safe_text(row.get('site_name'), 'Outlet not recorded')} — "
                    f"{safe_text(row.get('monitoring_station'), 'Station not recorded')}"
                ),
                axis=1,
            )
            combined_view["screening_colour"] = combined_view[
                "screening_band"
            ].map(
                {
                    "Higher relative concern": "#D97A76",
                    "Closer review": "#E8B867",
                    "Lower relative concern": "#62A887",
                }
            ).fillna("#9AB8C3")

            ranking_view = combined_view.sort_values(
                "combined_relative_concern_score",
                ascending=True,
            ).tail(20)
            combined_figure = go.Figure(
                go.Bar(
                    x=ranking_view["combined_relative_concern_score"],
                    y=ranking_view["display_location"],
                    orientation="h",
                    marker=dict(
                        color=ranking_view["screening_colour"],
                        line=dict(color="#FFFFFF", width=1.2),
                    ),
                    text=ranking_view["combined_relative_concern_score"].map(
                        lambda value: f"{value:.0f}/100"
                    ),
                    textposition="outside",
                    customdata=ranking_view[
                        [
                            "water_company_name",
                            "screening_band",
                            "scored_indicators",
                            "indicators_measured",
                        ]
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>Company: %{customdata[0]}<br>"
                        "Relative screening: %{x:.0f}/100<br>"
                        "Band: %{customdata[1]}<br>"
                        "Indicators scored: %{customdata[2]:.0f}<br>"
                        "Indicators measured: %{customdata[3]:.0f}<extra></extra>"
                    ),
                )
            )
            combined_figure.update_layout(
                title="Linked locations ranked for closer investigation",
                xaxis_title="Relative-concern screening score (0–100)",
                yaxis_title="Linked outlet and monitoring station",
                height=max(580, 38 * len(ranking_view) + 180),
                margin=dict(l=250, r=90, t=80, b=75),
                showlegend=False,
            )
            combined_figure.update_xaxes(range=[0, 105])
            st.plotly_chart(
                plot_style(
                    combined_figure,
                    max(580, 38 * len(ranking_view) + 180),
                ),
                use_container_width=True,
                key="combined_water_quality_ranking",
                config={"displayModeBar": False},
            )

            location_options = available_values(
                combined_view,
                "display_location",
            )
            selected_combined_location = st.selectbox(
                "Choose a linked location to see all of its 2025 indicators",
                location_options,
                key="combined_quality_location",
            )
            selected_combined = combined_view.loc[
                combined_view["display_location"]
                .astype(str)
                .eq(selected_combined_location)
            ].iloc[0]
            metric_cards(
                [
                    {
                        "label": "Water company",
                        "value": str(selected_combined["water_company_name"]),
                        "note": str(selected_combined.get("site_name", "Linked outlet")),
                        "accent": "#B7DDE5",
                    },
                    {
                        "label": "Linked spill risk",
                        "value": str(selected_combined.get("linked_spill_risk", "Not available")),
                        "note": "EDM evidence—not water-quality status",
                        "accent": "#E8CD6A",
                    },
                    {
                        "label": "Combined relative score",
                        "value": f"{selected_combined['combined_relative_concern_score']:.0f}/100",
                        "note": str(selected_combined["screening_band"]),
                        "accent": str(selected_combined["screening_colour"]),
                    },
                    {
                        "label": "Indicators measured",
                        "value": value_text(selected_combined["indicators_measured"]),
                        "note": "All retained in the profile",
                        "accent": "#CDBDDE",
                    },
                ]
            )

            if not quality_profiles.empty:
                location_profiles = quality_profiles.loc[
                    quality_profiles["location_id"]
                    .astype(str)
                    .eq(str(selected_combined["location_id"]))
                    & quality_profiles["monitoring_station"]
                    .astype(str)
                    .eq(str(selected_combined["monitoring_station"]))
                ].copy()
                for column in [
                    "median_2025_result",
                    "relative_concern_percentile",
                    "exact_measurements",
                ]:
                    location_profiles[column] = pd.to_numeric(
                        location_profiles[column],
                        errors="coerce",
                    )

                scored_profiles = location_profiles.loc[
                    location_profiles["relative_concern_percentile"].notna()
                ].sort_values("relative_concern_percentile")
                if not scored_profiles.empty:
                    profile_figure = go.Figure(
                        go.Bar(
                            x=scored_profiles["relative_concern_percentile"],
                            y=scored_profiles["water_quality_indicator"],
                            orientation="h",
                            marker=dict(
                                color=scored_profiles["relative_concern_percentile"],
                                colorscale=[
                                    [0, "#8FC9B2"],
                                    [.5, "#E8CD6A"],
                                    [1, "#D97A76"],
                                ],
                                cmin=0,
                                cmax=100,
                                showscale=False,
                            ),
                            text=scored_profiles["relative_concern_percentile"].map(
                                lambda value: f"{value:.0f}/100"
                            ),
                            textposition="outside",
                            customdata=scored_profiles[
                                [
                                    "median_2025_result",
                                    "reported_unit",
                                    "exact_measurements",
                                ]
                            ],
                            hovertemplate=(
                                "<b>%{y}</b><br>Relative concern: %{x:.0f}/100<br>"
                                "2025 median: %{customdata[0]:.3f} %{customdata[1]}<br>"
                                "Measurements: %{customdata[2]:.0f}<extra></extra>"
                            ),
                        )
                    )
                    profile_figure.update_layout(
                        title="How each directional indicator contributes",
                        xaxis_title="Relative concern within the linked 2025 sample",
                        yaxis_title="",
                        height=max(440, 55 * len(scored_profiles) + 190),
                        margin=dict(l=220, r=80, t=75, b=70),
                        showlegend=False,
                    )
                    profile_figure.update_xaxes(range=[0, 105])
                    st.plotly_chart(
                        plot_style(
                            profile_figure,
                            max(440, 55 * len(scored_profiles) + 190),
                        ),
                        use_container_width=True,
                        key="combined_quality_profile",
                        config={"displayModeBar": False},
                    )

                profile_table = location_profiles[
                    [
                        "water_quality_indicator",
                        "median_2025_result",
                        "reported_unit",
                        "exact_measurements",
                        "screening_direction",
                        "relative_concern_percentile",
                    ]
                ].rename(
                    columns={
                        "water_quality_indicator": "2025 indicator",
                        "median_2025_result": "Station median",
                        "reported_unit": "Unit",
                        "exact_measurements": "Measurements",
                        "screening_direction": "How it is used",
                        "relative_concern_percentile": "Relative concern (0–100)",
                    }
                )
                st.dataframe(
                    profile_table.round(2),
                    use_container_width=True,
                    hide_index=True,
                )

        st.caption(
            "Use this screening to decide where more monitoring or investigation may be useful. "
            "It is not an official ecological-status classification, a legal breach assessment, "
            "or evidence that a nearby outlet or water company caused a measurement."
        )

    if quality.empty:
        st.info("The public 2025 water-quality measurements are not available yet.")
    else:
        quality = quality.copy()
        date_column = first_existing(
            quality,
            ["measurement_datetime", "measurement_date", "sample_date"],
        )
        if date_column:
            quality["_measurement_datetime"] = pd.to_datetime(
                quality[date_column],
                errors="coerce",
            )
            records_2025 = quality.loc[
                quality["_measurement_datetime"].dt.year.eq(2025)
            ].copy()
            if not records_2025.empty:
                quality = records_2025
        else:
            quality["_measurement_datetime"] = pd.NaT
            if "measurement_year" in quality.columns:
                year_values = pd.to_numeric(
                    quality["measurement_year"],
                    errors="coerce",
                )
                records_2025 = quality.loc[year_values.eq(2025)].copy()
                if not records_2025.empty:
                    quality = records_2025

        company_column = first_existing(quality, ["company", "water_company_name"])
        site_column = first_existing(
            quality,
            ["site_name", "source_site_name_ea_consents_database"],
        )
        station_column = first_existing(
            quality,
            ["sampling_point_name", "sampling_point_id"],
        )
        parameter_column = first_existing(
            quality,
            ["project_parameter_name", "determinand_name", "parameter"],
        )
        unit_column = first_existing(quality, ["reported_unit", "unit"])
        reported_result_column = first_existing(
            quality,
            ["result_as_reported", "result_numeric", "exact_numeric_result"],
        )
        linked_risk_column = first_existing(
            quality,
            ["risk_category", "period_risk_category", "observed_2025_risk"],
        )

        if not parameter_column or not reported_result_column:
            st.warning("The water-quality export is missing its parameter or result column.")
        else:
            first_filters = st.columns(3)
            with first_filters[0]:
                company_options = (
                    ["All water companies"] + available_values(quality, company_column)
                    if company_column
                    else ["All water companies"]
                )
                selected_company = st.selectbox(
                    "Water company",
                    company_options,
                    key="quality_page_company",
                )

            company_filtered = quality.copy()
            if company_column and selected_company != "All water companies":
                company_filtered = company_filtered.loc[
                    company_filtered[company_column].astype(str).eq(selected_company)
                ]

            with first_filters[1]:
                site_options = (
                    ["All linked outlets"] + available_values(company_filtered, site_column)
                    if site_column
                    else ["All linked outlets"]
                )
                selected_site = st.selectbox(
                    "Linked outlet",
                    site_options,
                    key="quality_page_site",
                )

            site_filtered = company_filtered.copy()
            if site_column and selected_site != "All linked outlets":
                site_filtered = site_filtered.loc[
                    site_filtered[site_column].astype(str).eq(selected_site)
                ]

            parameter_options = available_values(site_filtered, parameter_column)
            if not parameter_options:
                st.warning("No 2025 water-quality indicators match those choices.")
                st.stop()
            default_parameter_index = next(
                (
                    index
                    for index, option in enumerate(parameter_options)
                    if "dissolved oxygen" in option.casefold()
                ),
                0,
            )
            with first_filters[2]:
                selected_parameter = st.selectbox(
                    "Water-quality indicator",
                    parameter_options,
                    index=default_parameter_index,
                    key="quality_page_parameter",
                )

            filtered = site_filtered.loc[
                site_filtered[parameter_column].astype(str).eq(selected_parameter)
            ].copy()

            if unit_column:
                unit_options = available_values(filtered, unit_column)
                if len(unit_options) > 1:
                    selected_unit = st.selectbox(
                        "Measurement unit",
                        unit_options,
                        key="quality_page_unit",
                    )
                    filtered = filtered.loc[
                        filtered[unit_column].astype(str).eq(selected_unit)
                    ]
                else:
                    selected_unit = unit_options[0] if unit_options else "Unit not reported"
            else:
                selected_unit = "Unit not reported"

            numeric_column = first_existing(
                filtered,
                ["exact_numeric_result", "result_numeric"],
            )
            if numeric_column:
                filtered["_result_numeric"] = pd.to_numeric(
                    filtered[numeric_column],
                    errors="coerce",
                )
            else:
                numeric_text = filtered[reported_result_column].astype(str).str.extract(
                    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
                    expand=False,
                )
                filtered["_result_numeric"] = pd.to_numeric(
                    numeric_text,
                    errors="coerce",
                )

            observation_column = first_existing(
                filtered,
                ["observation_id", "measurement_id"],
            )
            observations = (
                int(filtered[observation_column].nunique())
                if observation_column
                else int(len(filtered))
            )
            linked_sites = (
                int(filtered["location_id"].nunique())
                if "location_id" in filtered.columns
                else int(filtered[site_column].nunique()) if site_column else 0
            )
            monitoring_stations = (
                int(filtered["sampling_point_id"].nunique())
                if "sampling_point_id" in filtered.columns
                else int(filtered[station_column].nunique()) if station_column else 0
            )
            exact_results = int(filtered["_result_numeric"].notna().sum())

            indicator_key = selected_parameter.casefold()
            if "ammon" in indicator_key:
                indicator_type = "Potential pollutant concentration"
                indicator_meaning = (
                    "Ammoniacal nitrogen can come from sewage, agriculture and natural decay."
                )
                indicator_watch = (
                    "Higher results may need closer investigation; lower is generally preferable."
                )
                review_direction = "higher"
            elif "orthophosphate" in indicator_key or "phosphate" in indicator_key:
                indicator_type = "Nutrient concentration"
                indicator_meaning = (
                    "Orthophosphate is plant-available phosphorus. Too much can encourage excessive algae and plant growth."
                )
                indicator_watch = (
                    "Higher results may need closer investigation; lower is generally preferable."
                )
                review_direction = "higher"
            elif "bod" in indicator_key or "biochemical oxygen" in indicator_key:
                indicator_type = "Organic-pollution pressure indicator"
                indicator_meaning = (
                    "BOD shows how much oxygen microorganisms use while breaking down organic material."
                )
                indicator_watch = "Higher BOD can indicate greater organic pollution pressure."
                review_direction = "higher"
            elif "dissolved oxygen" in indicator_key and "percentage" in indicator_key:
                indicator_type = "Water-condition indicator—not a pollutant"
                indicator_meaning = (
                    "Oxygen saturation compares the oxygen present with the amount the water could hold."
                )
                indicator_watch = (
                    "Lower saturation may need closer review; temperature and sampling conditions also matter."
                )
                review_direction = "lower"
            elif "dissolved oxygen" in indicator_key:
                indicator_type = "Water-condition indicator—not a pollutant"
                indicator_meaning = (
                    "Dissolved oxygen is essential for fish and other aquatic organisms."
                )
                indicator_watch = (
                    "Lower oxygen may need closer review; well-oxygenated water is generally beneficial."
                )
                review_direction = "lower"
            elif indicator_key.strip() == "ph" or indicator_key.startswith("ph "):
                indicator_type = "Water-condition indicator—not a pollutant"
                indicator_meaning = "pH describes how acidic or alkaline the water is."
                indicator_watch = (
                    "Both unusually low and unusually high results need local environmental context."
                )
                review_direction = "both"
            elif "temperature" in indicator_key:
                indicator_type = "Environmental context—not a pollutant"
                indicator_meaning = (
                    "Temperature affects aquatic life and how much oxygen water can hold."
                )
                indicator_watch = (
                    "Interpret it by season, time and water-body type; it is not ranked here."
                )
                review_direction = "context"
            else:
                indicator_type = "Water-quality measurement"
                indicator_meaning = (
                    "This result describes a condition measured at a nearby monitoring station."
                )
                indicator_watch = (
                    "Interpret it using indicator-specific environmental guidance and local context."
                )
                review_direction = "context"

            metric_cards(
                [
                    {"label": "2025 measurements", "value": value_text(observations),
                     "note": selected_parameter, "accent": "#A8D8D0"},
                    {"label": "Linked outlets", "value": value_text(linked_sites),
                     "note": "Geographically linked sites", "accent": "#B7DDE5"},
                    {"label": "Monitoring stations", "value": value_text(monitoring_stations),
                     "note": "Environment Agency stations", "accent": "#CDBDDE"},
                    {"label": "Numeric results plotted", "value": value_text(exact_results),
                     "note": selected_unit, "accent": "#F1D39D"},
                ]
            )

            st.html(
                f"""
                <div class="edm-quality-guide" style="gap:.75rem;
                  margin:.75rem 0 1rem;">
                  <div style="padding:1rem;border-radius:18px;background:#E8F5F1;
                    border-top:6px solid #68A98F;box-shadow:0 9px 22px rgba(45,102,94,.08);">
                    <b style="color:#245F56">1. What is it?</b><br>
                    <span style="color:#486B67">{html.escape(indicator_type)}</span>
                  </div>
                  <div style="padding:1rem;border-radius:18px;background:#E8F3F8;
                    border-top:6px solid #68AFC2;box-shadow:0 9px 22px rgba(45,102,94,.08);">
                    <b style="color:#245F56">2. What does it tell us?</b><br>
                    <span style="color:#486B67">{html.escape(indicator_meaning)}</span>
                  </div>
                  <div style="padding:1rem;border-radius:18px;background:#FFF3DD;
                    border-top:6px solid #D8A34E;box-shadow:0 9px 22px rgba(45,102,94,.08);">
                    <b style="color:#6D5529">3. What should users notice?</b><br>
                    <span style="color:#655A43">{html.escape(indicator_watch)}</span>
                  </div>
                </div>
                """
            )

            chart_data = filtered.loc[
                filtered["_result_numeric"].notna()
                & filtered["_measurement_datetime"].notna()
            ].copy()

            st.html(
                """
                <div class="edm-quality-flow" style="
                  align-items:center;gap:.55rem;margin:.5rem 0 1rem;padding:1rem;
                  border-radius:18px;background:linear-gradient(110deg,#EAF6F0,#E9F4F8,#F2EDF7);
                  border:1px solid rgba(69,133,124,.18);text-align:center;">
                  <div><b>Recorded spill evidence</b><br><small>frequency, duration and risk</small></div>
                  <div style="font-size:1.4rem;color:#4A9C7D">&#8594;</div>
                  <div><b>Nearby 2025 monitoring</b><br><small>measured water conditions</small></div>
                  <div style="font-size:1.4rem;color:#4A9C7D">&#8594;</div>
                  <div><b>Priority for review</b><br><small>investigate—not proof of cause</small></div>
                </div>
                """
            )

            st.html(
                """
                <div style="margin:0 0 1rem;padding:.9rem 1rem;border-radius:16px;
                  background:rgba(255,255,255,.76);border:1px solid rgba(74,156,125,.20);">
                  <b style="color:#245F56">How this supports the spill-risk project</b>
                  <div style="display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.55rem;color:#486B67;">
                    <span style="padding:.35rem .65rem;border-radius:999px;background:#E8F5F1;">Adds environmental context</span>
                    <span style="padding:.35rem .65rem;border-radius:999px;background:#E8F3F8;">Helps compare linked places</span>
                    <span style="padding:.35rem .65rem;border-radius:999px;background:#FFF3DD;">Screens priorities for investigation</span>
                    <span style="padding:.35rem .65rem;border-radius:999px;background:#F1ECF7;">Creates a 2025 baseline</span>
                  </div>
                </div>
                """
            )

            review_group = site_column or station_column
            review_rows = pd.DataFrame()
            if not chart_data.empty and review_group:
                review_columns = [review_group]
                for column in [
                    company_column,
                    site_column,
                    station_column,
                    linked_risk_column,
                ]:
                    if column and column not in review_columns:
                        review_columns.append(column)
                review_rows = (
                    chart_data.groupby(
                        review_columns,
                        as_index=False,
                        dropna=False,
                    )
                    .agg(
                        median_result=("_result_numeric", "median"),
                        measurements=("_result_numeric", "size"),
                    )
                    .dropna(subset=["median_result"])
                )

            if len(review_rows) >= 4 and review_direction != "context":
                lower_quartile = review_rows["median_result"].quantile(.25)
                upper_quartile = review_rows["median_result"].quantile(.75)
                if review_direction == "lower":
                    priority_rows = review_rows.loc[
                        review_rows["median_result"].le(lower_quartile)
                    ].sort_values("median_result", ascending=True)
                    priority_reason = "lower end of the linked-station results"
                elif review_direction == "higher":
                    priority_rows = review_rows.loc[
                        review_rows["median_result"].ge(upper_quartile)
                    ].sort_values("median_result", ascending=False)
                    priority_reason = "higher end of the linked-station results"
                else:
                    middle_value = review_rows["median_result"].median()
                    priority_rows = review_rows.loc[
                        review_rows["median_result"].le(lower_quartile)
                        | review_rows["median_result"].ge(upper_quartile)
                    ].copy()
                    priority_rows["_distance_from_middle"] = (
                        priority_rows["median_result"] - middle_value
                    ).abs()
                    priority_rows = priority_rows.sort_values(
                        "_distance_from_middle",
                        ascending=False,
                    )
                    priority_reason = "outer ends of the linked-station results"

                section_header(
                    "Locations for closer review",
                    f"Descriptive screening: sites at the {priority_reason}. This is not a regulatory pass/fail test.",
                )
                review_display_columns = [
                    column for column in [
                        company_column,
                        site_column,
                        station_column,
                        linked_risk_column,
                        "median_result",
                        "measurements",
                    ]
                    if column and column in priority_rows.columns
                ]
                review_display = priority_rows[review_display_columns].copy()
                review_labels = {
                    "median_result": f"2025 median ({selected_unit})",
                    "measurements": "Measurements",
                }
                if company_column:
                    review_labels[company_column] = "Water company"
                if site_column:
                    review_labels[site_column] = "Linked outlet/site"
                if station_column:
                    review_labels[station_column] = "Monitoring station"
                if linked_risk_column:
                    review_labels[linked_risk_column] = "Linked spill-risk category"
                review_display = review_display.rename(columns=review_labels)
                st.dataframe(
                    review_display.round(3),
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "These locations are unusual only within this linked 2025 sample. "
                    "The flag is not evidence of a legal breach and does not show that the nearby outlet or water company caused the measurement."
                )
            elif review_direction == "context":
                st.info(
                    "This indicator is shown for environmental context. The dashboard does not rank locations without an appropriate indicator-specific standard."
                )
            elif not review_rows.empty:
                st.info(
                    "Too few linked locations are available to create a responsible comparative review list for this selection."
                )

            section_header(
                "2025 measurement results over time",
                f"Interactive results for {selected_parameter}; hover over a point for its site, station and reported value.",
            )

            if chart_data.empty:
                st.info(
                    "No exact numeric dated results are available for these choices. "
                    "The original reported results remain available in the table below."
                )
            else:
                colour_column = (
                    company_column
                    if company_column and chart_data[company_column].nunique() > 1
                    else site_column if site_column and chart_data[site_column].nunique() > 1
                    else station_column
                )
                hover_columns = [
                    column
                    for column in [
                        company_column,
                        site_column,
                        station_column,
                        "station_distance_km",
                        reported_result_column,
                        unit_column,
                    ]
                    if column and column in chart_data.columns
                ]
                pastel_palette = [
                    "#65B7C5", "#79BEAB", "#A8A6D8", "#E4AFC3", "#EBC27A",
                    "#94CFA4", "#8FBCE2", "#C8A8DD", "#EFA79E", "#A7D8D2",
                ]
                timeline = px.scatter(
                    chart_data,
                    x="_measurement_datetime",
                    y="_result_numeric",
                    color=colour_column,
                    hover_data=hover_columns,
                    color_discrete_sequence=pastel_palette,
                    title=f"{selected_parameter} · official 2025 observations",
                    labels={
                        "_measurement_datetime": "Measurement date",
                        "_result_numeric": f"Reported result ({selected_unit})",
                    },
                )
                timeline.update_traces(
                    marker=dict(size=11, opacity=.86, line=dict(color="#FFFFFF", width=1.4))
                )
                timeline.update_layout(
                    legend_title_text=(pretty(colour_column) if colour_column else "Series"),
                    hovermode="closest",
                )
                st.plotly_chart(
                    plot_style(timeline, 590),
                    use_container_width=True,
                    key="water_quality_2025_timeline",
                    config={"displayModeBar": False},
                )

                summary_group = site_column or station_column
                if summary_group:
                    site_summary = (
                        chart_data.groupby(summary_group, as_index=False)
                        .agg(
                            median_result=("_result_numeric", "median"),
                            measurements=("_result_numeric", "size"),
                        )
                        .sort_values(["measurements", "median_result"], ascending=[False, False])
                        .head(20)
                    )
                    indicator_name = selected_parameter.casefold()
                    lower_values_may_need_review = "dissolved oxygen" in indicator_name
                    higher_values_may_need_review = any(
                        term in indicator_name
                        for term in ["ammonia", "ammoniacal", "bod", "phosphate"]
                    )
                    # Plotly places the final category at the top of a horizontal
                    # bar chart. Reverse the dissolved-oxygen order so lower
                    # values, which may merit closer review, are shown first.
                    site_summary = site_summary.sort_values(
                        "median_result",
                        ascending=not lower_values_may_need_review,
                    )
                    if lower_values_may_need_review:
                        reading_note = (
                            "Site names are shown on the left. For dissolved oxygen, lower values "
                            "are placed near the top because lower oxygen can require closer review."
                        )
                    elif higher_values_may_need_review:
                        reading_note = (
                            "Site names are shown on the left. Higher measured values are placed "
                            "near the top for this indicator, but they are not automatically a breach."
                        )
                    else:
                        reading_note = (
                            "Site names are shown on the left. Bar length compares the median "
                            "measurement only; it does not identify a worst site."
                        )
                    st.html(
                        f"""
                        <div style="margin:.6rem 0 .8rem;padding:.85rem 1rem;border-radius:14px;
                          border-left:6px solid #68AFC2;background:linear-gradient(110deg,#E9F4F8,#F1ECF7);
                          color:#315E5A;">
                          <b>How to read the site comparison</b><br>{html.escape(reading_note)}
                        </div>
                        """
                    )
                    median_chart = go.Figure(
                        go.Bar(
                            x=site_summary["median_result"],
                            y=site_summary[summary_group],
                            orientation="h",
                            marker=dict(
                                color=site_summary["median_result"],
                                colorscale=[
                                    [0.0, "#DDF3ED"],
                                    [0.50, "#88C8D0"],
                                    [1.0, "#B8A9DD"],
                                ],
                                line=dict(color="#FFFFFF", width=1),
                                showscale=False,
                            ),
                            text=[f"{value:,.2f}" for value in site_summary["median_result"]],
                            textposition="outside",
                            customdata=site_summary[["measurements"]],
                            hovertemplate=(
                                "<b>Linked outlet/site:</b> %{y}<br>Median: %{x:,.3f} " + html.escape(selected_unit)
                                + "<br>Measurements: %{customdata[0]:,.0f}<extra></extra>"
                            ),
                        )
                    )
                    median_chart.update_layout(
                        title=f"Typical 2025 {selected_parameter} near each linked outlet",
                        xaxis_title=f"Median reported result ({selected_unit})",
                        yaxis_title="Linked outlet/site name",
                        showlegend=False,
                    )
                    st.plotly_chart(
                        plot_style(median_chart, max(480, 31 * len(site_summary) + 170)),
                        use_container_width=True,
                        key="water_quality_site_medians",
                        config={"displayModeBar": False},
                    )
                    st.caption(
                        "A site name identifies the outlet linked geographically to the nearby monitoring station. "
                        "The median is a descriptive summary—not a pass/fail score, proof of causation or a confirmed ranking of water-quality issues."
                    )

            with st.expander("See the 2025 measurements behind the graphs"):
                display_columns = [
                    column
                    for column in [
                        company_column,
                        site_column,
                        station_column,
                        "station_distance_km",
                        date_column,
                        parameter_column,
                        reported_result_column,
                        unit_column,
                    ]
                    if column and column in filtered.columns
                ]
                st.dataframe(
                    filtered[display_columns].sort_values(
                        date_column,
                        ascending=False,
                        na_position="last",
                    ) if date_column else filtered[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )
                download_table(filtered[display_columns], "filtered_2025_water_quality_measurements.csv")

            if not coverage.empty:
                with st.expander("See water-quality monitoring coverage by company"):
                    st.dataframe(coverage, use_container_width=True, hide_index=True)


# =============================================================================
# PAGE 6 — INDIVIDUAL PREDICTION
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
# PAGE 7 — AUDIT, SEARCH AND LIMITATIONS
# =============================================================================

else:
    section_header(
        "About the evidence",
        "See how the records were checked, find source information and understand what this website can and cannot show.",
    )

    audit_tab, search_tab, method_tab = st.tabs(
        ["How the data was checked", "Find a record", "How it works"]
    )

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
        section_header(
            "How a combined sewer works",
            "Normal flow goes to treatment; heavy rain can use the overflow route.",
        )
        render_sewer_story()
        st.html(
            """
            <div class="edm-journey">
              <div class="edm-journey-step"><span class="edm-journey-number">1</span><h4>Recorded information</h4><p>The map starts with cleaned records supplied for 2021–2025.</p></div>
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
      Sewage Overflow Insights · verified evidence, transparent forecasts and responsible interpretation
    </div>
    """,
)
