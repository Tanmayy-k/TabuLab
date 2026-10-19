"""
TabuLab — Synthetic Tabular Data Generator
Premium Dark UI  |  SDV 2.0  |  CTGAN
Generate. Validate. Trust.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ──────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TabuLab",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
#  DESIGN SYSTEM — FULL CSS BLOCK
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── CSS Variables ── */
:root {
  --bg:        #090A0A;
  --bg2:       #111313;
  --surface:   #171A18;
  --surface2:  #1E221E;
  --text:      #F5F5F0;
  --text2:     #A0A49F;
  --text3:     #5A5E59;
  --accent:    #C7FF55;
  --accent-dim:#8FB33C;
  --border:    #292D29;
  --border2:   #1E221E;
  --red:       #FF5C5C;
  --amber:     #FFB347;
  --green:     #4ADE80;
  --font:      'Inter', system-ui, sans-serif;
  --mono:      'JetBrains Mono', 'Fira Code', monospace;
}

/* ── Global Reset ── */
html, body, .stApp {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--font) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  padding: 0 !important;
  max-width: 100% !important;
}
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Typography ── */
h1,h2,h3,h4,h5,h6 {
  font-family: var(--font) !important;
  color: var(--text) !important;
  letter-spacing: -0.02em;
}
p, li, span, label { font-family: var(--font) !important; }
.stExpander [data-testid="stIconMaterial"] {
  font-family: "Material Symbols Rounded" !important;
}

/* ── Page wrapper ── */
.tl-page { padding: 0 48px 80px; max-width: 1280px; margin: 0 auto; }
@media(max-width:768px){ .tl-page { padding: 0 20px 60px; } }

/* ══ HERO ═══════════════════════════════════════════════════════════════════ */
.tl-hero {
  padding: 80px 0 64px;
  border-bottom: 1px solid var(--border);
  position: relative;
  overflow: hidden;
}
.tl-hero::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background-image:
    linear-gradient(var(--border2) 1px, transparent 1px),
    linear-gradient(90deg, var(--border2) 1px, transparent 1px);
  background-size: 48px 48px;
  opacity: 0.35;
  pointer-events: none;
}
.tl-hero::after {
  content: '';
  position: absolute;
  top: -120px; right: -80px;
  width: 480px; height: 480px;
  background: radial-gradient(ellipse at center, rgba(199,255,85,0.07) 0%, transparent 65%);
  pointer-events: none;
}
.tl-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.16em;
  color: var(--accent-dim); text-transform: uppercase;
  border: 1px solid rgba(199,255,85,0.2);
  background: rgba(199,255,85,0.05);
  padding: 4px 12px; border-radius: 100px;
  margin-bottom: 24px;
}
.tl-eyebrow-dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 6px var(--accent);
  animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
  0%,100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.5; transform: scale(0.8); }
}
.tl-headline {
  font-size: clamp(40px, 6vw, 72px);
  font-weight: 900;
  line-height: 1.05;
  letter-spacing: -0.04em;
  color: var(--text) !important;
  margin: 0 0 20px;
}
.tl-headline em {
  font-style: normal;
  color: var(--accent);
}
.tl-subheadline {
  font-size: 18px; font-weight: 400; line-height: 1.6;
  color: var(--text2); max-width: 560px;
  margin: 0 0 36px;
}

/* ══ LOGO NAV ════════════════════════════════════════════════════════════════ */
.tl-nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 20px 48px;
  border-bottom: 1px solid var(--border);
  background: rgba(9,10,10,0.85);
  backdrop-filter: blur(12px);
  position: sticky; top: 0; z-index: 100;
}
@media(max-width:768px){ .tl-nav { padding: 16px 20px; } }
.tl-logo {
  display: flex; align-items: center; gap: 10px;
  font-size: 18px; font-weight: 800; letter-spacing: -0.04em;
  color: var(--text);
}
.tl-logo-mark {
  width: 28px; height: 28px;
  background: var(--accent);
  clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
  display: flex; align-items: center; justify-content: center;
}
.tl-nav-tag {
  font-size: 11px; color: var(--text3); font-family: var(--mono);
  letter-spacing: 0.08em;
}

/* ══ PIPELINE STEPPER ════════════════════════════════════════════════════════ */
.tl-stepper {
  display: flex; align-items: center;
  padding: 20px 0; overflow-x: auto;
  gap: 0; scrollbar-width: none;
}
.tl-stepper::-webkit-scrollbar { display: none; }
.tl-step {
  display: flex; align-items: center; gap: 0;
  cursor: pointer; flex-shrink: 0;
}
.tl-step-body {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 16px; border-radius: 8px;
  transition: background 0.15s;
}
.tl-step-body:hover { background: var(--surface2); }
.tl-step-body.active { background: var(--surface2); }
.tl-step-num {
  width: 26px; height: 26px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 11px; font-weight: 700; flex-shrink: 0;
  border: 1.5px solid var(--border);
  color: var(--text3);
  font-family: var(--mono);
}
.tl-step-num.done {
  background: var(--accent); border-color: var(--accent);
  color: var(--bg);
}
.tl-step-num.active {
  border-color: var(--accent); color: var(--accent);
}
.tl-step-label {
  font-size: 12px; font-weight: 500;
  color: var(--text3);
  white-space: nowrap;
}
.tl-step-label.active { color: var(--text); }
.tl-step-label.done { color: var(--text2); }
.tl-step-connector {
  width: 32px; height: 1px;
  background: var(--border);
  flex-shrink: 0;
}

/* ══ SECTION HEADERS ══════════════════════════════════════════════════════════ */
.tl-section {
  padding: 40px 0 0;
}
.tl-section-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: var(--text3);
  margin-bottom: 8px;
  font-family: var(--mono);
}
.tl-section-title {
  font-size: 24px; font-weight: 700; letter-spacing: -0.03em;
  color: var(--text); margin: 0 0 4px;
}
.tl-section-desc {
  font-size: 14px; color: var(--text2); line-height: 1.6;
  margin: 0 0 28px;
}

/* ══ CARDS & SURFACES ═════════════════════════════════════════════════════════ */
.tl-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 24px;
}
.tl-card-sm {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
}
.tl-ctgan-explainer {
  display: grid;
  gap: 16px;
  padding: 20px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.tl-ctgan-intro {
  color: var(--text2);
  font-size: 13px;
  line-height: 1.65;
}
.tl-ctgan-intro strong { color: var(--text); }
.tl-ctgan-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
}
.tl-ctgan-card {
  padding: 14px 16px;
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 2px solid var(--accent-dim);
  border-radius: 8px;
}
.tl-ctgan-card-title {
  margin-bottom: 5px;
  color: var(--accent);
  font-size: 12px;
  font-weight: 700;
}
.tl-ctgan-card-desc {
  color: var(--text2);
  font-size: 12px;
  line-height: 1.55;
}
.tl-ctgan-note {
  display: grid;
  gap: 4px;
  padding: 12px 14px;
  background: rgba(199,255,85,0.05);
  border: 1px solid rgba(199,255,85,0.16);
  border-radius: 8px;
  color: var(--text2);
  font-size: 12px;
  line-height: 1.6;
}
.tl-ctgan-note-title {
  color: var(--accent);
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
@media(max-width:640px){ .tl-ctgan-grid { grid-template-columns: 1fr; } }

/* ══ METRIC TILES ═════════════════════════════════════════════════════════════ */
.tl-metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 20px;
}
.tl-metric-label {
  font-size: 11px; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--text3);
  margin-bottom: 8px; font-family: var(--mono);
}
.tl-metric-value {
  font-size: 28px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--text); font-variant-numeric: tabular-nums;
  line-height: 1;
}
.tl-metric-sub {
  font-size: 12px; color: var(--text3); margin-top: 4px;
}

/* ══ STATUS BADGES ════════════════════════════════════════════════════════════ */
.tl-badge {
  display: inline-flex; align-items: center; gap: 5px;
  font-size: 11px; font-weight: 600; letter-spacing: 0.06em;
  padding: 3px 10px; border-radius: 100px;
}
.tl-badge-idle  { background:rgba(90,94,89,0.18); color:var(--text3); border:1px solid var(--border); }
.tl-badge-ready { background:rgba(199,255,85,0.1); color:var(--accent); border:1px solid rgba(199,255,85,0.25); }
.tl-badge-done  { background:rgba(74,222,128,0.1); color:var(--green); border:1px solid rgba(74,222,128,0.25); }
.tl-badge-fail  { background:rgba(255,92,92,0.1);  color:var(--red);   border:1px solid rgba(255,92,92,0.25); }

/* ══ CALLOUTS ═════════════════════════════════════════════════════════════════ */
.tl-callout {
  border-radius: 8px; padding: 14px 18px;
  font-size: 13px; line-height: 1.6;
  display: flex; gap: 10px; align-items: flex-start;
}
.tl-callout-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; }
.tl-callout-info  { background:rgba(74,165,220,0.08); border:1px solid rgba(74,165,220,0.2); color:#7BC8E8; }
.tl-callout-warn  { background:rgba(255,179,71,0.08); border:1px solid rgba(255,179,71,0.2); color:#FFB347; }
.tl-callout-ok    { background:rgba(74,222,128,0.08); border:1px solid rgba(74,222,128,0.2); color:var(--green); }
.tl-callout-subtle{ background:var(--surface); border:1px solid var(--border); color:var(--text2); }

/* ══ FEATURE TRIO ═════════════════════════════════════════════════════════════ */
.tl-feature-grid {
  display: grid; grid-template-columns: repeat(3,1fr); gap: 16px;
  margin: 32px 0;
}
@media(max-width:768px){ .tl-feature-grid { grid-template-columns:1fr; } }
.tl-feature {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 28px 24px;
  position: relative; overflow: hidden;
}
.tl-feature::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0;
  height: 2px; background: linear-gradient(90deg, transparent, var(--accent), transparent);
  opacity: 0;
  transition: opacity 0.3s;
}
.tl-feature:hover::before { opacity: 0.6; }
.tl-feature-num {
  font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
  color: var(--accent); font-family: var(--mono); margin-bottom: 12px;
}
.tl-feature-title {
  font-size: 17px; font-weight: 700; letter-spacing: -0.02em;
  color: var(--text); margin-bottom: 8px;
}
.tl-feature-desc { font-size: 13px; color: var(--text2); line-height: 1.6; }

/* ══ WORKFLOW FLOW ════════════════════════════════════════════════════════════ */
.tl-flow {
  display: flex; align-items: center; gap: 0;
  padding: 24px 0; overflow-x: auto;
  scrollbar-width: none;
}
.tl-flow::-webkit-scrollbar { display: none; }
.tl-flow-node {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 20px;
  font-size: 13px; font-weight: 600; color: var(--text);
  white-space: nowrap; flex-shrink: 0;
  text-align: center;
}
.tl-flow-node-sub {
  font-size: 10px; color: var(--text3); font-weight: 400;
  margin-top: 2px; font-family: var(--mono);
}
.tl-flow-arrow {
  color: var(--border); font-size: 16px; flex-shrink: 0; padding: 0 8px;
}
.tl-flow-node.hl {
  border-color: rgba(199,255,85,0.3);
  background: rgba(199,255,85,0.04);
}

/* ══ COMPARISON PANELS ════════════════════════════════════════════════════════ */
.tl-compare-label {
  font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; padding: 4px 10px;
  border-radius: 4px; display: inline-block;
  margin-bottom: 10px; font-family: var(--mono);
}
.tl-compare-real { background:rgba(79,142,247,0.15); color:#4f8ef7; }
.tl-compare-syn  { background:rgba(199,255,85,0.12); color:var(--accent); }

/* ══ RISK INDICATOR ═══════════════════════════════════════════════════════════ */
.tl-risk-bar-track {
  background: var(--surface2); border-radius: 4px; height: 6px;
  margin: 6px 0 12px; overflow: hidden;
}
.tl-risk-bar-fill {
  height: 100%; border-radius: 4px;
  transition: width 0.6s ease;
}

/* ══ DIVIDERS ═════════════════════════════════════════════════════════════════ */
.tl-divider {
  border: none; border-top: 1px solid var(--border);
  margin: 32px 0;
}

/* ══ UPLOAD HINT ══════════════════════════════════════════════════════════════ */
.tl-upload-hint {
  font-size: 12px; color: var(--text3); font-family: var(--mono);
  margin-top: 6px;
}

/* ══ Streamlit overrides ══════════════════════════════════════════════════════ */
.stButton > button {
  background: var(--accent) !important;
  color: #0A0A0A !important;
  border: none !important;
  font-weight: 700 !important;
  font-family: var(--font) !important;
  letter-spacing: -0.01em !important;
  border-radius: 7px !important;
  padding: 10px 22px !important;
  font-size: 13px !important;
  transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button[kind="secondary"] {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
.stDownloadButton > button {
  background: var(--surface) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  font-weight: 600 !important;
  font-family: var(--font) !important;
  border-radius: 7px !important;
  padding: 9px 20px !important;
  font-size: 13px !important;
}
.stDownloadButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
}
div[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 16px 20px !important;
}
div[data-testid="stMetricLabel"] > div {
  font-size: 11px !important;
  font-weight: 600 !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  color: var(--text3) !important;
  font-family: var(--mono) !important;
}
div[data-testid="stMetricValue"] > div {
  font-size: 26px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  letter-spacing: -0.03em !important;
}
div[data-testid="stMetricDelta"] > div { font-size: 12px !important; }
[data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
[data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text3) !important;
  font-family: var(--font) !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  padding: 10px 18px !important;
  border-bottom: 2px solid transparent !important;
}
[aria-selected="true"][data-baseweb="tab"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}
[data-baseweb="tab-highlight"] { display: none !important; }
div[data-testid="stFileUploader"] {
  background: var(--surface) !important;
  border: 1px dashed var(--border) !important;
  border-radius: 10px !important;
  padding: 8px !important;
}
div[data-testid="stFileUploader"]:hover {
  border-color: rgba(199,255,85,0.35) !important;
}
.stSlider > div > div { color: var(--accent) !important; }
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 7px !important;
}
.stNumberInput > div > div > input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 7px !important;
}
.stTextInput > div > div > input {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
}
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  overflow: hidden !important;
}
.stAlert {
  border-radius: 8px !important;
  font-family: var(--font) !important;
  font-size: 13px !important;
}
div[data-testid="stExpander"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
}
.stProgress > div > div > div {
  background: var(--accent) !important;
}
[data-testid="stForm"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 24px !important;
}
/* ── JSON display ── */
.stJson { background: var(--surface2) !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ──────────────────────────────────────────────────────────────────────────────
_state_defaults = {
    "df": None,
    "df_train": None,
    "df_test": None,
    "metadata": None,
    "synthesizer": None,
    "synthetic_df": None,
    "quality_score": None,
    "num_cols": [],
    "cat_cols": [],
    "dataset_name": "",
    "active_stage": 0,        # 0=home, 1=dataset, 2=train, 3=generate, 4=evaluate, 5=report
    "ml_results": None,
    "privacy_results": None,
    "quality_details": None,
    "train_epochs": 100,
    "train_batch": 200,
}
for k, v in _state_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────────────────────
#  HELPER FUNCTIONS  (identical logic to working implementation)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data
def load_demo_dataset():
    try:
        df = sns.load_dataset("titanic")
        return df.drop(columns=["alive", "embark_town", "who", "adult_male", "deck"], errors="ignore")
    except Exception:
        rng = np.random.default_rng(42)
        n = 500
        return pd.DataFrame({
            "age":      rng.integers(18, 70, n).astype(float),
            "fare":     rng.uniform(5, 300, n).round(2),
            "pclass":   rng.choice([1, 2, 3], n),
            "sex":      rng.choice(["male", "female"], n),
            "embarked": rng.choice(["S", "C", "Q"], n),
            "survived": rng.integers(0, 2, n),
        })


def detect_column_types(df):
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return num, cat


def preprocess_dataframe(df):
    df = df.copy()
    thresh = int(len(df) * 0.2)
    df = df.dropna(axis=1, thresh=thresh)
    num_cols, cat_cols = detect_column_types(df)
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0] if not df[c].mode().empty else "Unknown")
        df[c] = df[c].astype(str)
    return df, num_cols, cat_cols


def build_metadata(df):
    from sdv.metadata import Metadata
    return Metadata.detect_from_dataframes({"table": df})


def train_ctgan(df, metadata, epochs, batch_size):
    from sdv.single_table import CTGANSynthesizer
    synth = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, verbose=False)
    synth.fit({"table": df})
    return synth


def generate_synthetic(synthesizer, n_rows):
    result = synthesizer.sample(num_rows=n_rows)
    return result["table"] if isinstance(result, dict) else result


def run_quality_report(real_df, syn_df, metadata):
    try:
        from sdmetrics.reports.single_table import QualityReport
        tmeta = metadata.get_table_metadata("table").to_dict()
        single = tmeta.get("tables", {}).get("table", tmeta)
        report = QualityReport()
        report.generate(real_df, syn_df, single, verbose=False)
        return report.get_score(), report.get_details(property_name="Column Shapes")
    except Exception as e:
        return None, str(e)


def compute_privacy_metrics(train_df, syn_df, num_cols, cat_cols):
    res = {}
    merged = syn_df.merge(train_df, how="inner")
    res["exact_matches"] = len(merged)
    res["exact_match_rate"] = len(merged) / max(len(syn_df), 1)
    try:
        cols = [c for c in num_cols + cat_cols if c in train_df and c in syn_df]
        if not cols:
            res["mean_nn_dist"] = None
            return res
        le = LabelEncoder()
        real_enc = train_df[cols].copy()
        syn_enc  = syn_df[cols].copy()
        for c in cat_cols:
            if c in cols:
                all_v = pd.concat([real_enc[c], syn_enc[c]]).astype(str)
                le.fit(all_v)
                real_enc[c] = le.transform(real_enc[c].astype(str))
                syn_enc[c]  = le.transform(syn_enc[c].astype(str))
        real_arr = real_enc.values.astype(float)
        syn_arr  = syn_enc.values.astype(float)
        rng_range = real_arr.max(0) - real_arr.min(0)
        rng_range[rng_range == 0] = 1
        real_n = (real_arr - real_arr.min(0)) / rng_range
        syn_n  = (syn_arr  - real_arr.min(0)) / rng_range
        rng = np.random.default_rng(42)
        if len(syn_n) > 500:
            syn_n = syn_n[rng.choice(len(syn_n), 500, replace=False)]
        dists = [np.linalg.norm(real_n - row, axis=1).min() for row in syn_n]
        res["mean_nn_dist"] = float(np.mean(dists))
        res["min_nn_dist"]  = float(np.min(dists))
    except Exception as e:
        res["mean_nn_dist"] = None
        res["nn_error"] = str(e)
    return res


def encode_for_ml(df, target, num_cols, cat_cols):
    feature_cols = [c for c in num_cols + cat_cols if c != target]
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    return X, df[target].copy()


def run_ml_utility(real_train, real_test, syn_df, target, num_cols, cat_cols):
    is_clf = (real_train[target].dtype == object or real_train[target].nunique() <= 15)
    X_real, y_real = encode_for_ml(real_train, target, num_cols, cat_cols)
    X_test, y_test = encode_for_ml(real_test,  target, num_cols, cat_cols)
    if target not in syn_df.columns:
        return None, "Target column not in synthetic data."
    X_syn, y_syn = encode_for_ml(syn_df, target, num_cols, cat_cols)
    X_syn  = X_syn.reindex(columns=X_real.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_real.columns, fill_value=0)
    if is_clf:
        le_t = LabelEncoder()
        y_real = le_t.fit_transform(y_real.astype(str))
        y_syn  = le_t.transform(pd.Series(y_syn).astype(str).map(
            lambda x: x if x in le_t.classes_ else le_t.classes_[0]))
        y_test = le_t.transform(pd.Series(y_test).astype(str).map(
            lambda x: x if x in le_t.classes_ else le_t.classes_[0]))
        base = RandomForestClassifier(n_estimators=50, random_state=42)
        base.fit(X_real, y_real)
        y_pb = base.predict(X_test)
        tstr = RandomForestClassifier(n_estimators=50, random_state=42)
        tstr.fit(X_syn, y_syn)
        y_pt = tstr.predict(X_test)
        avg = "binary" if len(le_t.classes_) == 2 else "macro"
        return {"task": "Classification", "target": target,
                "baseline_accuracy": accuracy_score(y_test, y_pb),
                "tstr_accuracy":     accuracy_score(y_test, y_pt),
                "baseline_f1":       f1_score(y_test, y_pb, average=avg, zero_division=0),
                "tstr_f1":           f1_score(y_test, y_pt, average=avg, zero_division=0),
                "baseline_precision": precision_score(y_test, y_pb, average=avg, zero_division=0),
                "tstr_precision":    precision_score(y_test, y_pt, average=avg, zero_division=0),
                "baseline_recall":   recall_score(y_test, y_pb, average=avg, zero_division=0),
                "tstr_recall":       recall_score(y_test, y_pt, average=avg, zero_division=0)}, None
    else:
        y_real = y_real.astype(float)
        y_syn  = y_syn.astype(float)
        y_test = y_test.astype(float)
        base = RandomForestRegressor(n_estimators=50, random_state=42)
        base.fit(X_real, y_real)
        y_pb = base.predict(X_test)
        tstr = RandomForestRegressor(n_estimators=50, random_state=42)
        tstr.fit(X_syn, y_syn)
        y_pt = tstr.predict(X_test)
        return {"task": "Regression", "target": target,
                "baseline_mae":  mean_absolute_error(y_test, y_pb),
                "tstr_mae":      mean_absolute_error(y_test, y_pt),
                "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, y_pb))),
                "tstr_rmse":     float(np.sqrt(mean_squared_error(y_test, y_pt)))}, None


# ──────────────────────────────────────────────────────────────────────────────
#  CHART THEME  (consistent dark matplotlib style)
# ──────────────────────────────────────────────────────────────────────────────
def apply_chart_style(fig, ax_or_axes):
    fig.patch.set_facecolor("#111313")
    axes = ax_or_axes if isinstance(ax_or_axes, (list, np.ndarray)) else [ax_or_axes]
    for ax in np.array(axes).flatten():
        ax.set_facecolor("#171A18")
        ax.tick_params(colors="#5A5E59", labelsize=9)
        ax.xaxis.label.set_color("#A0A49F")
        ax.yaxis.label.set_color("#A0A49F")
        ax.title.set_color("#F5F5F0")
        for spine in ax.spines.values():
            spine.set_color("#292D29")


# ──────────────────────────────────────────────────────────────────────────────
#  HTML HELPERS  — all produce properly rendered HTML via unsafe_allow_html
# ──────────────────────────────────────────────────────────────────────────────

def section_label(text, style=""):
    """Render a small mono uppercase section label."""
    extra = f' style="{style}"' if style else ""
    st.markdown(
        f'<div class="tl-section-label"{extra}>{text}</div>',
        unsafe_allow_html=True,
    )


def callout(kind, icon, body):
    """Render a callout box. kind: info | warn | ok | subtle."""
    st.markdown(
        f'<div class="tl-callout tl-callout-{kind}">'
        f'<div class="tl-callout-icon">{icon}</div>'
        f'<div>{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def compare_label(kind, text):
    """Render a real/synthetic comparison label badge."""
    st.markdown(
        f'<div class="tl-compare-label tl-compare-{kind}">{text}</div>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<hr class="tl-divider">', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  PIPELINE STATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def stage_status():
    """Return done-flags dict for the 4 pipeline stages."""
    s = st.session_state
    return {
        1: s.df_train is not None,       # Dataset + preprocess
        2: s.synthesizer is not None,    # Trained
        3: s.synthetic_df is not None,   # Generated
        4: s.quality_score is not None,  # Evaluated
    }


def render_nav():
    """Sticky top navigation bar."""
    st.markdown("""
<div class="tl-nav">
  <div class="tl-logo">
    <div class="tl-logo-mark"></div>
    TabuLab
  </div>
  <div class="tl-nav-tag">PS 65 &nbsp;·&nbsp; SDV 2.0 &nbsp;·&nbsp; CTGAN</div>
</div>
""", unsafe_allow_html=True)


def render_stepper(active: int):
    """Horizontal pipeline stepper. active = 1..4."""
    done = stage_status()
    stages = [
        (1, "Dataset"),
        (2, "Train CTGAN"),
        (3, "Generate"),
        (4, "Evaluate"),
    ]
    parts = ['<div class="tl-stepper">']
    for i, (num, label) in enumerate(stages):
        is_done   = done.get(num, False)
        is_active = (num == active)
        num_cls   = "done" if is_done else ("active" if is_active else "")
        lbl_cls   = "done" if is_done else ("active" if is_active else "")
        num_char  = "&#10003;" if is_done else str(num)
        active_cls = "active" if is_active else ""
        parts.append(
            f'<div class="tl-step">'
            f'<div class="tl-step-body {active_cls}">'
            f'<div class="tl-step-num {num_cls}">{num_char}</div>'
            f'<div class="tl-step-label {lbl_cls}">{label}</div>'
            f'</div>'
            f'</div>'
        )
        if i < len(stages) - 1:
            parts.append('<div class="tl-step-connector"></div>')
    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
#  NAVIGATION (top tabs, no sidebar)
# ──────────────────────────────────────────────────────────────────────────────
render_nav()

# Open main page wrapper
st.markdown('<div class="tl-page">', unsafe_allow_html=True)

NAV_LABELS = ["Home", "Dataset", "Train", "Generate", "Evaluate", "Report"]

selected_tab = st.session_state.active_stage
top_cols = st.columns(len(NAV_LABELS))
for i, (col, label) in enumerate(zip(top_cols, NAV_LABELS)):
    with col:
        if st.button(
            label,
            key=f"nav_{i}",
            use_container_width=True,
            type="primary" if i == selected_tab else "secondary",
        ):
            st.session_state.active_stage = i
            st.rerun()

page_idx = st.session_state.active_stage

divider()

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 0 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page_idx == 0:

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="tl-hero">
  <div class="tl-eyebrow">
    <span class="tl-eyebrow-dot"></span>
    Synthetic Data Intelligence
  </div>
  <div class="tl-headline">Create data.<br><em>Not exposure.</em></div>
  <div class="tl-subheadline">
    Generate realistic synthetic tabular records using CTGAN —
    a Conditional Tabular GAN. Measure statistical quality, evaluate
    machine-learning utility, and investigate privacy risks, without
    sharing your real data.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── CTA row ───────────────────────────────────────────────────────────────
    cta1, cta2, _ = st.columns([1, 1, 4])
    with cta1:
        if st.button("Start generating →", key="cta_start", type="primary", use_container_width=True):
            st.session_state.active_stage = 1
            st.rerun()
    with cta2:
        if st.button("Explore workflow", key="cta_workflow", type="secondary", use_container_width=True):
            st.session_state.active_stage = 1
            st.rerun()

    # ── Three feature pillars ─────────────────────────────────────────────────
    st.markdown("""
<div class="tl-feature-grid">
  <div class="tl-feature">
    <div class="tl-feature-num">01 — GENERATE</div>
    <div class="tl-feature-title">Synthetic Records</div>
    <div class="tl-feature-desc">
      Upload any CSV or load the built-in demo. CTGAN learns the statistical
      patterns and generates artificial rows that preserve distributions,
      correlations, and category frequencies.
    </div>
  </div>
  <div class="tl-feature">
    <div class="tl-feature-num">02 — MEASURE</div>
    <div class="tl-feature-title">Statistical Quality</div>
    <div class="tl-feature-desc">
      SDV's quality report scores how closely synthetic distributions
      match the real data across every column. Visual overlays make the
      comparison immediate and intuitive.
    </div>
  </div>
  <div class="tl-feature">
    <div class="tl-feature-num">03 — EVALUATE</div>
    <div class="tl-feature-title">Utility &amp; Privacy</div>
    <div class="tl-feature-desc">
      Train-on-Synthetic, Test-on-Real (TSTR) measures downstream
      predictive value. Exact-match and nearest-neighbour checks give
      a heuristic picture of re-identification risk.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Workflow flow diagram ─────────────────────────────────────────────────
    section_label("The pipeline")
    st.markdown("""
<div class="tl-flow">
  <div class="tl-flow-node hl">
    Dataset
    <div class="tl-flow-node-sub">CSV upload or demo</div>
  </div>
  <div class="tl-flow-arrow">&#8594;</div>
  <div class="tl-flow-node">
    Preprocess
    <div class="tl-flow-node-sub">clean &#183; split &#183; metadata</div>
  </div>
  <div class="tl-flow-arrow">&#8594;</div>
  <div class="tl-flow-node hl">
    CTGAN Training
    <div class="tl-flow-node-sub">configurable epochs</div>
  </div>
  <div class="tl-flow-arrow">&#8594;</div>
  <div class="tl-flow-node">
    Generation
    <div class="tl-flow-node-sub">N synthetic rows</div>
  </div>
  <div class="tl-flow-arrow">&#8594;</div>
  <div class="tl-flow-node hl">
    Quality &#183; Utility &#183; Privacy
    <div class="tl-flow-node-sub">SDV &#183; TSTR &#183; NN distance</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Evaluation philosophy ─────────────────────────────────────────────────
    divider()
    section_label("Evaluation philosophy")
    ph1, ph2, ph3 = st.columns(3)
    with ph1:
        st.markdown("""
<div class="tl-card-sm">
  <div class="tl-feature-num">Statistical Fidelity</div>
  <div style="font-size:13px;color:#A0A49F;line-height:1.6;margin-top:6px">
    Does the synthetic data look like the real data? Distributions,
    correlations, and category frequencies are compared column-by-column
    using the SDV quality report (0&#8211;100 scale).
  </div>
</div>
""", unsafe_allow_html=True)
    with ph2:
        st.markdown("""
<div class="tl-card-sm">
  <div class="tl-feature-num">ML Utility (TSTR)</div>
  <div style="font-size:13px;color:#A0A49F;line-height:1.6;margin-top:6px">
    Is the synthetic data useful for training models? A classifier trained
    on synthetic records is evaluated on real held-out data and compared
    against a real-data baseline.
  </div>
</div>
""", unsafe_allow_html=True)
    with ph3:
        st.markdown("""
<div class="tl-card-sm">
  <div class="tl-feature-num">Privacy Risk (Heuristic)</div>
  <div style="font-size:13px;color:#A0A49F;line-height:1.6;margin-top:6px">
    Are synthetic rows too similar to real training records? Exact-match
    counts and nearest-neighbour distances provide indicators &#8212; not
    formal anonymity guarantees.
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Privacy disclaimer ────────────────────────────────────────────────────
    divider()
    callout(
        "warn", "&#9888;",
        "<strong>Privacy disclaimer.</strong> "
        "Synthetic data generated by CTGAN is not automatically anonymous. "
        "The checks in this application are heuristics &#8212; they cannot prove that "
        "re-identification is impossible. No formal differential-privacy mechanism "
        "is implemented. Do not publish synthetic data derived from sensitive "
        "personal records without a professional privacy audit."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page_idx == 1:
    render_stepper(1)

    st.markdown("""
<div class="tl-section">
  <div class="tl-section-label">Stage 01</div>
  <div class="tl-section-title">Dataset</div>
  <div class="tl-section-desc">Upload a CSV file or load the built-in Titanic demo. TabuLab will
  clean, split, and prepare the data for CTGAN training.</div>
</div>
""", unsafe_allow_html=True)

    tab_upload, tab_demo, tab_eda = st.tabs(
        ["Upload CSV", "Demo Dataset", "Explore & Preprocess"]
    )

    with tab_upload:
        st.write("")
        uploaded = st.file_uploader(
            "Drop your CSV here",
            type=["csv"],
            help="Recommended: up to 50 000 rows and 50 columns. Larger files will train slowly.",
            label_visibility="visible",
        )
        st.markdown(
            '<div class="tl-upload-hint">Supported format: UTF-8 CSV &nbsp;&#183;&nbsp; Max recommended: 50k rows</div>',
            unsafe_allow_html=True,
        )

        if uploaded:
            try:
                df_up = pd.read_csv(uploaded)
                if df_up.empty:
                    st.error("The uploaded file is empty.")
                elif len(df_up.columns) < 2:
                    st.error("Dataset must have at least 2 columns.")
                else:
                    st.session_state.df = df_up
                    st.session_state.dataset_name = uploaded.name
                    # Reset downstream state when new data is loaded
                    for k in ["df_train", "df_test", "metadata", "synthesizer",
                               "synthetic_df", "quality_score", "ml_results",
                               "privacy_results", "quality_details"]:
                        st.session_state[k] = None
                    st.session_state.num_cols = []
                    st.session_state.cat_cols = []
                    callout(
                        "ok", "&#10003;",
                        f"<strong>{uploaded.name}</strong> loaded &#8212; "
                        f"{df_up.shape[0]:,} rows &#215; {df_up.shape[1]} columns"
                    )
                    st.dataframe(df_up.head(8), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to read file: {e}")

    with tab_demo:
        st.write("")
        st.markdown("""
<div class="tl-card-sm" style="margin-bottom:20px">
  <div class="tl-feature-num">Titanic Passenger Dataset</div>
  <div style="font-size:13px;color:#A0A49F;margin-top:6px;line-height:1.6">
    891 passengers &#183; 10 features &#183; public domain<br>
    Includes numerical (age, fare) and categorical (sex, pclass, embarked) columns.
    A suitable demo for exploring the full CTGAN pipeline.
  </div>
</div>
""", unsafe_allow_html=True)
        if st.button("Load Titanic demo", key="btn_load_demo", type="primary"):
            with st.spinner("Loading…"):
                df_demo = load_demo_dataset()
                st.session_state.df = df_demo
                st.session_state.dataset_name = "titanic_demo"
                for k in ["df_train", "df_test", "metadata", "synthesizer",
                           "synthetic_df", "quality_score", "ml_results",
                           "privacy_results", "quality_details"]:
                    st.session_state[k] = None
                st.session_state.num_cols = []
                st.session_state.cat_cols = []
            callout(
                "ok", "&#10003;",
                f"Titanic demo loaded &#8212; {df_demo.shape[0]:,} rows &#215; {df_demo.shape[1]} columns"
            )
            st.dataframe(df_demo.head(8), use_container_width=True)

    with tab_eda:
        if st.session_state.df is None:
            callout("subtle", "&#9675;", "Load a dataset first using the Upload or Demo tab.")
        else:
            df = st.session_state.df
            num_cols_raw, cat_cols_raw = detect_column_types(df)

            # ── Quick stats row ───────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows",           f"{df.shape[0]:,}")
            m2.metric("Columns",        df.shape[1])
            m3.metric("Missing values", int(df.isna().sum().sum()))
            m4.metric("Duplicate rows", int(df.duplicated().sum()))

            st.write("")

            # ── Column overview table ─────────────────────────────────────────
            section_label("Column overview")
            overview = pd.DataFrame({
                "Type":          df.dtypes.astype(str),
                "Non-null":      df.notnull().sum(),
                "Missing %":     (df.isnull().mean() * 100).round(2),
                "Unique values": df.nunique(),
            })
            st.dataframe(overview, use_container_width=True)

            # ── Numerical histograms ──────────────────────────────────────────
            if num_cols_raw:
                section_label("Numerical distributions", "margin-top:24px")
                n = min(len(num_cols_raw), 6)
                nc = min(3, n)
                nr = (n + nc - 1) // nc
                fig, axes = plt.subplots(nr, nc, figsize=(13, 3.5 * nr))
                axes = np.array(axes).flatten()
                for i, c in enumerate(num_cols_raw[:n]):
                    df[c].dropna().plot.hist(ax=axes[i], bins=25, color="#C7FF55",
                                             edgecolor="#111313", alpha=0.85)
                    axes[i].set_title(c, fontsize=10, pad=6)
                for j in range(n, len(axes)):
                    axes[j].set_visible(False)
                apply_chart_style(fig, axes)
                plt.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── Categorical bars ──────────────────────────────────────────────
            if cat_cols_raw:
                section_label("Categorical distributions", "margin-top:24px")
                show = cat_cols_raw[:4]
                n = len(show)
                if n > 1:
                    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5))
                else:
                    fig, ax_single = plt.subplots(1, 1, figsize=(5, 3.5))
                    axes = [ax_single]
                axes = np.array(axes).flatten()
                palette = ["#C7FF55", "#4f8ef7", "#ff6b6b", "#FFB347"]
                for i, c in enumerate(show):
                    vc = df[c].value_counts().head(8)
                    axes[i].bar(range(len(vc)), vc.values, color=palette[i % len(palette)])
                    axes[i].set_xticks(range(len(vc)))
                    axes[i].set_xticklabels(vc.index, rotation=30, ha="right", fontsize=8)
                    axes[i].set_title(c, fontsize=10, pad=6)
                apply_chart_style(fig, axes)
                plt.tight_layout(pad=1.2)
                st.pyplot(fig, use_container_width=True)
                plt.close()

            # ── Preprocess & split ────────────────────────────────────────────
            divider()
            section_label("Preprocessing & train/test split")
            callout(
                "subtle", "&#9675;",
                "<div style=\"font-size:13px\">"
                "<strong>Automatic cleaning:</strong> columns with &gt;80% missing values are dropped. "
                "Numerical gaps fill with the column median. Categorical gaps fill with the column mode. "
                "Boolean and object columns become string categories. "
                "The test split is held out entirely &#8212; CTGAN never sees those rows."
                "</div>"
            )

            st.write("")
            test_size = st.slider(
                "Hold-out test split", 10, 40, 20,
                format="%d%%",
                help="Rows reserved for ML utility evaluation. CTGAN trains only on the remaining rows.",
            )

            if st.button("Apply preprocessing", key="btn_preprocess", type="primary"):
                with st.spinner("Cleaning and splitting…"):
                    df_clean, num_cols, cat_cols = preprocess_dataframe(df)
                    df_train, df_test = train_test_split(
                        df_clean, test_size=test_size / 100, random_state=42, shuffle=True)
                    meta = build_metadata(df_train)
                    st.session_state.df       = df_clean
                    st.session_state.df_train = df_train
                    st.session_state.df_test  = df_test
                    st.session_state.metadata = meta
                    st.session_state.num_cols = num_cols
                    st.session_state.cat_cols = cat_cols
                    # Reset downstream
                    for k in ["synthesizer", "synthetic_df", "quality_score",
                               "ml_results", "privacy_results", "quality_details"]:
                        st.session_state[k] = None

                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Train rows", f"{len(df_train):,}")
                pm2.metric("Test rows",  f"{len(df_test):,}")
                pm3.metric("Columns",    df_clean.shape[1])

                callout(
                    "ok", "&#10003;",
                    "Preprocessing complete. Proceed to <strong>Train</strong>."
                )

                with st.expander("SDV metadata (auto-detected column types)"):
                    st.json(meta.to_dict(), expanded=False)

            # Show current split status if already done
            elif st.session_state.df_train is not None:
                pm1, pm2, pm3 = st.columns(3)
                pm1.metric("Train rows", f"{len(st.session_state.df_train):,}")
                pm2.metric("Test rows",  f"{len(st.session_state.df_test):,}")
                pm3.metric("Columns",    st.session_state.df.shape[1])
                callout(
                    "ok", "&#10003;",
                    "Dataset is preprocessed and ready for training."
                )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — TRAIN CTGAN
# ══════════════════════════════════════════════════════════════════════════════
elif page_idx == 2:
    render_stepper(2)

    st.markdown("""
<div class="tl-section">
  <div class="tl-section-label">Stage 02</div>
  <div class="tl-section-title">Train CTGAN</div>
  <div class="tl-section-desc">
    Configure the Conditional Tabular GAN and start training.
    The model learns to generate realistic synthetic rows from your dataset.
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.df_train is None:
        callout("warn", "&#9888;",
                "Complete the <strong>Dataset</strong> step (preprocess) before training.")
        st.stop()

    # ── Architecture explainer ────────────────────────────────────────────────
    with st.expander("How CTGAN works", expanded=False):
        st.markdown("""
<div class="tl-ctgan-explainer">
  <div class="tl-ctgan-intro">
    CTGAN <strong>(Conditional Tabular GAN)</strong> adapts generative adversarial networks
    to structured data, learning realistic combinations of numerical and categorical values.
  </div>
  <div class="tl-ctgan-grid">
    <div class="tl-ctgan-card">
      <div class="tl-ctgan-card-title">Generator</div>
      <div class="tl-ctgan-card-desc">Creates synthetic rows from random noise and the selected category condition.</div>
    </div>
    <div class="tl-ctgan-card">
      <div class="tl-ctgan-card-title">Discriminator</div>
      <div class="tl-ctgan-card-desc">Learns to distinguish real training rows from generated rows.</div>
    </div>
    <div class="tl-ctgan-card">
      <div class="tl-ctgan-card-title">Conditional sampling</div>
      <div class="tl-ctgan-card-desc">Focuses training on categories so less frequent values are represented.</div>
    </div>
    <div class="tl-ctgan-card">
      <div class="tl-ctgan-card-title">Mode-specific normalisation</div>
      <div class="tl-ctgan-card-desc">Models multi-modal numerical columns without flattening their distinct ranges.</div>
    </div>
  </div>
  <div class="tl-ctgan-note">
    <span class="tl-ctgan-note-title">Training dynamics</span>
    <span>The generator and discriminator improve together through adversarial training. More epochs can refine fidelity, with diminishing returns for typical tabular datasets. SDV 2.0 fits and samples the table through its single-table synthesizer API.</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Current dataset info ──────────────────────────────────────────────────
    df_train = st.session_state.df_train
    ti1, ti2, ti3, ti4 = st.columns(4)
    ti1.metric("Dataset",         st.session_state.dataset_name or "—")
    ti2.metric("Train rows",      f"{len(df_train):,}")
    ti3.metric("Numerical cols",  len(st.session_state.num_cols))
    ti4.metric("Categorical cols", len(st.session_state.cat_cols))

    st.write("")

    # ── Training form ─────────────────────────────────────────────────────────
    with st.form("train_form"):
        section_label("Training configuration")
        fc1, fc2 = st.columns(2)
        epochs = fc1.number_input(
            "Epochs", min_value=5, max_value=2000,
            value=st.session_state.train_epochs, step=10,
            help="Number of full passes through the training data. More = better fidelity but slower.")
        batch_size = fc2.selectbox(
            "Batch size", [100, 200, 500, 1000],
            index=[100, 200, 500, 1000].index(st.session_state.train_batch),
            help="Rows per gradient update. Larger batches are more stable but use more memory.")

        st.write("")
        callout(
            "warn", "&#9888;",
            "<div style=\"font-size:12px\">"
            "<strong>Streamlit Community Cloud free tier:</strong> keep epochs &#8804; 100 to avoid timeout. "
            "Locally, 200&#8211;500 epochs typically give better quality. "
            "Training cannot be paused &#8212; refresh the page to abort."
            "</div>"
        )
        submitted = st.form_submit_button("Start training →", type="primary")

    if submitted:
        st.session_state.train_epochs = epochs
        st.session_state.train_batch  = batch_size
        metadata = st.session_state.metadata

        status_area = st.empty()
        status_area.markdown(
            f'<div class="tl-callout tl-callout-info">'
            f'<div class="tl-callout-icon">&#9697;</div>'
            f'<div>Training CTGAN &#8212; {epochs} epochs &#215; {len(df_train):,} rows'
            f' &#215; {df_train.shape[1]} columns. '
            f'This may take several minutes. Do not navigate away.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        prog = st.progress(0, text="Initialising…")

        try:
            with st.spinner(f"Training ({epochs} epochs)…"):
                synth = train_ctgan(df_train, metadata, epochs, batch_size)
                st.session_state.synthesizer = synth
                # Reset downstream
                for k in ["synthetic_df", "quality_score", "ml_results",
                           "privacy_results", "quality_details"]:
                    st.session_state[k] = None
            prog.progress(100, text="Done")
            status_area.markdown(
                '<div class="tl-callout tl-callout-ok">'
                '<div class="tl-callout-icon">&#10003;</div>'
                '<div><strong>Training complete.</strong> Proceed to <strong>Generate</strong>.</div>'
                '</div>',
                unsafe_allow_html=True,
            )
            st.balloons()
        except Exception as e:
            status_area.markdown(
                f'<div class="tl-callout tl-callout-warn">'
                f'<div class="tl-callout-icon">&#10007;</div>'
                f'<div><strong>Training failed:</strong> {e}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            with st.expander("Error details"):
                st.exception(e)

    elif st.session_state.synthesizer is not None:
        callout(
            "ok", "&#10003;",
            "CTGAN is trained and ready. You can retrain with different settings or proceed to "
            "<strong>Generate</strong>."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — GENERATE
# ══════════════════════════════════════════════════════════════════════════════
elif page_idx == 3:
    render_stepper(3)

    st.markdown("""
<div class="tl-section">
  <div class="tl-section-label">Stage 03</div>
  <div class="tl-section-title">Generate Synthetic Data</div>
  <div class="tl-section-desc">
    Draw new records from the trained CTGAN generator.
    These rows are entirely artificial &#8212; they do not correspond to any real individual.
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.synthesizer is None:
        callout("warn", "&#9888;", "Train CTGAN first (Stage 02).")
        st.stop()

    gc1, gc2 = st.columns([2, 3])
    with gc1:
        n_rows = st.number_input(
            "Synthetic rows to generate",
            min_value=10, max_value=100_000, value=500, step=50,
            help="More rows increase download size and evaluation time.")
        st.markdown(
            '<div class="tl-upload-hint">Larger counts may take longer to generate.</div>',
            unsafe_allow_html=True,
        )
        if st.button("Generate →", key="btn_generate", type="primary", use_container_width=True):
            with st.spinner(f"Generating {n_rows:,} rows…"):
                syn_df = generate_synthetic(st.session_state.synthesizer, n_rows)
                st.session_state.synthetic_df = syn_df
                # Reset evaluation results since new data was generated
                for k in ["quality_score", "ml_results", "privacy_results", "quality_details"]:
                    st.session_state[k] = None

    if st.session_state.synthetic_df is not None:
        syn_df = st.session_state.synthetic_df
        with gc2:
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Rows generated", f"{len(syn_df):,}")
            sm2.metric("Columns",        syn_df.shape[1])
            sm3.metric("Null values",    int(syn_df.isna().sum().sum()))

        st.write("")
        callout(
            "info", "&#9432;",
            "<div style=\"font-size:12px\">"
            "<strong>These rows are synthetic.</strong> They were sampled from the generator network &#8212; "
            "not retrieved from any real record. They are labelled here for transparency."
            "</div>"
        )

        section_label("Preview &#8212; first 20 rows")
        st.dataframe(syn_df.head(20), use_container_width=True)

        csv_bytes = syn_df.to_csv(index=False).encode("utf-8")
        dl1, dl2 = st.columns([1, 3])
        with dl1:
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name="synthetic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 — EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
elif page_idx == 4:
    render_stepper(4)

    st.markdown("""
<div class="tl-section">
  <div class="tl-section-label">Stage 04</div>
  <div class="tl-section-title">Quality, Utility &amp; Privacy</div>
  <div class="tl-section-desc">
    Three independent dimensions of evaluation. High statistical quality does not
    guarantee ML utility, and neither guarantees privacy.
  </div>
</div>
""", unsafe_allow_html=True)

    if st.session_state.synthetic_df is None or st.session_state.df_train is None:
        callout("warn", "&#9888;", "Generate synthetic data first (Stage 03).")
        st.stop()

    real_df  = st.session_state.df_train
    syn_df   = st.session_state.synthetic_df
    num_cols = st.session_state.num_cols
    cat_cols = st.session_state.cat_cols
    metadata = st.session_state.metadata

    eval_tab, dist_tab, util_tab, priv_tab = st.tabs([
        "Quality Score", "Distributions", "ML Utility (TSTR)", "Privacy Risk"
    ])

    # ── TAB 1: Quality Score ──────────────────────────────────────────────────
    with eval_tab:
        st.write("")
        if st.button("Run SDV quality report", key="btn_quality", type="primary"):
            with st.spinner("Computing quality metrics…"):
                score, details = run_quality_report(real_df, syn_df, metadata)
                st.session_state.quality_score   = score
                st.session_state.quality_details = details

        if st.session_state.quality_score is not None:
            score   = st.session_state.quality_score
            details = st.session_state.quality_details
            pct     = round(score * 100, 1)

            qc1, qc2, qc3 = st.columns([1, 1, 2])
            with qc1:
                color = "#4ADE80" if pct >= 70 else ("#FFB347" if pct >= 50 else "#FF5C5C")
                st.markdown(
                    f'<div class="tl-metric">'
                    f'<div class="tl-metric-label">Overall quality</div>'
                    f'<div class="tl-metric-value" style="color:{color}">{pct}%</div>'
                    f'<div class="tl-metric-sub">SDV Column Shapes + Pair Trends</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with qc2:
                grade = "Good" if pct >= 70 else ("Moderate" if pct >= 50 else "Low")
                st.markdown(
                    f'<div class="tl-metric">'
                    f'<div class="tl-metric-label">Rating</div>'
                    f'<div class="tl-metric-value">{grade}</div>'
                    f'<div class="tl-metric-sub">&#8805;70% is generally good</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with qc3:
                callout(
                    "subtle", "&#9432;",
                    "<div style=\"font-size:12px\">"
                    "<strong>Column Shapes</strong> measures how closely each column&#39;s distribution "
                    "is reproduced. <strong>Column Pair Trends</strong> measures pairwise correlation "
                    "preservation. The overall score is the average of both properties across all columns. "
                    "A score above 70% indicates good statistical fidelity for most use cases."
                    "</div>"
                )

            if isinstance(details, pd.DataFrame) and not details.empty:
                st.write("")
                section_label("Per-column detail")
                st.dataframe(details, use_container_width=True)

                if "Score" in details.columns and "Column" in details.columns:
                    col_scores = details[["Column", "Score"]].dropna().sort_values("Score")
                    fig, ax = plt.subplots(figsize=(10, max(3, len(col_scores) * 0.35)))
                    colors = ["#4ADE80" if v >= 0.7 else ("#FFB347" if v >= 0.5 else "#FF5C5C")
                              for v in col_scores["Score"]]
                    ax.barh(col_scores["Column"], col_scores["Score"] * 100, color=colors)
                    ax.axvline(70, color="#C7FF55", linestyle="--", linewidth=1, alpha=0.5,
                               label="70% threshold")
                    ax.set_xlabel("Score (%)")
                    ax.set_title("Column Shape Scores", pad=8)
                    ax.legend(fontsize=8)
                    apply_chart_style(fig, ax)
                    plt.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close()
        else:
            callout("subtle", "&#9675;",
                    "Click <strong>Run SDV quality report</strong> to compute scores.")

    # ── TAB 2: Distributions ──────────────────────────────────────────────────
    with dist_tab:
        st.write("")

        all_viz_cols = [c for c in (num_cols + cat_cols) if c in syn_df.columns]
        if not all_viz_cols:
            st.info("No common columns between real and synthetic data.")
        else:
            selected_col = st.selectbox(
                "Select column to compare",
                options=all_viz_cols,
                help="Choose any column to see its real vs synthetic distribution.",
            )

            is_numerical = selected_col in num_cols
            rc_left, rc_right = st.columns(2)

            if is_numerical:
                rd = real_df[selected_col].dropna()
                sd = syn_df[selected_col].dropna()
                with rc_left:
                    compare_label("real", "Real")
                    st.markdown(f"**{selected_col}** — numerical")
                    st.markdown(
                        f"mean `{rd.mean():.3f}` · std `{rd.std():.3f}` · "
                        f"min `{rd.min():.2f}` · max `{rd.max():.2f}`"
                    )
                with rc_right:
                    compare_label("syn", "Synthetic")
                    st.markdown(f"**{selected_col}** — numerical")
                    st.markdown(
                        f"mean `{sd.mean():.3f}` · std `{sd.std():.3f}` · "
                        f"min `{sd.min():.2f}` · max `{sd.max():.2f}`"
                    )

                fig, ax = plt.subplots(figsize=(11, 4))
                rd.plot.hist(ax=ax, bins=30, alpha=0.65, color="#4f8ef7", label="Real",
                             edgecolor="#111313")
                sd.plot.hist(ax=ax, bins=30, alpha=0.65, color="#C7FF55", label="Synthetic",
                             edgecolor="#111313")
                ax.set_title(f"{selected_col} — Real vs Synthetic", pad=8)
                ax.set_xlabel(selected_col)
                ax.set_ylabel("Count")
                ax.legend(fontsize=9)
                apply_chart_style(fig, ax)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            else:
                real_vc = real_df[selected_col].value_counts(normalize=True).head(10)
                syn_vc  = syn_df[selected_col].value_counts(normalize=True).head(10)
                compare = pd.DataFrame({"Real": real_vc, "Synthetic": syn_vc}).fillna(0)

                with rc_left:
                    compare_label("real", "Real")
                    st.markdown(
                        f"**{selected_col}** — {real_df[selected_col].nunique()} unique values"
                    )
                with rc_right:
                    compare_label("syn", "Synthetic")
                    st.markdown(
                        f"**{selected_col}** — {syn_df[selected_col].nunique()} unique values"
                    )

                fig, ax = plt.subplots(figsize=(11, 4))
                x = np.arange(len(compare))
                w = 0.38
                ax.bar(x - w / 2, compare["Real"] * 100,      w, color="#4f8ef7", label="Real",      alpha=0.85)
                ax.bar(x + w / 2, compare["Synthetic"] * 100, w, color="#C7FF55", label="Synthetic", alpha=0.85)
                ax.set_xticks(x)
                ax.set_xticklabels(compare.index, rotation=30, ha="right", fontsize=9)
                ax.set_ylabel("Proportion (%)")
                ax.set_title(f"{selected_col} — Category Proportions", pad=8)
                ax.legend(fontsize=9)
                apply_chart_style(fig, ax)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                compare_pct = (compare * 100).round(2)
                compare_pct["Difference (pp)"] = (compare_pct["Synthetic"] - compare_pct["Real"]).round(2)
                section_label("Proportion table", "margin-top:16px")
                st.dataframe(
                    compare_pct.rename(columns={"Real": "Real %", "Synthetic": "Synthetic %"}),
                    use_container_width=True,
                )

            # ── Correlation comparison (numerical only) ───────────────────────
            common_num = [c for c in num_cols if c in syn_df.columns]
            if len(common_num) >= 2:
                divider()
                section_label("Correlation matrix comparison")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                kw = dict(cmap="RdYlGn", annot=True, fmt=".2f", center=0,
                          annot_kws={"size": 8}, cbar=False, linewidths=0.5,
                          linecolor="#111313")
                sns.heatmap(real_df[common_num].corr(), ax=ax1, **kw)
                ax1.set_title("Real", pad=8)
                sns.heatmap(syn_df[common_num].corr(), ax=ax2, **kw)
                ax2.set_title("Synthetic", pad=8)
                apply_chart_style(fig, [ax1, ax2])
                plt.suptitle("Pairwise Correlations", color="#F5F5F0", fontsize=12, y=1.02)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

                callout(
                    "subtle", "&#9432;",
                    "<div style=\"font-size:12px\">"
                    "Similar heatmaps indicate that CTGAN preserved inter-column relationships. "
                    "Large differences suggest the GAN did not fully capture correlated structure."
                    "</div>"
                )

    # ── TAB 3: ML Utility ─────────────────────────────────────────────────────
    with util_tab:
        st.write("")
        callout(
            "subtle", "&#9432;",
            "<div style=\"font-size:12px\">"
            "<strong>TSTR methodology:</strong> "
            "The real dataset was split into train and test partitions <em>before</em> CTGAN training. "
            "A Random Forest is trained on synthetic data and evaluated on the real test set (TSTR). "
            "A second Random Forest trained on real training data provides the baseline. "
            "Both are tested on the same real held-out set. The closer TSTR metrics are to the "
            "baseline, the more predictively useful the synthetic data is. "
            "Target leakage is prevented by excluding the target column from feature encoding."
            "</div>"
        )

        st.write("")
        all_cols = st.session_state.df_train.columns.tolist()
        target = st.selectbox(
            "Target column",
            options=all_cols,
            help="The column you want to predict. Choose a meaningful outcome variable.",
        )

        if st.button("Run TSTR evaluation", key="btn_tstr", type="primary"):
            real_test = st.session_state.df_test
            if real_test[target].nunique() < 2:
                st.error("Target column has fewer than 2 unique values in the test set. Choose another column.")
            else:
                with st.spinner("Training and evaluating models…"):
                    metrics, err = run_ml_utility(
                        st.session_state.df_train, real_test,
                        syn_df, target, num_cols, cat_cols)
                if err:
                    st.error(f"Evaluation failed: {err}")
                else:
                    st.session_state.ml_results = metrics

        if st.session_state.ml_results is not None:
            m = st.session_state.ml_results
            section_label(
                f"Results &#8212; {m['task']} &#183; target: {m['target']}",
                "margin-top:8px"
            )

            if m["task"] == "Classification":
                um1, um2, um3, um4 = st.columns(4)
                um1.metric("Baseline Accuracy", f"{m['baseline_accuracy']:.3f}",
                           help="Random Forest trained on real data")
                um2.metric("TSTR Accuracy", f"{m['tstr_accuracy']:.3f}",
                           delta=f"{m['tstr_accuracy'] - m['baseline_accuracy']:.3f}",
                           help="Random Forest trained on synthetic data")
                um3.metric("Baseline F1", f"{m['baseline_f1']:.3f}")
                um4.metric("TSTR F1", f"{m['tstr_f1']:.3f}",
                           delta=f"{m['tstr_f1'] - m['baseline_f1']:.3f}")

                fig, ax = plt.subplots(figsize=(9, 4))
                metrics_names = ["Accuracy", "F1", "Precision", "Recall"]
                base_v = [m["baseline_accuracy"], m["baseline_f1"],
                          m["baseline_precision"], m["baseline_recall"]]
                tstr_v = [m["tstr_accuracy"], m["tstr_f1"],
                          m["tstr_precision"], m["tstr_recall"]]
                x = np.arange(len(metrics_names))
                w = 0.35
                ax.bar(x - w / 2, base_v, w, color="#4f8ef7", label="Baseline (Real)", alpha=0.9)
                ax.bar(x + w / 2, tstr_v, w, color="#C7FF55", label="TSTR (Synthetic)", alpha=0.9)
                ax.set_xticks(x)
                ax.set_xticklabels(metrics_names)
                ax.set_ylim(0, 1.15)
                ax.set_ylabel("Score")
                ax.set_title("Baseline vs TSTR", pad=8)
                ax.legend(fontsize=9)
                apply_chart_style(fig, ax)
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close()

            else:
                rm1, rm2, rm3, rm4 = st.columns(4)
                rm1.metric("Baseline MAE",  f"{m['baseline_mae']:.3f}")
                rm2.metric("TSTR MAE",      f"{m['tstr_mae']:.3f}",
                           delta=f"{m['tstr_mae'] - m['baseline_mae']:.3f}")
                rm3.metric("Baseline RMSE", f"{m['baseline_rmse']:.3f}")
                rm4.metric("TSTR RMSE",     f"{m['tstr_rmse']:.3f}",
                           delta=f"{m['tstr_rmse'] - m['baseline_rmse']:.3f}")

            callout(
                "subtle", "&#9432;",
                "<div style=\"font-size:12px\">"
                "A small gap between TSTR and Baseline means the synthetic data preserves most "
                "predictive signal. A large gap indicates the GAN did not capture the relationships "
                "between features and the target. Increasing training epochs may help &#8212; but is not "
                "guaranteed to close the gap on all datasets."
                "</div>"
            )

        else:
            callout("subtle", "&#9675;",
                    "Select a target column and click <strong>Run TSTR evaluation</strong>.")

    # ── TAB 4: Privacy Risk ───────────────────────────────────────────────────
    with priv_tab:
        st.write("")
        callout(
            "warn", "&#9888;",
            "<div style=\"font-size:12px\">"
            "<strong>Important disclaimer.</strong> "
            "These checks are <em>heuristics only</em>. They cannot prove anonymity and "
            "do not constitute formal differential-privacy guarantees. CTGAN does not implement "
            "DP-SGD or any certified privacy mechanism. Low exact-match counts do not mean "
            "synthetic data is safe to share publicly. Always consult a privacy professional "
            "before releasing data derived from sensitive personal records."
            "</div>"
        )

        st.write("")
        if st.button("Run privacy checks", key="btn_privacy", type="primary"):
            with st.spinner("Computing privacy metrics…"):
                priv = compute_privacy_metrics(real_df, syn_df, num_cols, cat_cols)
                st.session_state.privacy_results = priv

        if st.session_state.privacy_results is not None:
            priv = st.session_state.privacy_results

            # Exact match
            section_label("Exact-match check", "margin-top:4px")
            exact  = priv["exact_matches"]
            rate   = priv["exact_match_rate"] * 100
            ec1, ec2, ec3 = st.columns(3)
            ec1.metric("Exact matches", exact)
            ec2.metric("Match rate",    f"{rate:.2f}%")
            risk_label = "High" if rate > 5 else ("Moderate" if rate > 1 else "Low")
            risk_color = "#FF5C5C" if rate > 5 else ("#FFB347" if rate > 1 else "#4ADE80")
            with ec3:
                st.markdown(
                    f'<div class="tl-metric">'
                    f'<div class="tl-metric-label">Risk indicator</div>'
                    f'<div class="tl-metric-value" style="color:{risk_color}">{risk_label}</div>'
                    f'<div class="tl-metric-sub">Exact-match heuristic</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Risk bar
            bar_pct = min(rate * 10, 100)
            st.markdown(
                f'<div class="tl-risk-bar-track">'
                f'<div class="tl-risk-bar-fill" style="width:{bar_pct:.1f}%;background:{risk_color}"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Nearest-neighbour
            section_label("Nearest-neighbour distance", "margin-top:20px")
            if priv.get("mean_nn_dist") is not None:
                nn_mean = priv["mean_nn_dist"]
                nn_min  = priv.get("min_nn_dist", 0.0)
                nc1, nc2 = st.columns(2)
                nc1.metric("Mean NN distance", f"{nn_mean:.4f}",
                           help="Average Euclidean distance in normalised feature space")
                nc2.metric("Min NN distance",  f"{nn_min:.4f}",
                           help="Closest synthetic row to any real training record")

                callout(
                    "subtle", "&#9432;",
                    "<div style=\"font-size:12px\">"
                    "Distance is measured in normalised feature space (each feature scaled 0&#8211;1 using "
                    "training-data range). A mean distance above 0.05 generally suggests synthetic rows "
                    "are not near-copies of real records. A minimum distance near zero warrants "
                    "closer investigation. These are indicators, not proofs."
                    "</div>"
                )

                if nn_min < 0.001:
                    callout(
                        "warn", "&#9888;",
                        "<div style=\"font-size:12px\">"
                        "Some synthetic rows are very close to real training records. "
                        "Review carefully before sharing."
                        "</div>"
                    )
                else:
                    callout(
                        "ok", "&#10003;",
                        "<div style=\"font-size:12px\">"
                        "Synthetic rows maintain reasonable distance from real training records."
                        "</div>"
                    )
            else:
                st.info(f"NN distance could not be computed: {priv.get('nn_error', 'Unknown')}")

            # Summary table
            section_label("Privacy summary", "margin-top:20px")
            nn_val = f"{priv['mean_nn_dist']:.4f}" if priv.get("mean_nn_dist") is not None else "N/A"
            summary_df = pd.DataFrame({
                "Check": ["Exact duplicates", "Match rate",
                          "Mean NN distance", "Formal DP", "Public-safe"],
                "Result": [
                    str(exact),
                    f"{rate:.2f}%",
                    nn_val,
                    "None (not implemented)",
                    "Unknown — requires professional audit",
                ],
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
        else:
            callout("subtle", "&#9675;",
                    "Click <strong>Run privacy checks</strong> to compute metrics.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 — REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif page_idx == 5:
    st.markdown("""
<div class="tl-section">
  <div class="tl-section-label">Summary</div>
  <div class="tl-section-title">Results Report</div>
  <div class="tl-section-desc">Pipeline status, key metrics, and downloads.</div>
</div>
""", unsafe_allow_html=True)

    # ── Pipeline status ───────────────────────────────────────────────────────
    section_label("Pipeline status")
    s = st.session_state
    steps = [
        ("Dataset loaded",           s.df is not None),
        ("Preprocessed & split",     s.df_train is not None),
        ("CTGAN trained",            s.synthesizer is not None),
        ("Synthetic data generated", s.synthetic_df is not None),
        ("Quality evaluated",        s.quality_score is not None),
    ]
    sc1, sc2 = st.columns(2)
    for i, (label, done) in enumerate(steps):
        col = sc1 if i % 2 == 0 else sc2
        badge_cls = "tl-badge-done" if done else "tl-badge-idle"
        badge_txt = "Complete" if done else "Pending"
        col.markdown(
            f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0;'
            f'border-bottom:1px solid #1E221E">'
            f'<span class="tl-badge {badge_cls}">{badge_txt}</span>'
            f'<span style="font-size:13px;color:#F5F5F0">{label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    done_count = sum(d for _, d in steps)
    if done_count < len(steps):
        callout(
            "info", "&#9432;",
            f"{done_count}/{len(steps)} steps complete. "
            "Complete the remaining steps to unlock all report metrics."
        )

    # ── Key metrics ───────────────────────────────────────────────────────────
    if s.df_train is not None:
        divider()
        section_label("Key metrics")
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Train rows",    f"{len(s.df_train):,}")
        km2.metric("Test rows",     f"{len(s.df_test):,}" if s.df_test is not None else "—")
        km3.metric("Synthetic rows", f"{len(s.synthetic_df):,}" if s.synthetic_df is not None else "—")
        km4.metric("Quality score", f"{round(s.quality_score * 100, 1)}%" if s.quality_score is not None else "—")

    # ── Downloads ─────────────────────────────────────────────────────────────
    divider()
    section_label("Downloads")

    # Pre-compute all conditionals so the f-string never crashes on None
    _real_rows  = f"{len(s.df):,}"           if s.df            is not None else "—"
    _train_rows = f"{len(s.df_train):,}"     if s.df_train      is not None else "—"
    _test_rows  = f"{len(s.df_test):,}"      if s.df_test       is not None else "—"
    _ncols      = str(s.df.shape[1])         if s.df            is not None else "—"
    _trained    = "Yes"                       if s.synthesizer   is not None else "No"
    _syn_rows   = f"{len(s.synthetic_df):,}" if s.synthetic_df  is not None else "—"
    _q_score    = f"{round(s.quality_score * 100, 1)}%" if s.quality_score is not None else "Not computed"
    _ml         = s.ml_results or {}
    _priv       = s.privacy_results or {}
    _match_rate = (f"{_priv.get('exact_match_rate', 0) * 100:.2f}%"
                   if _priv.get("exact_match_rate") is not None else "—")
    _nn_dist    = (f"{_priv.get('mean_nn_dist'):.4f}"
                   if _priv.get("mean_nn_dist") is not None else "—")

    report_txt = f"""TabuLab — Synthetic Data Generation Report
Generate. Validate. Trust.
==========================================
Dataset         : {s.dataset_name or 'Unknown'}
Real rows       : {_real_rows} (train: {_train_rows} | test: {_test_rows})
Columns         : {_ncols}
Numerical cols  : {', '.join(s.num_cols) or '—'}
Categorical cols: {', '.join(s.cat_cols) or '—'}

CTGAN Training
--------------
Epochs     : {s.train_epochs}
Batch size : {s.train_batch}
Trained    : {_trained}

Synthetic Data
--------------
Rows generated : {_syn_rows}

Quality (SDV)
--------------
Overall score  : {_q_score}

ML Utility (TSTR)
-----------------
Task           : {_ml.get('task', '—')}
Target         : {_ml.get('target', '—')}
Baseline Acc   : {_ml.get('baseline_accuracy', '—')}
TSTR Accuracy  : {_ml.get('tstr_accuracy', '—')}

Privacy Checks
--------------
Exact matches  : {_priv.get('exact_matches', '—')}
Match rate     : {_match_rate}
Mean NN dist   : {_nn_dist}
Formal DP      : None implemented

Disclaimer
----------
Privacy checks in this report are heuristics only.
They do not prove anonymity or provide formal DP guarantees.
Do not publish synthetic data derived from sensitive records
without a professional privacy audit.

Generated by TabuLab · PS 65 · SDV 2.0.0 · CTGAN 0.12
"""

    dl1, dl2 = st.columns(2)
    if s.synthetic_df is not None:
        with dl1:
            st.download_button(
                "Download synthetic_data.csv",
                data=s.synthetic_df.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_data.csv",
                mime="text/csv",
                use_container_width=True,
            )
    with dl2:
        st.download_button(
            "Download report.txt",
            data=report_txt.encode("utf-8"),
            file_name="tabulab_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Limitations ───────────────────────────────────────────────────────────
    divider()
    section_label("Limitations & responsible use")
    st.markdown("""
<div class="tl-card" style="margin-top:12px">
<ul style="font-size:13px;color:#A0A49F;line-height:2;margin:0;padding-left:20px">
  <li><strong style="color:#F5F5F0">Training time</strong> &#8212; CTGAN is a neural network.
      100 epochs on a small dataset typically takes 30&#8211;120 seconds.</li>
  <li><strong style="color:#F5F5F0">Dataset size</strong> &#8212; Best results with &#8805; 500 rows and
      a mix of numerical and categorical columns.</li>
  <li><strong style="color:#F5F5F0">Rare categories</strong> &#8212; Values appearing very infrequently
      in the training data may not appear in synthetic data.</li>
  <li><strong style="color:#F5F5F0">No time-series support</strong> &#8212; CTGAN is designed for
      independent, identically distributed tabular data only.</li>
  <li><strong style="color:#F5F5F0">Privacy</strong> &#8212; Synthetic data is not inherently private.
      No formal differential-privacy guarantees are provided.</li>
  <li><strong style="color:#F5F5F0">Cloud limits</strong> &#8212; On Streamlit Community Cloud free tier,
      keep epochs &#8804; 100 and row counts &#8804; 50,000.</li>
</ul>
</div>
""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1E221E;margin-top:64px;padding:28px 0;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
  <div style="display:flex;align-items:center;gap:8px">
    <div style="width:18px;height:18px;background:#C7FF55;
      clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%)"></div>
    <span style="font-size:14px;font-weight:700;color:#F5F5F0;letter-spacing:-0.03em">TabuLab</span>
  </div>
  <span style="font-size:11px;color:#3A3E39;font-family:'JetBrains Mono',monospace">
    PS 65 &nbsp;&#183;&nbsp; SDV 2.0.0 &nbsp;&#183;&nbsp; CTGAN 0.12.1 &nbsp;&#183;&nbsp; Streamlit
  </span>
</div>
""", unsafe_allow_html=True)

# Close tl-page wrapper
st.markdown("</div>", unsafe_allow_html=True)
