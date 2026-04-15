import sys
from pathlib import Path

# Ensure dashboard dir is on path so components can import data_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st
from data_loader import load_predictions, load_signals, get_available_dates
from components.kpi_cards import render_kpi_cards
from components.signals_table import render_signals_table
from components.zscore_chart import render_zscore_chart
from components.performance_chart import render_performance_chart
from components.daily_panel import render_daily_panel

st.set_page_config(
    page_title="BANKNIFTY Mispricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; letter-spacing: 0.05em; }
[data-testid="stSidebar"] { background-color: #0f1117; }
.block-container { padding-top: 1.5rem; }
[data-testid="stDataFrame"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 BANKNIFTY\nMispricing")
    st.divider()

    model_choice = st.radio(
        "Model",
        ["Walk-Forward ✓", "Static"],
        index=0,
        help="Walk-Forward is recommended. Falls back to Static if WF files are not present.",
    )
    model_key = "wf" if "Walk" in model_choice else "static"

    df_pred    = load_predictions(model_key)
    df_signals = load_signals(model_key)

    if df_signals.empty:
        st.error("No signal data found. Run the pipeline first.")
        st.stop()

    available_dates = get_available_dates(df_signals)
    selected_date   = st.date_input(
        "Trading Date",
        value=available_dates[0],
        min_value=available_dates[-1],
        max_value=available_dates[0],
    )
    st.caption(f"Data: {available_dates[-1]} → {available_dates[0]}")
    st.caption(f"{len(available_dates)} trading days loaded")

    st.divider()
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# ── Filter to selected date ───────────────────────────────────────────────────
day_signals = df_signals[df_signals["date"].dt.date == selected_date]
day_pred    = df_pred[df_pred["date"].dt.date == selected_date] if not df_pred.empty else df_pred

# ── Header ────────────────────────────────────────────────────────────────────
st.title("BANKNIFTY Option Mispricing Dashboard")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_hist, tab_daily = st.tabs(["📈 Historical Analysis", "📅 Today's Run"])

with tab_hist:
    if day_signals.empty:
        st.warning(f"No signals found for {selected_date}. Select another date.")
        st.stop()

    st.caption(f"Showing signals for **{selected_date}** · Model: **{model_choice}**")

    # Panel 1: KPI Cards
    render_kpi_cards(day_signals)
    st.divider()

    # Panels 2 + 3: Table | Histogram
    col_left, col_right = st.columns([6, 4])
    with col_left:
        render_signals_table(day_signals)
    with col_right:
        render_zscore_chart(day_pred, model_key)

    st.divider()

    # Panel 4: Model R² Trend
    render_performance_chart(model_key)

with tab_daily:
    render_daily_panel(model_key)
