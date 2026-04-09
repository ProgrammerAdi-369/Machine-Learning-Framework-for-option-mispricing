import streamlit as st


def render_kpi_cards(day_signals):
    buy_count  = (day_signals["signal"] == "BUY").sum()
    sell_count = (day_signals["signal"] == "SELL").sum()
    total      = len(day_signals)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", total)
    col2.metric("BUY  (Underpriced)", buy_count)
    col3.metric("SELL  (Overpriced)", sell_count)
