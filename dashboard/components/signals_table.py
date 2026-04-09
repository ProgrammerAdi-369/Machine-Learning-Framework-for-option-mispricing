import streamlit as st
import pandas as pd


def render_signals_table(day_signals):
    st.subheader("Top Signals")

    display_cols = ["strike", "expiry", "option_type", "close_price",
                    "predicted_price", "z_score", "signal"]

    available = [c for c in display_cols if c in day_signals.columns]
    df = day_signals[available].copy()

    # Sort by absolute z-score, top 10
    if "z_score" in df.columns:
        df = df.reindex(df["z_score"].abs().sort_values(ascending=False).index).head(10)

    # Format columns
    if "z_score" in df.columns:
        df["z_score"] = df["z_score"].round(2)
    if "close_price" in df.columns:
        df["close_price"] = df["close_price"].round(2)
    if "predicted_price" in df.columns:
        df["predicted_price"] = df["predicted_price"].round(2)
    if "strike" in df.columns:
        df["strike"] = df["strike"].astype(int)
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%d %b %Y")

    def color_signal(val):
        if val == "BUY":
            return "background-color: #0d3b1e; color: #00e676"
        if val == "SELL":
            return "background-color: #3b0d0d; color: #ff5252"
        return ""

    if "signal" in df.columns:
        styled = df.style.map(color_signal, subset=["signal"])
    else:
        styled = df.style

    st.dataframe(styled, use_container_width=True, hide_index=True)
