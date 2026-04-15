import streamlit as st
import pandas as pd


def render_signals_table(day_signals):
    st.subheader("Top Signals")

    df = day_signals.copy()

    # Normalise column names so the table works for both static and walk-forward signals.
    # Static signals use 'predicted_price' and 'z_score'.
    # Walk-forward signals use 'wf_predicted_price' and 'wf_z_score'.
    if "wf_predicted_price" in df.columns and "predicted_price" not in df.columns:
        df = df.rename(columns={"wf_predicted_price": "predicted_price"})
    if "wf_z_score" in df.columns and "z_score" not in df.columns:
        df = df.rename(columns={"wf_z_score": "z_score"})

    display_cols = ["strike", "option_type", "close_price",
                    "predicted_price", "z_score", "DTE", "signal"]

    available = [c for c in display_cols if c in df.columns]
    df = df[available].copy()

    # Sort by absolute z-score descending, top 10
    if "z_score" in df.columns:
        df = df.reindex(df["z_score"].abs().sort_values(ascending=False).index).head(10)
    else:
        df = df.head(10)

    # Format columns
    for col, decimals in [("z_score", 2), ("close_price", 2), ("predicted_price", 2)]:
        if col in df.columns:
            df[col] = df[col].round(decimals)
    if "strike" in df.columns:
        df["strike"] = df["strike"].astype(int)
    if "DTE" in df.columns:
        df["DTE"] = df["DTE"].astype(int)

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
