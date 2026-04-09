import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from data_loader import (
    load_daily_predictions,
    load_daily_signals,
    get_daily_available_dates,
)


def _color_signal(val: str) -> str:
    if val == "BUY":
        return "background-color: #0d3b1e; color: #00e676"
    if val == "SELL":
        return "background-color: #3b0d0d; color: #ff5252"
    return ""


def render_daily_panel(model_key: str):
    """Panel 5 — deep-dive on today's daily_run.py output."""

    available = get_daily_available_dates()

    if not available:
        st.info(
            "No daily results yet. "
            "Run `python daily_run.py --auto` after market close to populate this panel."
        )
        return

    # Date selector (separate from the historical sidebar selector)
    selected = st.selectbox(
        "Daily run date",
        options=available,
        index=0,
        help="Dates where daily_run.py has been executed.",
    )

    day_pred    = load_daily_predictions(selected)
    day_signals = load_daily_signals(selected)

    # ── Sub-section 1: Day Summary Banner ────────────────────────────────────
    buy_n  = (day_signals["signal"] == "BUY").sum()  if not day_signals.empty else 0
    sell_n = (day_signals["signal"] == "SELL").sum() if not day_signals.empty else 0
    hold_n = (day_signals["signal"] == "HOLD").sum() if not day_signals.empty else 0
    liquid = len(day_pred) if not day_pred.empty else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Liquid Contracts", liquid)
    c2.metric("BUY (Underpriced)", buy_n)
    c3.metric("SELL (Overpriced)", sell_n)
    c4.metric("HOLD", hold_n)

    if day_signals.empty and day_pred.empty:
        st.warning("No signal data for this date (likely expiry-day data — DTE=0).")
        if not day_pred.empty:
            st.caption(f"{len(day_pred)} predictions computed but no liquid contracts passed DTE>5 filter.")
        return

    # Only actionable signals below
    if day_signals.empty:
        st.info("No actionable signals (BUY/SELL) for this date.")
        return

    st.divider()

    # ── Sub-section 2: Full Signals Table ────────────────────────────────────
    st.subheader("All Signals")

    display_cols = ["strike", "expiry", "option_type", "close",
                    "predicted_price", "mispricing", "z_score", "signal"]
    available_cols = [c for c in display_cols if c in day_signals.columns]

    # Rename close → close_price for display
    df_table = day_signals[available_cols].copy()
    if "close" in df_table.columns:
        df_table = df_table.rename(columns={"close": "close_price"})

    if "z_score" in df_table.columns:
        df_table = df_table.reindex(
            df_table["z_score"].abs().sort_values(ascending=False).index
        )
    for col in ["z_score", "close_price", "predicted_price", "mispricing"]:
        if col in df_table.columns:
            df_table[col] = df_table[col].round(2)
    if "strike" in df_table.columns:
        df_table["strike"] = df_table["strike"].astype(int)
    if "expiry" in df_table.columns:
        df_table["expiry"] = pd.to_datetime(df_table["expiry"]).dt.strftime("%d %b %Y")

    if "signal" in df_table.columns:
        styled = df_table.style.map(_color_signal, subset=["signal"])
    else:
        styled = df_table.style

    st.dataframe(styled, use_container_width=True, hide_index=True)

    if day_pred.empty:
        return

    st.divider()

    # ── Sub-sections 3 & 4 side by side ──────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Sub-section 3: Volatility Smile
    with col_left:
        st.subheader("Volatility Smile")
        iv_col   = "IV" if "IV" in day_pred.columns else None
        mon_col  = "moneyness" if "moneyness" in day_pred.columns else None
        type_col = "option_type" if "option_type" in day_pred.columns else None

        if iv_col and mon_col and type_col:
            fig = go.Figure()
            for opt_type, color in [("CE", "#4a9eff"), ("PE", "#ff9e4a")]:
                subset = (
                    day_pred[day_pred[type_col] == opt_type]
                    .sort_values(mon_col)
                    .dropna(subset=[mon_col, iv_col])
                )
                if not subset.empty:
                    fig.add_trace(go.Scatter(
                        x=subset[mon_col],
                        y=subset[iv_col],
                        mode="lines+markers",
                        name=opt_type,
                        line=dict(color=color),
                        marker=dict(size=4),
                    ))
            fig.add_vline(x=1.0, line_dash="dot", line_color="#888",
                          annotation_text="ATM", annotation_position="top")
            fig.update_layout(
                xaxis_title="Moneyness",
                yaxis_title="IV",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cccccc",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("IV or moneyness data not available for smile chart.")

    # Sub-section 4: Mispricing Scatter
    with col_right:
        st.subheader("Mispricing vs Moneyness")
        mis_col = "mispricing" if "mispricing" in day_pred.columns else None
        z_col   = "z_score"    if "z_score"    in day_pred.columns else None
        oi_col  = "OI_normalized" if "OI_normalized" in day_pred.columns else None

        if mis_col and mon_col:
            plot_df = day_pred.dropna(subset=[mon_col, mis_col]).copy()
            # Clip mispricing for display only
            plot_df["_mis_disp"] = plot_df[mis_col].clip(-500, 500)

            marker_kwargs = dict(
                color="#4a9eff",
                size=6,
            )
            if z_col and z_col in plot_df.columns:
                marker_kwargs = dict(
                    color=plot_df[z_col].clip(-3, 3),
                    colorscale="RdYlGn_r",
                    cmin=-3, cmax=3,
                    colorbar=dict(title="Z-Score"),
                    size=(plot_df[oi_col].clip(1, 8) * 2).tolist()
                        if oi_col and oi_col in plot_df.columns
                        else 6,
                )

            fig2 = go.Figure(go.Scatter(
                x=plot_df[mon_col],
                y=plot_df["_mis_disp"],
                mode="markers",
                marker=marker_kwargs,
            ))
            fig2.add_hline(y=0, line_color="#555")
            fig2.update_layout(
                xaxis_title="Moneyness",
                yaxis_title="Mispricing",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cccccc",
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Mispricing data not available for scatter chart.")

    st.divider()

    # ── Sub-section 5: DTE Breakdown ─────────────────────────────────────────
    st.subheader("Signal DTE Breakdown")

    if "DTE" in day_signals.columns and "signal" in day_signals.columns:
        bins   = [0, 7, 14, 30, 60, 90, 999]
        labels = ["1–7", "8–14", "15–30", "31–60", "61–90", "90+"]
        dte_df = day_signals.copy()
        dte_df["dte_bucket"] = pd.cut(
            dte_df["DTE"], bins=bins, labels=labels, right=True
        )
        breakdown = (
            dte_df.groupby(["dte_bucket", "signal"], observed=True)
            .size()
            .unstack(fill_value=0)
        )

        fig3 = go.Figure()
        for sig, color in [("BUY", "#00e676"), ("SELL", "#ff5252")]:
            if sig in breakdown.columns:
                fig3.add_trace(go.Bar(
                    x=breakdown.index.astype(str),
                    y=breakdown[sig],
                    name=sig,
                    marker_color=color,
                ))
        fig3.update_layout(
            barmode="group",
            xaxis_title="DTE Bucket",
            yaxis_title="Signal Count",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#cccccc",
            margin=dict(l=20, r=20, t=20, b=20),
            legend=dict(orientation="h", y=1.1),
        )
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("DTE data not available for breakdown chart.")
