import streamlit as st
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_loader import load_monthly_r2


def render_performance_chart(model_key):
    st.subheader("Model Performance (Monthly R²)")

    r2_df = load_monthly_r2(model_key)

    if r2_df.empty:
        st.info("R² data not available.")
        return

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r2_df["month"].astype(str),
        y=r2_df["r2"],
        mode="lines+markers",
        line=dict(color="#4a9eff", width=2),
        marker=dict(size=6),
        name="Monthly R²",
    ))

    fig.add_hline(
        y=0.90,
        line_dash="dot",
        line_color="#00e676",
        annotation_text="Good (0.90)",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=0.80,
        line_dash="dot",
        line_color="#ffaa00",
        annotation_text="Caution (0.80)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="R²",
        yaxis_range=[0.5, 1.0],
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cccccc",
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
