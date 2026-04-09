import streamlit as st
import plotly.graph_objects as go


def render_zscore_chart(day_pred):
    st.subheader("Z-Score Distribution")

    if "z_score" not in day_pred.columns or len(day_pred) < 10:
        st.info("Too few contracts for distribution chart.")
        return

    # Clip for display only — do not modify actual data
    z = day_pred["z_score"].clip(-6, 6)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=z,
        nbinsx=40,
        marker_color="#4a9eff",
        opacity=0.75,
        name="Z-Score"
    ))

    for x_val, color, label in [
        (-2, "#00e676", "BUY threshold"),
        (2, "#ff5252", "SELL threshold"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    fig.update_layout(
        xaxis_title="Z-Score",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cccccc",
        bargap=0.05,
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
