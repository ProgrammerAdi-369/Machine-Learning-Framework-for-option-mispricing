import streamlit as st
import plotly.graph_objects as go


def render_zscore_chart(day_pred, model_key="static"):
    st.subheader("Z-Score Distribution")

    # Select the correct z-score column based on the active model.
    # wf_predictions.parquet contains both 'z_score' (static) and 'wf_z_score' (walk-forward).
    # Always prefer the column that matches the selected model.
    if model_key == "wf" and "wf_z_score" in day_pred.columns:
        z_col = "wf_z_score"
    elif "z_score" in day_pred.columns:
        z_col = "z_score"
    elif "wf_z_score" in day_pred.columns:
        z_col = "wf_z_score"
    else:
        st.info("Z-score data not available for this date.")
        return

    if day_pred.empty:
        st.info("No prediction data loaded for this date.")
        return

    z_data = day_pred[z_col].dropna()
    if len(z_data) < 10:
        st.info("Too few contracts for distribution chart.")
        return

    # Clip for display only — does not modify the underlying data
    z = z_data.clip(-6, 6)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=z,
        nbinsx=40,
        marker_color="#4a9eff",
        opacity=0.75,
        name="Z-Score",
    ))

    for x_val, color, label in [
        (-2, "#00e676", "BUY  −2"),
        ( 2, "#ff5252", "SELL +2"),
    ]:
        fig.add_vline(
            x=x_val,
            line_dash="dash",
            line_color=color,
            annotation_text=label,
            annotation_position="top",
        )

    z_std  = z_data.std()
    z_mean = z_data.mean()
    subtitle = (
        f"{'Walk-Forward' if z_col == 'wf_z_score' else 'Static'} model  ·  "
        f"n={len(z_data):,}  mean={z_mean:.2f}  std={z_std:.2f}"
    )
    fig.update_layout(
        title=dict(text=subtitle, font=dict(size=11, color="#888888")),
        xaxis_title="Z-Score",
        yaxis_title="Count",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cccccc",
        bargap=0.05,
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)
