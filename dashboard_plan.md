# BANKNIFTY Option Mispricing Dashboard — Implementation Guide
### A complete step-by-step plan for Claude Code

---

## Overview

Build a Streamlit dashboard for personal daily use. It reads from existing pipeline output files (parquet/CSV), never re-runs the pipeline itself. The dashboard shows 5 panels: KPI cards, top signals table, z-score distribution, model R² trend, and a sidebar with date + model selector.

**Stack:** Python · Streamlit · Pandas · Plotly · Joblib (read-only)

---

## File & Data Assumptions (Confirm Before Starting)

The dashboard reads these files produced by the existing pipeline. Verify all paths exist before Step 1:

| File | Used For |
|---|---|
| `outputs/full_predictions.parquet` | Static model signals |
| `outputs/wf_predictions.parquet` | Walk-forward model signals |
| `outputs/trading_signals.csv` | Static BUY/SELL signals |
| `outputs/wf_trading_signals.csv` | Walk-forward BUY/SELL signals |

**Required columns in predictions parquet:**
`date, strike, expiry, option_type, close_price, predicted_price, mispricing, z_score, OI_normalized, Volume_normalized, DTE`

**Required columns in trading_signals CSV:**
`date, strike, expiry, option_type, close_price, predicted_price, z_score, signal`

> If any file is missing, each step below notes how to handle it gracefully.

---

## Project Structure

```
dashboard/
├── app.py                  ← main Streamlit entry point
├── data_loader.py          ← all file loading + caching logic
├── components/
│   ├── kpi_cards.py        ← Panel 1: hero KPI row
│   ├── signals_table.py    ← Panel 2: top signals table
│   ├── zscore_chart.py     ← Panel 3: z-score histogram
│   └── performance_chart.py← Panel 4: monthly R² line chart
├── utils.py                ← shared helpers (formatting, color maps)
└── requirements.txt
```

---

## Step 1 — Project Scaffold & Dependencies

### What to do
1. Create the folder structure above.
2. Create `requirements.txt`:
```
streamlit>=1.32.0
pandas>=2.0.0
plotly>=5.18.0
pyarrow>=14.0.0
numpy>=1.26.0
```
3. Create a minimal `app.py` that just runs `st.title("BANKNIFTY Mispricing Dashboard")` and confirm it boots with `streamlit run app.py`.

### Potential Problems & Solutions

**Problem:** Streamlit version conflicts with existing project environment.
**Solution:** Always create a separate virtual environment for the dashboard. `python -m venv dashboard_env`. Do NOT install into the pipeline's environment.

**Problem:** pyarrow not installed, parquet files won't load.
**Solution:** `pip install pyarrow` is mandatory. Add it explicitly to requirements.txt. FastParquet is NOT a substitute here.

---

## Step 2 — Data Loader (`data_loader.py`)

### What to do
Build a single module that loads and caches all data. Every other component imports from here — no component should directly read files.

```python
import streamlit as st
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent  # adjust to your repo root

@st.cache_data
def load_predictions(model: str = "wf") -> pd.DataFrame:
    """
    model: "static" -> full_predictions.parquet
           "wf"     -> wf_predictions.parquet
    """
    fname = "wf_predictions.parquet" if model == "wf" else "full_predictions.parquet"
    path = BASE / "outputs" / fname
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

@st.cache_data
def load_signals(model: str = "wf") -> pd.DataFrame:
    fname = "wf_trading_signals.csv" if model == "wf" else "trading_signals.csv"
    path = BASE / "outputs" / fname
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df

def get_available_dates(df: pd.DataFrame) -> list:
    return sorted(df["date"].dt.date.unique().tolist(), reverse=True)
```

### Key design decisions
- Use `@st.cache_data` on every loader. Parquet files are large; without caching, every widget interaction reloads from disk.
- Return full dataframes from loaders; let components filter by date. This way the cache is shared across all components.
- Always normalize `date` column to `pd.Timestamp` immediately on load.

### Potential Problems & Solutions

**Problem:** `date` column in parquet is stored as string or integer (YYYYMMDD).
**Solution:** In the loader, after reading: `df["date"] = pd.to_datetime(df["date"], errors="coerce")`. Then drop NaT rows with `df.dropna(subset=["date"])`. Log how many rows were dropped to stdout so you notice if something is wrong.

**Problem:** Column names differ from what the dashboard expects (e.g., `Close` vs `close_price`).
**Solution:** Add a column normalization function in `data_loader.py`:
```python
COLUMN_MAP = {
    "Close": "close_price",
    "Predicted": "predicted_price",
    "ZScore": "z_score",
    # add mappings as needed
}
def normalize_columns(df):
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
```
Call this inside every loader before returning.

**Problem:** `wf_predictions.parquet` doesn't have an R² column (needed for Panel 4).
**Solution:** Compute it inside the loader on the fly:
```python
from sklearn.metrics import r2_score
df["month"] = df["date"].dt.to_period("M")
r2_by_month = df.groupby("month").apply(
    lambda g: r2_score(g["close_price"], g["predicted_price"])
).reset_index(columns=["month", "r2"])
```
Cache this separately with `@st.cache_data`.

**Problem:** File not found (pipeline hasn't been run yet or path is wrong).
**Solution:** Wrap all loaders in try/except and return an empty DataFrame with the correct schema. The UI components must handle empty DataFrames gracefully (show a warning, not a crash).

---

## Step 3 — Sidebar (`app.py`)

### What to do
The sidebar has two controls that drive the entire dashboard:

```python
with st.sidebar:
    st.title("⚙️ Controls")

    # Model selector
    model_choice = st.radio(
        "Model",
        options=["Walk-Forward (Recommended)", "Static"],
        index=0
    )
    model_key = "wf" if "Walk" in model_choice else "static"

    # Load data based on model choice
    df_pred = load_predictions(model_key)
    df_signals = load_signals(model_key)

    # Date selector — only dates that actually have signals
    available_dates = get_available_dates(df_signals)
    selected_date = st.date_input(
        "Date",
        value=available_dates[0],      # default to latest
        min_value=available_dates[-1],
        max_value=available_dates[0]
    )

    st.caption(f"{len(available_dates)} trading days loaded")
```

Then filter at the top of `app.py` before rendering any panel:
```python
day_signals = df_signals[df_signals["date"].dt.date == selected_date]
day_pred    = df_pred[df_pred["date"].dt.date == selected_date]
```

### Potential Problems & Solutions

**Problem:** User picks a date that has no signals (e.g., a holiday that's somehow in the date range).
**Solution:** After filtering, check `if day_signals.empty:` and show `st.warning("No signals for this date. Try another date.")` and `st.stop()` to halt rendering gracefully.

**Problem:** `st.date_input` returns a `datetime.date` object but your DataFrame dates are `pd.Timestamp`.
**Solution:** Always compare with `.dt.date == selected_date` (not `==`). The `.dt.date` accessor strips time, making the comparison reliable.

---

## Step 4 — Panel 1: KPI Cards (`components/kpi_cards.py`)

### What to do
Three metric cards in a single row at the top of the page.

```python
import streamlit as st

def render_kpi_cards(day_signals):
    buy_count  = (day_signals["signal"] == "BUY").sum()
    sell_count = (day_signals["signal"] == "SELL").sum()
    total      = len(day_signals)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Signals", total)
    col2.metric("🟢 BUY (Underpriced)", buy_count)
    col3.metric("🔴 SELL (Overpriced)", sell_count)
```

### Styling note
`st.metric` is plain by default. To improve look, add this to `app.py` or a `style.css` injected via `st.markdown`:
```css
[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.85rem; color: #888; }
```
Inject with: `st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)`

### Potential Problems & Solutions

**Problem:** `signal` column values are inconsistent (`"BUY"`, `"buy"`, `1`, `-1`).
**Solution:** In `data_loader.py`, after loading signals, normalize:
```python
df["signal"] = df["signal"].astype(str).str.upper().str.strip()
# If numeric: df["signal"] = df["signal"].map({1: "BUY", -1: "SELL"})
```

**Problem:** Zero signals on a date but cards show 0/0/0 confusingly.
**Solution:** This is handled by the `st.stop()` guard in Step 3 — you'll never reach the cards if day_signals is empty.

---

## Step 5 — Panel 2: Top Signals Table (`components/signals_table.py`)

### What to do
Show top 10 contracts ranked by absolute z-score, with color-coded signal column.

```python
import streamlit as st
import pandas as pd

def render_signals_table(day_signals):
    st.subheader("Top Signals")

    display_cols = ["strike", "expiry", "option_type", "close_price",
                    "predicted_price", "z_score", "signal"]

    # Keep only columns that exist
    available = [c for c in display_cols if c in day_signals.columns]
    df = day_signals[available].copy()

    # Rank by absolute z-score, top 10
    df = df.reindex(df["z_score"].abs().sort_values(ascending=False).index).head(10)

    # Round for display
    df["z_score"]        = df["z_score"].round(2)
    df["close_price"]    = df["close_price"].round(2)
    df["predicted_price"]= df["predicted_price"].round(2)

    # Color rows by signal
    def color_signal(val):
        if val == "BUY":  return "background-color: #0d3b1e; color: #00e676"
        if val == "SELL": return "background-color: #3b0d0d; color: #ff5252"
        return ""

    styled = df.style.applymap(color_signal, subset=["signal"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
```

### Potential Problems & Solutions

**Problem:** `expiry` column is a raw date object and looks ugly in the table.
**Solution:** `df["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%d %b %Y")`

**Problem:** `strike` shows as float (e.g., `48000.0`) because parquet stores it that way.
**Solution:** `df["strike"] = df["strike"].astype(int)`

**Problem:** `predicted_price` column doesn't exist in `trading_signals.csv` (it may only be in predictions parquet).
**Solution:** In `data_loader.py`, after loading both files, merge predicted_price in:
```python
pred_cols = df_pred[["date", "strike", "expiry", "option_type", "predicted_price"]].drop_duplicates()
df_signals = df_signals.merge(pred_cols, on=["date", "strike", "expiry", "option_type"], how="left")
```
Do this merge once in the loader, not in the component.

**Problem:** `st.dataframe` with `.style` can be slow for large tables.
**Solution:** You're already limiting to 10 rows, so this is not an issue in practice.

---

## Step 6 — Panel 3: Z-Score Distribution (`components/zscore_chart.py`)

### What to do
A histogram of all z-scores for the selected date with BUY/SELL threshold lines.

```python
import streamlit as st
import plotly.graph_objects as go

def render_zscore_chart(day_pred):
    st.subheader("Z-Score Distribution")

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=day_pred["z_score"],
        nbinsx=40,
        marker_color="#4a9eff",
        opacity=0.75,
        name="Z-Score"
    ))

    # Threshold lines
    for x_val, color, label in [(-2, "#00e676", "BUY threshold"), (2, "#ff5252", "SELL threshold")]:
        fig.add_vline(x=x_val, line_dash="dash", line_color=color,
                      annotation_text=label, annotation_position="top")

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
```

### Potential Problems & Solutions

**Problem:** `day_pred` has very few rows (< 20 contracts on a low-volume day), making the histogram useless.
**Solution:** Add a fallback: `if len(day_pred) < 10: st.info("Too few contracts for distribution chart."); return`

**Problem:** Z-scores have extreme outliers (±10+) that compress the histogram.
**Solution:** Clip for display only: `z = day_pred["z_score"].clip(-6, 6)`. Do NOT clip the actual data — only the chart input.

**Problem:** Chart background clashes with Streamlit's default white/light theme.
**Solution:** Use `plot_bgcolor="rgba(0,0,0,0)"` and `paper_bgcolor="rgba(0,0,0,0)"` as shown above. This makes it transparent and it adapts to both light and dark Streamlit themes.

---

## Step 7 — Panel 4: Model R² Trend (`components/performance_chart.py`)

### What to do
A line chart showing monthly R² across the full study period. This is always shown for the full period, not just the selected date — it gives context on whether the model is healthy right now.

```python
import streamlit as st
import plotly.graph_objects as go
from data_loader import load_monthly_r2

def render_performance_chart(model_key):
    st.subheader("Model Performance (Monthly R²)")

    r2_df = load_monthly_r2(model_key)   # returns DataFrame: month (str), r2 (float)

    if r2_df.empty:
        st.info("R² data not available.")
        return

    # Highlight current month
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=r2_df["month"].astype(str),
        y=r2_df["r2"],
        mode="lines+markers",
        line=dict(color="#4a9eff", width=2),
        marker=dict(size=6),
        name="Monthly R²"
    ))

    # Reference lines
    fig.add_hline(y=0.90, line_dash="dot", line_color="#00e676",
                  annotation_text="Good (0.90)", annotation_position="bottom right")
    fig.add_hline(y=0.80, line_dash="dot", line_color="#ffaa00",
                  annotation_text="Caution (0.80)", annotation_position="bottom right")

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
```

And add `load_monthly_r2` to `data_loader.py`:
```python
@st.cache_data
def load_monthly_r2(model: str = "wf") -> pd.DataFrame:
    df = load_predictions(model)
    if df.empty:
        return pd.DataFrame(columns=["month", "r2"])
    from sklearn.metrics import r2_score
    df["month"] = df["date"].dt.to_period("M")
    result = (
        df.groupby("month")
        .apply(lambda g: r2_score(g["close_price"], g["predicted_price"]) if len(g) > 10 else None)
        .dropna()
        .reset_index()
    )
    result.columns = ["month", "r2"]
    result["month"] = result["month"].astype(str)
    return result
```

### Potential Problems & Solutions

**Problem:** `close_price` and `predicted_price` have NaN in some months, crashing r2_score.
**Solution:** Inside the groupby lambda: `g = g.dropna(subset=["close_price", "predicted_price"])`. Also add `if len(g) > 10` guard as shown above.

**Problem:** `sklearn` is not in the dashboard's environment (it's in the pipeline env).
**Solution:** Add `scikit-learn>=1.3.0` to `requirements.txt`. It's a small dependency and safer than reimplementing R².

**Problem:** The period axis labels (`"2025-04"`) look ugly on x-axis.
**Solution:** `result["month"] = result["month"].astype(str)` and Plotly renders it as categorical text — readable enough. Optionally reformat: `pd.Period("2025-04").strftime("%b %Y")`.

---

## Step 8 — Assemble `app.py`

### What to do
Wire everything together in the correct order.

```python
import streamlit as st
from data_loader import load_predictions, load_signals, get_available_dates
from components.kpi_cards import render_kpi_cards
from components.signals_table import render_signals_table
from components.zscore_chart import render_zscore_chart
from components.performance_chart import render_performance_chart

st.set_page_config(
    page_title="BANKNIFTY Mispricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.title("📊 BANKNIFTY\nMispricing")
    st.divider()

    model_choice = st.radio("Model", ["Walk-Forward ✓", "Static"], index=0)
    model_key = "wf" if "Walk" in model_choice else "static"

    df_pred    = load_predictions(model_key)
    df_signals = load_signals(model_key)

    available_dates = get_available_dates(df_signals)
    selected_date   = st.date_input(
        "Trading Date",
        value=available_dates[0],
        min_value=available_dates[-1],
        max_value=available_dates[0]
    )
    st.caption(f"Data: {available_dates[-1]} → {available_dates[0]}")

# ── Filter to selected date ───────────────────────────────
day_signals = df_signals[df_signals["date"].dt.date == selected_date]
day_pred    = df_pred[df_pred["date"].dt.date == selected_date]

if day_signals.empty:
    st.warning(f"No signals found for {selected_date}. Select another date.")
    st.stop()

# ── Panel 1: KPI Cards ────────────────────────────────────
render_kpi_cards(day_signals)
st.divider()

# ── Panels 2 + 3: Table | Histogram ──────────────────────
col_left, col_right = st.columns([6, 4])
with col_left:
    render_signals_table(day_signals)
with col_right:
    render_zscore_chart(day_pred)

st.divider()

# ── Panel 4: Model R² Trend ───────────────────────────────
render_performance_chart(model_key)
```

### Potential Problems & Solutions

**Problem:** `load_predictions` and `load_signals` are called twice (once in sidebar, might be re-called in components).
**Solution:** Because of `@st.cache_data`, calling them multiple times with the same arguments costs nothing after the first call. Safe by design.

**Problem:** Layout breaks on a smaller laptop screen.
**Solution:** The `[6, 4]` column ratio works down to ~1200px wide. For very small screens, Streamlit stacks columns automatically — no action needed.

---

## Step 9 — Styling

### What to do
Add a minimal CSS injection in `app.py` to improve visual quality. Put this right after `st.set_page_config`:

```python
st.markdown("""
<style>
/* Tighten metric card font sizes */
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; letter-spacing: 0.05em; }

/* Slightly darken the sidebar */
[data-testid="stSidebar"] { background-color: #0f1117; }

/* Remove top padding on main area */
.block-container { padding-top: 1.5rem; }

/* Table font size */
[data-testid="stDataFrame"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)
```

### Potential Problems & Solutions

**Problem:** CSS selectors break in a future Streamlit update (Streamlit uses hashed class names internally).
**Solution:** Use `data-testid` selectors, not class names. The `data-testid` attributes are stable across minor Streamlit versions. If something breaks, inspect element in browser to find the new testid.

---

## Step 10 — Testing Checklist

Run through this checklist before calling the dashboard done:

| Test | What to check |
|---|---|
| Latest date loads by default | Open app, verify date picker shows most recent trading day |
| BUY/SELL counts are correct | Cross-check with `wf_trading_signals.csv` manually for one date |
| Table sorted by abs(z_score) | Highest absolute z-score should be row 1 |
| Histogram threshold lines visible | ±2 dashed lines must be clearly visible |
| R² chart covers full study period | Should show Apr 2025 → Mar 2026 |
| Switching Static ↔ WF updates all panels | Change model radio, verify all 4 panels update |
| Date with no signals shows warning | Manually enter a weekend date, should show warning not crash |
| App loads in < 3 seconds after first cache | Reload page (not hard refresh), should be near-instant |

---

## Common Global Problems

**Problem:** `st.cache_data` doesn't invalidate when parquet files are updated (e.g., after rerunning the pipeline).
**Solution:** Add a "Refresh Data" button in the sidebar:
```python
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()
```

**Problem:** Running dashboard from wrong working directory breaks relative paths.
**Solution:** In `data_loader.py`, always anchor paths to the file's location:
```python
BASE = Path(__file__).resolve().parent.parent
```
Never use `os.getcwd()`.

**Problem:** Plotly charts don't respect Streamlit dark/light theme.
**Solution:** Set `plot_bgcolor="rgba(0,0,0,0)"` and `paper_bgcolor="rgba(0,0,0,0)"` on every figure. Font and axis colors should use `font_color="#cccccc"` for dark-mode or detect via a theme variable.

---

## Final File Checklist

```
dashboard/
├── app.py                     ✓ main entry point, layout assembly
├── data_loader.py             ✓ all @st.cache_data loaders + column normalization
├── components/
│   ├── kpi_cards.py           ✓ 3 metric cards
│   ├── signals_table.py       ✓ top 10 table with color
│   ├── zscore_chart.py        ✓ histogram with ±2 lines
│   └── performance_chart.py  ✓ monthly R² line chart
├── utils.py                   ✓ shared formatters (optional)
└── requirements.txt           ✓ streamlit, pandas, plotly, pyarrow, scikit-learn
```

Run with:
```bash
cd your_project_root
streamlit run dashboard/app.py
```
