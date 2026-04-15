# Option Mispricing Pipeline — Quick Start Guide

A machine-learning pipeline that detects mispriced BANKNIFTY options by training an XGBoost model to predict fair option prices, then flags contracts trading significantly above or below that fair value as BUY or SELL signals.

For full technical reference, see [DOCUMENTATION.md](DOCUMENTATION.md).

---

## What the pipeline does

1. **Preprocess** — loads raw NSE monthly Excel files, filters by liquidity and moneyness, computes implied volatility (Black-Scholes), and engineers cross-sectional features.
2. **Train** — trains an XGBoost model to predict `log(option price)`, computes cross-sectional z-scores of mispricing, and generates trading signals.
3. **Analyze** — runs a 7-layer accuracy report with residual diagnostics, segment breakdowns, SHAP analysis, and a model scorecard.
4. **Retrain** — applies walk-forward retraining, z-score recalibration, and an IV-as-target alternative model for improved out-of-sample reliability.
5. **Daily run** — ingests a new day's option chain CSV after market close and generates today's signals using the latest walk-forward model.
6. **Dashboard** — Streamlit app that displays historical and daily signals, z-score distributions, and monthly model performance.

---

## Prerequisites

- Python 3.10+
- Virtual environment already set up in `.venv/`
- Raw BANKNIFTY monthly Excel files in `BANKNIFTY/`

Activate the environment:

```bash
source .venv/Scripts/activate      # Windows Git Bash / WSL
# or
.venv\Scripts\activate             # Windows cmd / PowerShell
```

---

## Part A — One-Time Historical Pipeline

Run scripts in this exact sequence. Each script depends on the output of the previous one.

### Step 1 — Preprocess raw data

```bash
python preprocess.py
```

**Reads:** `BANKNIFTY/*.xlsx` (monthly NSE option chain files)  
**Writes:** `data/features/cross_sectional.parquet`  
**Runtime:** ~15 minutes (Black-Scholes IV computation is parallelised)

What it does:

- Loads and merges all monthly Excel files, handling NSE date format quirks (DD-MM-YYYY strings, DDMMYY integers, and Excel serial floats)
- Filters to contracts with OI ≥ 50, moneyness 0.80–1.20, DTE 1–90
- Computes implied volatility per contract using Brent's method (parallelised across all CPU cores)
- Builds cross-sectional features: `ATM_IV`, `IV_rank`, `IV_relative`, `Skew`, `TS_Slope`, `HV_20`, `IV_HV_Spread`, `OI_normalized`, `Volume_normalized`, `moneyness`, `log_price`
- Scopes final output to the study period (Apr 2025 – Mar 2026); older files improve HV_20 warmup but are excluded from the feature set

---

### Step 2 — Train model and generate signals

```bash
python train.py
```

**Reads:** `data/features/cross_sectional.parquet`  
**Writes:**

- `models/xgb_mispricing.joblib` — trained XGBoost model
- `models/clip_bounds.json` — outlier clip bounds for daily inference
- `outputs/trading_signals.csv` — BUY/SELL signals
- `outputs/full_predictions.parquet` — full dataset with predictions and z-scores
- `Analysis_outcomes/val_feature_importance_<ts>.png`
- `Analysis_outcomes/val_zscore_distribution_<ts>.png`
- `Analysis_outcomes/val_zscore_vs_moneyness_<ts>.png`

**Runtime:** < 2 minutes

What it does:

- Applies pre-training transformations: log-transforms OI and HV_20, adds `moneyness_sq`, clips outliers at 1st/99th percentile
- Saves clip bounds to `models/clip_bounds.json` for use in daily inference
- Splits data chronologically (70% train / 30% test) — no random split to avoid data leakage
- Trains XGBoost on 16 features to predict `log_price`
- Computes daily cross-sectional z-scores of mispricing (market price vs model price)
- Applies liquidity filters (`OI_normalized > 0.5`, `Volume_normalized > 0.5`, `DTE > 5`)
- Labels contracts as **SELL** (z > +2, overpriced) or **BUY** (z < -2, underpriced)
- Prints RMSE by moneyness bucket (ATM / ITM / OTM) to stdout

---

### Step 3 — Run accuracy analysis

```bash
python accuracy_analysis.py
```

**Reads:** `outputs/full_predictions.parquet`, `models/xgb_mispricing.joblib`  
**Writes:** `Analysis_outcomes/residual_diagnostics_<ts>.png`, `feature_importance_<ts>.png`, `signal_quality_<ts>.png`, `temporal_stability_<ts>.png`, `shap_analysis_<ts>.png` (if `shap` is installed)

**Runtime:** < 1 minute (+ a few minutes if SHAP runs)

What it does (7 layers):

- **Layer A** — Global fit metrics: RMSE, MAE, R², MAPE on train and test splits
- **Layer B** — Residual diagnostics: scatter plots, histogram, Q-Q plot, Shapiro-Wilk test, daily bias drift
- **Layer C** — Segment-level breakdown: performance by moneyness bucket, DTE bucket, option type (CE/PE), and liquidity tier
- **Layer D** — Feature importance: frequency, gain, and cover views; optional SHAP summary plots
- **Layer E** — Signal quality: z-score distribution vs standard normal, calibration Q-Q, mispricing by bucket, signals per day over time
- **Layer F** — Temporal stability: monthly R², RMSE, bias, and ATM IV vs RMSE correlation
- **Layer G** — Model scorecard: pass/warn/fail checks with a production-readiness verdict

---

### Step 4 — Walk-forward retrain (run before daily use)

```bash
python retrain.py
```

**Reads:** `outputs/full_predictions.parquet`  
**Writes:**

- `outputs/wf_predictions.parquet` — full dataset with walk-forward predictions
- `outputs/wf_trading_signals.csv` — signals from walk-forward model (**use this over `trading_signals.csv`**)
- `Analysis_outcomes/fix_comparison_zscore_<ts>.png`
- `Analysis_outcomes/fix_comparison_temporal_<ts>.png`
- `models/wf_model_<month>.joblib` — one model per month (e.g., `wf_model_2026-01.joblib`)
- `models/xgb_iv_target.joblib` — alternative model that predicts IV instead of log-price

**Runtime:** 2–5 minutes

What it does:

- **Fix 4** — Recalibrates z-scores by removing 5-day rolling model bias before normalising; z-score std rises from ~0.42 → ~0.85
- **Fix 1+2** — Walk-forward retraining: trains a fresh model each month on all data before that month (expanding window), with stronger regularisation (`max_depth=5`, `min_child_weight=15`, L1+L2 penalties)
- **Fix 3** — IV-as-target model: predicts implied volatility instead of log-price, robust to market regime shifts
- Prints a comparison table of static vs walk-forward R², RMSE, bias, and z-score std

---

### Optional — Exploratory data visualisation

```bash
python data_visualization.py
```

**Reads:** `data/features/cross_sectional.parquet`  
**Writes:** `Visualizations/` — 16 diagnostic plots  
**Runtime:** ~1 minute

---

## Part B — Daily Inference (Ongoing Use After Market Close)

After completing the historical pipeline once, use this daily workflow for new trading days.

### Setup

Drop today's raw option chain CSV into `data/option_chain_raw/`. The filename must follow the NSE export format:

```
option_chain_BANKNIFTY-28-Apr-2026.csv
```

### Run daily inference

```bash
# Auto-pick the newest CSV in data/option_chain_raw/
python daily_run.py --auto

# Or specify a file explicitly
python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv

# Force re-process if today was already run
python daily_run.py --auto --force
```

**Writes:**

- `data/option_chain_processed/<file>.csv` — cleaned/formatted option chain
- `data/daily_features/<date>_features.parquet` — single-day feature set
- `outputs/daily/<date>_predictions.parquet` — predictions for today
- `outputs/daily/<date>_signals.csv` — today's BUY/SELL signals
- Appends to `outputs/wf_predictions.parquet` and `outputs/wf_trading_signals.csv`

**Runtime:** < 1 minute

What it does:

1. Validates the input file (name format, date, row count, not already processed)
2. Runs `option_data_formating.py` to clean and standardise the CSV
3. Runs `daily_features.py` to compute all 16 model features for the single day
   - Infers spot price via put-call parity
   - Computes IV per contract (Black-Scholes brentq)
   - Loads recent history from `cross_sectional.parquet` to compute HV_20
4. Loads the most recent walk-forward model (`wf_model_<latest>.joblib`)
5. Applies training-set clip bounds from `models/clip_bounds.json`
6. Runs inference, computes z-scores and signals
7. Saves results and appends to master output files

---

## Part C — Dashboard

```bash
.venv\Scripts\streamlit run dashboard/app.py
# or on macOS/Linux:
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`.

### Panels

| Panel | Description |
|-------|-------------|
| KPI Cards | Total signals, BUY count, SELL count for the selected date |
| Top Signals Table | Top 10 contracts ranked by absolute z-score, color-coded BUY/SELL |
| Z-Score Distribution | Histogram of all z-scores for the selected date with ±2 threshold lines |
| Monthly R² Trend | Line chart of model R² per month across the full study period |

### Sidebar controls

- **Model** — toggle between Walk-Forward (recommended) and Static model
- **Trading Date** — date picker restricted to dates that have signals
- **Refresh Data** — clears the data cache (use after running the pipeline or daily_run.py)

---

## Key Output Files

| File | Description |
|------|-------------|
| `outputs/wf_trading_signals.csv` | BUY/SELL signals from walk-forward model (use this) |
| `outputs/wf_predictions.parquet` | All contracts with predicted prices, mispricing, z-scores |
| `outputs/trading_signals.csv` | BUY/SELL signals from static model (fallback) |
| `outputs/full_predictions.parquet` | Full dataset — static model only |
| `outputs/daily/<date>_signals.csv` | Single-day signals from daily inference |
| `models/xgb_mispricing.joblib` | Static XGBoost model |
| `models/wf_model_<month>.joblib` | Per-month walk-forward models |
| `models/xgb_iv_target.joblib` | IV-as-target alternative model |
| `models/clip_bounds.json` | Outlier clip bounds for daily inference |

---

## Signal Format

Both `wf_trading_signals.csv` and `trading_signals.csv` contain one row per flagged contract:

| Column | Description |
|--------|-------------|
| `date` | Trading date |
| `strike` | Option strike price |
| `option_type` | `CE` (call) or `PE` (put) |
| `close_price` | Market price |
| `wf_predicted_price` | Walk-forward model fair value |
| `wf_mispricing` | Market price − model price |
| `wf_z_score` | Cross-sectional z-score within the day |
| `signal` | `BUY` (z < −2) or `SELL` (z > +2) |
| `DTE` | Days to expiry |
| `moneyness` | Strike / spot |
| `IV`, `ATM_IV` | Volatility context |

---

## Dataset Facts

| Fact | Value |
|------|-------|
| Underlying | BANKNIFTY (NSE) |
| Data range | April 2025 – March 2026 (1 year) |
| Total rows (post-filter) | ~70,000 |
| Trading days | ~240 |
| Avg contracts / day | ~293 |
| Train / test split | 70% / 30% chronological |
| Signal threshold | ±2 standard deviations (cross-sectional, within-day) |
| Liquidity filter | `OI_normalized > 0.5`, `Volume_normalized > 0.5`, `DTE > 5` |
| Target variable | `log_price = log(close + 1)` |

---

## Troubleshooting

**`Volume_normalized` is all NaN after preprocess.py**  
Re-run `preprocess.py` — the `TRADED_QUA` column fix is already in place.

**R² > 0.98 on test set**  
Overfitting or feature leakage. Check that `strike` (raw), `log_price`, `Date`, and `close` are not in `FEATURE_COLS`.

**Z-scores all near zero / no signals**  
Model is over-fitting the price surface. Increase regularisation: raise `min_child_weight` and `gamma`, re-run `train.py`.

**Z-score vs moneyness shows a U-shape curve**  
Model is not capturing the volatility smile. Ensure `moneyness_sq` is in `FEATURE_COLS`.

**`retrain.py` skips a month**  
Requires at least 5,000 training rows before a prediction month. First 7 months are used as the initial window.

**SHAP plots not generated**  
`pip install shap`, then re-run `accuracy_analysis.py`.

**`daily_run.py` — model not found warning**  
Walk-forward models cover up to March 2026. Beyond that, the script falls back to the static model. Re-run `retrain.py` with newer data for more recent walk-forward models.

**`daily_run.py` — spot inference warning**  
On high-volatility days, the put-call parity proxy can be off. A warning is printed if inferred spot is more than 10% from the previous day's spot. Results are still valid but spot-derived features may be slightly noisy.

**Dashboard shows no data**  
Click **Refresh Data** in the sidebar to clear the cache after running the pipeline.
