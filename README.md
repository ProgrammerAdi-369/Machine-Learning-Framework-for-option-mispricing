# Option Mispricing Pipeline

A machine-learning pipeline that detects mispriced BANKNIFTY options. It ingests raw NSE option chain data, engineers volatility and liquidity features, trains an XGBoost regression model, computes cross-sectional z-scores of deviation from predicted fair value, and emits daily BUY/SELL trading signals. A walk-forward retraining layer keeps the model adapted to evolving market regimes. A Streamlit dashboard provides daily signal review.

---

## Documentation

Full technical reference is in [DOCUMENTATION.md](DOCUMENTATION.md) — system architecture, feature engineering, model internals, signal logic, walk-forward design, accuracy analysis layers, all output files, configuration constants, and known limitations.

---

## Quick Start

See [QuickStart.md](QuickStart.md) for the full step-by-step run guide.

```bash
# One-time historical pipeline (run in order)
python preprocess.py          # ~15 min — load Excel files, compute IV, build features
python train.py               # <2 min  — train XGBoost, compute z-scores, emit signals
python accuracy_analysis.py   # <1 min  — 7-layer validation and scorecard
python retrain.py             # 2–5 min — walk-forward retraining (preferred signals)

# Daily inference (after market close, ongoing use)
python daily_run.py --auto
streamlit run dashboard/app.py
```

---

## Project Structure

```
option-mispricing-pipeline/
│
├── BANKNIFTY/                  # Raw NSE monthly option chain Excel files (input)
├── data/                       # Intermediate pipeline data
│   ├── raw/                    # Phase 1: merged raw parquet
│   ├── processed/              # Phase 2–3: filtered + IV parquets
│   ├── features/               # Phase 4: cross_sectional.parquet (main ML input)
│   ├── option_chain_raw/       # Daily raw CSV files dropped here
│   ├── option_chain_processed/ # Formatted daily CSVs
│   └── daily_features/         # Single-day feature parquets
│
├── models/                     # Trained models
│   ├── xgb_mispricing.joblib   # Static model
│   ├── xgb_iv_target.joblib    # IV-as-target alternative model
│   ├── wf_model_<YYYY-MM>.joblib  # Per-month walk-forward models
│   └── clip_bounds.json        # Outlier clip bounds saved from train.py
│
├── outputs/                    # Signal and prediction files
│   ├── trading_signals.csv     # Static model signals
│   ├── wf_trading_signals.csv  # Walk-forward signals (preferred)
│   ├── full_predictions.parquet
│   ├── wf_predictions.parquet
│   └── daily/                  # Per-day inference results
│
├── Analysis_outcomes/          # Model accuracy and validation plots
├── Visualizations/             # Exploratory data plots
│
├── dashboard/                  # Streamlit dashboard
│   ├── app.py
│   ├── data_loader.py
│   ├── components/
│   └── utils.py
│
├── preprocess.py               # Step 1 — Phases 1–4: data loading, IV, features
├── train.py                    # Step 2 — Phases 5–7: training, prediction, signals
├── accuracy_analysis.py        # Step 3 — 7-layer accuracy report
├── retrain.py                  # Step 4 — walk-forward retraining
├── daily_run.py                # Daily inference entry point
├── daily_features.py           # Single-day feature engineering module
├── option_data_formating.py    # Raw daily CSV formatter
└── data_visualization.py       # Optional exploratory data plots
```

---

## Setup

**Requirements:** Python 3.10+, raw BANKNIFTY `.xlsx` files in `BANKNIFTY/`

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib seaborn pyarrow openpyxl streamlit plotly

# Optional (SHAP feature analysis)
pip install shap
```

---

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/wf_trading_signals.csv` | BUY/SELL signals — walk-forward model (use this) |
| `outputs/wf_predictions.parquet` | Full dataset with all model predictions and z-scores |
| `outputs/daily/<date>_signals.csv` | Today's signals from daily inference |
| `models/wf_model_<YYYY-MM>.joblib` | Per-month walk-forward models |

---

## Stack

| Component | Library |
|-----------|---------|
| Data manipulation | pandas 2.x, numpy 2.x |
| IV computation | scipy (brentq optimiser) |
| Machine learning | XGBoost 2.x |
| Model serialisation | joblib |
| Explainability | shap (optional) |
| Plots | matplotlib 3.x, seaborn |
| Data I/O | pyarrow (parquet), openpyxl (Excel) |
| Dashboard | streamlit, plotly |
| Parallelisation | concurrent.futures |

---

## Dataset

| Attribute | Value |
|-----------|-------|
| Underlying | BANKNIFTY (NSE) |
| Study period | April 2025 – March 2026 |
| Rows (post-filter) | ~70,000 |
| Trading days | ~240 |
| Avg contracts/day | ~293 |
| Option types | CE and PE (~50% each) |
| Moneyness range | 0.80 – 1.20 |
| DTE range | 1 – 90 days |
