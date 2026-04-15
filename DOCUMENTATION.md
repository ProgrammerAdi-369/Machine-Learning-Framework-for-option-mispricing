# Option Mispricing Pipeline — Technical Documentation

Complete technical reference for the BANKNIFTY options mispricing detection system.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Pipeline Workflow](#3-pipeline-workflow)
4. [Repository Structure](#4-repository-structure)
5. [Setup & Installation](#5-setup--installation)
6. [Running the Pipeline](#6-running-the-pipeline)
7. [Data](#7-data)
8. [Feature Engineering](#8-feature-engineering)
9. [Model](#9-model)
10. [Trading Signals](#10-trading-signals)
11. [Accuracy Analysis](#11-accuracy-analysis)
12. [Walk-Forward Retraining](#12-walk-forward-retraining)
13. [Daily Pipeline](#13-daily-pipeline)
14. [Dashboard](#14-dashboard)
15. [Output Reference](#15-output-reference)
16. [Configuration Reference](#16-configuration-reference)
17. [Known Limitations](#17-known-limitations)
18. [Troubleshooting](#18-troubleshooting)

---

## 1. Project Overview

### What it does

Options on the same underlying (BANKNIFTY) trade at many different strike prices and expiry dates simultaneously. On any given day, some contracts will be priced too high relative to their "fair value" and others too low — not because the market is wrong about direction, but because cross-sectional demand imbalances create transient mispricings.

This pipeline:

1. Loads one year of NSE BANKNIFTY option chain data (April 2025 – March 2026)
2. Computes implied volatility and a set of cross-sectional regime features for every contract on every day
3. Trains XGBoost to predict a contract's fair log-price given its structural characteristics
4. Measures each contract's deviation from its predicted fair value and normalises it as a daily z-score
5. Flags contracts with z-score > +2 as **SELL** (overpriced) and z-score < −2 as **BUY** (underpriced)
6. Validates the model through a 7-layer accuracy analysis
7. Applies a walk-forward retraining scheme to handle market regime shifts
8. Runs daily inference on new option chain data via `daily_run.py`

### What it is NOT

- Not a directional market prediction system — it does not forecast where BANKNIFTY will move
- Not a high-frequency trading system — batch daily signals only
- Not a backtested strategy with P&L simulation — signal generation only, no transaction cost modelling

### Dataset at a glance

| Attribute | Value |
|-----------|-------|
| Underlying | BANKNIFTY (NSE Bank Nifty Index) |
| Instrument | European-style index options (weekly + monthly expiry) |
| Study period | 1 April 2025 – 30 March 2026 |
| Input files | 35 monthly `.xlsx` files spanning Apr 2023 – Mar 2026 |
| Trading days | ~240 (within study period) |
| Raw rows (post-filter) | ~70,000+ (study period only; 2023/2024 files used for HV warmup) |
| Avg contracts/day | ~293 |
| Option types | CE (Call) and PE (Put), ~50% each |
| Strike range | 17,500 – 47,500 |
| Moneyness range | 0.80 – 1.20 (filtered) |
| DTE range | 1 – 90 days |

---

## 2. System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        OPTION MISPRICING PIPELINE                          ║
║                           System Architecture                               ╚══════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          DATA LAYER                                     │
 │                                                                         │
 │   BANKNIFTY/*.xlsx  ──►  data/raw/master_raw.parquet                   │
 │   (35 monthly files)     (merged, date-normalised)                      │
 │                                │                                        │
 │                                ▼                                        │
 │             data/processed/master_filtered.parquet                      │
 │             (OI ≥ 50, moneyness 0.80–1.20, DTE 1–90)                  │
 │                                │                                        │
 │                                ▼                                        │
 │             data/processed/master_with_iv.parquet                       │
 │             (+ Black-Scholes implied volatility)                        │
 │                                │                                        │
 │                                ▼                                        │
 │             data/features/cross_sectional.parquet    ◄── MAIN INPUT    │
 │             (+ ATM_IV, Skew, TS_Slope, HV_20, normalised OI/Vol)       │
 └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         MODEL LAYER                                     │
 │                                                                         │
 │   cross_sectional.parquet                                               │
 │          │                                                              │
 │          ├── Pre-train transforms (log_OI, log_HV_20, moneyness_sq,    │
 │          │   outlier clipping at 1st/99th percentile)                  │
 │          │                                                              │
 │          ├── Time-based split  ──►  Train (70%)  │  Test (30%)         │
 │          │                          Apr–Oct 2025    Nov–Mar 2026        │
 │          │                                                              │
 │          └── XGBoost Regressor  (16 features → predict log_price)      │
 │                 │                                                       │
 │                 ▼                                                       │
 │          models/xgb_mispricing.joblib   (static model)                 │
 │          outputs/full_predictions.parquet                               │
 └─────────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                        SIGNAL LAYER                                     │
 │                                                                         │
 │   For each trading day:                                                 │
 │     mispricing   = close_price − predicted_price                        │
 │     daily_mean   = mean(mispricing across all contracts that day)       │
 │     daily_std    = std(mispricing across all contracts that day)        │
 │     z_score      = (mispricing − daily_mean) / daily_std               │
 │                                                                         │
 │   Liquidity filter:  OI_normalised > 0.5                               │
 │                      Volume_normalised > 0.5                            │
 │                      DTE > 5                                            │
 │                                                                         │
 │   Signal logic:   z > +2  →  SELL  (overpriced vs daily peers)         │
 │                   z < −2  →  BUY   (underpriced vs daily peers)        │
 │                                                                         │
 │   outputs/trading_signals.csv   (static model signals)                 │
 └─────────────────────────────────────────────────────────────────────────┘
                                  │
                   ┌──────────────┴──────────────┐
                   ▼                             ▼
 ┌──────────────────────────┐   ┌─────────────────────────────────────────┐
 │   VALIDATION LAYER       │   │          ADAPTATION LAYER               │
 │                          │   │                                         │
 │  accuracy_analysis.py    │   │  retrain.py                             │
 │  ├── Layer A: Global     │   │  ├── Fix 4: Z-score recalibration       │
 │  │   fit metrics         │   │  │   (rolling bias removal)             │
 │  ├── Layer B: Residual   │   │  ├── Fix 1+2: Walk-forward retraining   │
 │  │   diagnostics         │   │  │   (monthly expanding-window models)  │
 │  ├── Layer C: Segment    │   │  └── Fix 3: IV-as-target model          │
 │  │   analysis            │   │       (regime-robust alternative)       │
 │  ├── Layer D: Feature    │   │                                         │
 │  │   importance + SHAP   │   │  outputs/wf_predictions.parquet         │
 │  ├── Layer E: Signal     │   │  outputs/wf_trading_signals.csv ◄ USE  │
 │  │   quality             │   │  models/wf_model_<month>.joblib         │
 │  ├── Layer F: Temporal   │   │  models/xgb_iv_target.joblib            │
 │  │   stability           │   └─────────────────────────────────────────┘
 │  └── Layer G: Scorecard  │
 │                          │
 │  Analysis_outcomes/*.png │
 └──────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                       VISUALISATION LAYER                               │
 │  data_visualization.py  →  Visualizations/*.png  (14+ diagnostic plots)│
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         DAILY INFERENCE LAYER                           │
 │  daily_run.py  ──►  daily_features.py  ──►  wf_model_<latest>.joblib   │
 │  outputs/daily/<date>_predictions.parquet                               │
 │  outputs/daily/<date>_signals.csv                                       │
 └─────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         DASHBOARD LAYER                                 │
 │  dashboard/app.py  (Streamlit)                                          │
 │  ├── Reads outputs/*.parquet + outputs/*.csv  (read-only, cached)       │
 │  ├── KPI cards · Top signals table · Z-score histogram                  │
 │  └── Monthly R² trend chart                                             │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component | Library |
|-----------|---------|
| Data manipulation | pandas 2.x, numpy 2.x |
| IV computation | scipy (brentq optimiser) |
| Machine learning | XGBoost 2.x |
| Model serialisation | joblib |
| Feature explainability | shap (optional) |
| Plots | matplotlib 3.x (Agg backend), seaborn |
| Data I/O | pyarrow (parquet), openpyxl (Excel) |
| Parallelisation | concurrent.futures (preprocess.py, N_JOBS = −1) |
| Dashboard | streamlit, plotly |

---

## 3. Pipeline Workflow

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                          EXECUTION WORKFLOW                                ║
╚══════════════════════════════════════════════════════════════════════════════╝

  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 1  │  python preprocess.py                     (~15 min)        │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Phase 1 ─ Load & merge                                                │
  │    • Read all BANKNIFTY/*.xlsx files (35 files, Apr 2023 – Mar 2026)  │
  │    • Normalise date formats across four formats:                       │
  │        - datetime64 (already parsed by openpyxl)                      │
  │        - String DD-MM-YYYY / DD/MM/YYYY (Dataset1 / newer files)      │
  │        - Integer DDMMYY (NSE FEB26 quirk)                             │
  │        - Float Excel serial (Dataset2 / older 2023 files)             │
  │    • Validate date ranges per file                                     │
  │    • Infer spot price from ATM strike (put-call parity proxy)          │
  │    • Save: data/raw/master_raw.parquet                                 │
  │                                                                        │
  │  Phase 2 ─ Filter                                                      │
  │    • Drop contracts with OI < 50                                       │
  │    • Keep moneyness 0.80 – 1.20 (no deep ITM/OTM noise)               │
  │    • Keep DTE 1 – 90 days                                              │
  │    • Drop dates with fewer than 10 valid rows                          │
  │    • Save: data/processed/master_filtered.parquet                      │
  │                                                                        │
  │  Phase 3 ─ Implied Volatility (parallelised, all CPU cores)           │
  │    • Black-Scholes brentq inversion per contract                       │
  │    • Risk-free rate = 6.5% (RBI repo rate)                            │
  │    • Cap IV at 200% (data errors)                                      │
  │    • Drop rows where IV cannot be computed                             │
  │    • Save: data/processed/master_with_iv.parquet                       │
  │                                                                        │
  │  Phase 4 ─ Cross-sectional features (daily scalars joined to rows)    │
  │    • ATM_IV    : mean IV of contracts within ±2% moneyness             │
  │    • Skew      : PE_0.95_IV − CE_1.05_IV                              │
  │    • TS_Slope  : near-expiry IV − far-expiry IV                       │
  │    • HV_20     : 20-day rolling realised vol (annualised)              │
  │    • IV_HV_Spread : ATM_IV − HV_20                                    │
  │    • OI_normalized, Volume_normalized : row / daily average            │
  │    • IV_rank   : within-day IV percentile                              │
  │    • log_price : log(close + 1)   ← MODEL TARGET                      │
  │    • Date-range filter: keep only Apr 2025 – Mar 2026 rows            │
  │    • Save: data/features/cross_sectional.parquet                      │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 2  │  python train.py                           (<2 min)        │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Phase 5a ─ Pre-training transforms                                    │
  │    • log_OI = log(OI_NO_CON + 1)                 — fixes skew ~5.8   │
  │    • log_HV_20 = log(HV_20.clip(0) + 1)          — fixes right skew  │
  │    • moneyness_sq = moneyness²                    — captures smile    │
  │    • Clip IV_relative, OI_normalised, Volume_normalised               │
  │      to [1st, 99th] percentile                                         │
  │    • Clip bounds saved to models/clip_bounds.json                     │
  │                                                                        │
  │  Phase 5b ─ Time-based train/test split                                │
  │    • Sort unique dates, take first 70% as train                        │
  │    • Train: Apr 2025 – Oct 2025  (~53,000 rows)                       │
  │    • Test : Nov 2025 – Mar 2026  (~17,000 rows)                       │
  │    • NO random split — avoids future-data leakage                     │
  │                                                                        │
  │  Phase 5c ─ XGBoost training                                           │
  │    • 16 features, target = log_price                                   │
  │    • Early stopping on eval RMSE (patience = 50 rounds)               │
  │    • Save: models/xgb_mispricing.joblib                                │
  │                                                                        │
  │  Phase 6 ─ Mispricing computation (full dataset)                      │
  │    • predicted_price      = exp(model.predict(features)) − 1          │
  │    • mispricing           = close − predicted_price                   │
  │    • z_score              = (mispricing − daily_mean) / daily_std     │
  │    • Save: outputs/full_predictions.parquet                            │
  │                                                                        │
  │  Phase 7 ─ Signal generation                                           │
  │    • Liquidity filter: OI_normalised > 0.5                            │
  │                        Volume_normalised > 0.5                         │
  │                        DTE > 5                                         │
  │    • z > +2  →  SELL                                                   │
  │    • z < −2  →  BUY                                                    │
  │    • Save: outputs/trading_signals.csv                                 │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 3  │  python accuracy_analysis.py              (<1 min)         │
  │          │  (reads outputs/full_predictions.parquet)                  │
  ├────────────────────────────────────────────────────────────────────────┤
  │  7 diagnostic layers — see Section 11 for details                     │
  │  Prints full report to stdout + saves plots to Analysis_outcomes/     │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 4  │  python retrain.py                        (2–5 min)        │
  │          │  (reads outputs/full_predictions.parquet)                  │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  Fix 4  ─ Z-score recalibration                                        │
  │    • Subtract 5-day rolling mean bias from mispricing before z-score  │
  │    • Result: z-score std rises from ~0.42 → ~0.85                     │
  │                                                                        │
  │  Fix 1+2 ─ Walk-forward retraining                                     │
  │    • For each month M starting from month 8:                          │
  │        Train on all data before month M (expanding window)            │
  │        Predict month M (true out-of-time)                             │
  │        Save model: models/wf_model_<month>.joblib                     │
  │    • Stricter regularisation vs static model                          │
  │                                                                        │
  │  Fix 3  ─ IV-as-target alternative                                     │
  │    • Same features, target = IV instead of log_price                  │
  │    • Regime-robust: IV is dimensionless, independent of spot level    │
  │    • Save: models/xgb_iv_target.joblib                                 │
  │                                                                        │
  │  Saves: outputs/wf_predictions.parquet   (PREFERRED data source)      │
  │         outputs/wf_trading_signals.csv   (PREFERRED signals)          │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                          (optional, any time)
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  OPTIONAL │  python data_visualization.py             (~1 min)        │
  ├────────────────────────────────────────────────────────────────────────┤
  │  Reads: data/features/cross_sectional.parquet                         │
  │  Saves: 16 exploratory plots → Visualizations/*_<timestamp>.png       │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
option-mispricing-pipeline/
│
├── BANKNIFTY/                        # Raw NSE option chain data (source files)
│   ├── BANK_NIFTY_APR25.xlsx         # Dataset1: study period (12 files)
│   ├── ...
│   └── BANK_NIFTY_MAR26.xlsx
│   ├── BANK_NIFTY_April2023.xlsx     # Dataset2: historical warmup (23 files)
│   └── ...
│
├── data/                             # Intermediate pipeline data
│   ├── raw/
│   │   └── master_raw.parquet        # Phase 1 output
│   ├── processed/
│   │   ├── master_filtered.parquet   # Phase 2 output
│   │   └── master_with_iv.parquet    # Phase 3 output
│   ├── features/
│   │   └── cross_sectional.parquet   # Phase 4 output — main ML input
│   ├── option_chain_raw/             # Daily raw CSV files (daily pipeline input)
│   ├── option_chain_processed/       # Formatted daily CSVs
│   └── daily_features/              # Single-day feature parquets
│
├── models/
│   ├── xgb_mispricing.joblib         # Static model (log-price target)
│   ├── xgb_iv_target.joblib          # IV-target alternative model
│   ├── wf_model_2025-12.joblib       # Walk-forward monthly models
│   ├── wf_model_2026-01.joblib
│   ├── wf_model_2026-02.joblib
│   ├── wf_model_2026-03.joblib
│   └── clip_bounds.json              # Outlier clip bounds for inference
│
├── outputs/
│   ├── full_predictions.parquet      # Static model predictions
│   ├── wf_predictions.parquet        # Walk-forward predictions (preferred)
│   ├── trading_signals.csv           # Static model signals
│   ├── wf_trading_signals.csv        # Walk-forward signals (preferred)
│   └── daily/                        # Per-day inference results
│       ├── YYYY-MM-DD_predictions.parquet
│       └── YYYY-MM-DD_signals.csv
│
├── Analysis_outcomes/                # Model accuracy and validation plots
├── Visualizations/                   # Preprocessing & data exploration plots
│
├── dashboard/
│   ├── app.py                        # Main entry point
│   ├── data_loader.py                # Cached data loaders
│   ├── components/
│   │   ├── kpi_cards.py
│   │   ├── signals_table.py
│   │   ├── zscore_chart.py
│   │   └── performance_chart.py
│   ├── utils.py
│   └── requirements.txt
│
├── preprocess.py                     # STEP 1 — Phases 1–4
├── train.py                          # STEP 2 — Phases 5–7
├── accuracy_analysis.py              # STEP 3 — 7-layer accuracy report
├── retrain.py                        # STEP 4 — Walk-forward retraining
├── daily_run.py                      # Daily inference entry point
├── daily_features.py                 # Single-day feature engineering module
├── option_data_formating.py          # Raw daily CSV formatter
└── data_visualization.py             # OPTIONAL — exploratory data plots
```

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Raw BANKNIFTY monthly `.xlsx` files in `BANKNIFTY/`

### Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### Install dependencies

```bash
pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib seaborn pyarrow openpyxl streamlit plotly
```

For optional SHAP feature analysis:

```bash
pip install shap
```

### Data directory

Two naming formats are accepted in `BANKNIFTY/`:

| Dataset | File naming example | Date format | Coverage | Role |
|---------|--------------------|-----------  |----------|------|
| Dataset1 | `BANK_NIFTY_APR25.xlsx` | String `DD-MM-YYYY` | Apr 2025 – Mar 2026 | Study period — model input |
| Dataset2 | `BANK_NIFTY_April2023.xlsx` | Float Excel serial | Apr 2023 – Mar 2025 | HV_20 warmup only |

All `.xlsx` files in the folder are loaded automatically. The 12 study-period files are required; older files improve `HV_20` accuracy for early April 2025 but are optional.

---

## 6. Running the Pipeline

```bash
# Step 1 — Data preprocessing and feature engineering (~15 min)
python preprocess.py

# Step 2 — Model training and signal generation (<2 min)
python train.py

# Step 3 — 7-layer accuracy analysis (<1 min)
python accuracy_analysis.py

# Step 4 — Walk-forward retraining and fixes (2–5 min)
python retrain.py

# Optional — Exploratory data visualisation (~1 min)
python data_visualization.py
```

### What to check after each step

| Step | Key output | Check |
|------|------------|-------|
| preprocess.py | `data/features/cross_sectional.parquet` | Rows within Apr 2025–Mar 2026; no NaN in IV or ATM_IV |
| preprocess.py | stdout date range table | Every file shows sensible earliest/latest dates |
| train.py | `outputs/full_predictions.parquet` | Test R² printed to stdout should be > 0.90 |
| accuracy_analysis.py | stdout scorecard | "0 critical failures" under Layer G |
| retrain.py | `outputs/wf_trading_signals.csv` | Walk-forward z-score std should be ~0.8–1.0 |

### Preferred output files

- **Trading signals:** `outputs/wf_trading_signals.csv` (walk-forward, more robust than static)
- **Full predictions:** `outputs/wf_predictions.parquet` (all model variants + z-scores)

---

## 7. Data

### Source

Monthly option chain files from NSE (National Stock Exchange of India) for the BANKNIFTY index. Each file covers one calendar month with daily end-of-day snapshots of all listed BANKNIFTY options — strike, expiry, close price, open interest, and volume.

### Preprocessing filters

| Filter | Condition | Reason |
|--------|-----------|--------|
| Open interest | OI ≥ 50 contracts | Remove illiquid/phantom contracts |
| Moneyness | 0.80 ≤ strike/spot ≤ 1.20 | Remove deep OTM/ITM contracts with unreliable pricing |
| DTE | 1 – 90 days | Remove contracts expiring today or beyond 3 months |
| Sparse dates | ≥ 10 valid rows/day | Remove holiday/partial-session anomalies |
| IV validity | IV must converge via brentq | Remove contracts where option price is below intrinsic |
| IV ceiling | IV ≤ 200% | Remove data errors |

### Key columns in `cross_sectional.parquet`

| Column | Description | Range |
|--------|-------------|-------|
| Date | Trading date | Apr 2025 – Mar 2026 |
| strike | Option strike price (₹) | 17,500 – 47,500 |
| option_type | CE (Call) or PE (Put) | {CE, PE} |
| close | Market close price (₹) | 0.05 – 5,000 |
| spot | BANKNIFTY spot price (₹) | 41,000 – 47,500 |
| DTE | Days to expiry | 1 – 90 |
| moneyness | strike / spot | 0.80 – 1.20 |
| IV | Implied volatility (annualised) | 0.05 – 2.00 |
| ATM_IV | Daily ATM-level implied vol | 0.05 – 0.50 |
| Skew | PE_0.95 IV − CE_1.05 IV | −0.30 – 0.50 |
| TS_Slope | Near-expiry IV − far-expiry IV | −0.10 – 0.40 |
| HV_20 | 20-day rolling realised vol | 0.05 – 0.80 |
| IV_HV_Spread | ATM_IV − HV_20 | −0.50 – 0.80 |
| OI_normalized | OI / daily mean OI | 0 – 10 |
| Volume_normalized | Volume / daily mean volume | 0 – 15 |
| IV_rank | Within-day IV percentile | 0 – 1 |
| log_price | log(close + 1) — model target | −4.6 – 8.5 |

### Critical column name notes

These exact names must be used across all scripts:

- Date column: `Date` (capital D, datetime64)
- Option price: `close` (not `close_price`)
- Spot price: `spot`
- Open interest: `OI_NO_CON`
- Volume: `VOLUME` (renamed from `TRADED_QUA` in Phase 4)
- Option type: `option_type` (values: `"CE"` or `"PE"`)

---

## 8. Feature Engineering

The model uses 16 engineered features grouped into four categories.

### Contract structure (5 features)

| Feature | How computed | What it captures |
|---------|-------------|-----------------|
| `option_type_encoded` | 1 for CE, 0 for PE | Call vs put premium difference |
| `DTE` | Days until expiry date | Time-value decay (theta) |
| `moneyness` | strike / spot | Position on vol smile |
| `abs_moneyness` | \|moneyness − 1\| | Distance from ATM (symmetric) |
| `moneyness_sq` | moneyness² | Smile curvature (non-linear) |

### Volatility surface (5 features)

| Feature | How computed | What it captures |
|---------|-------------|-----------------|
| `IV` | Black-Scholes inversion (brentq) | Contract-specific implied vol |
| `ATM_IV` | Mean IV within ±2% moneyness | Daily market-level vol |
| `IV_relative` | IV − ATM_IV | How far this contract's vol is from ATM |
| `IV_rank` | Percentile of IV within the day | Within-day vol ranking |
| `IV_HV_Spread` | ATM_IV − HV_20 | Vol premium (implied vs realised) |

### Regime (3 features)

| Feature | How computed | What it captures |
|---------|-------------|-----------------|
| `Skew` | PE_0.95_IV − CE_1.05_IV | Put premium / tail risk demand |
| `TS_Slope` | Near-expiry IV − far-expiry IV | Term structure shape |
| `log_HV_20` | log(HV_20 + 1) | 20-day realised vol, log-transformed |

### Liquidity (3 features)

| Feature | How computed | What it captures |
|---------|-------------|-----------------|
| `OI_normalized` | OI / daily avg OI | Relative open interest |
| `Volume_normalized` | Volume / daily avg volume | Relative trading activity |
| `log_OI` | log(OI_NO_CON + 1) | OI on log scale (removes skew ~5.8) |

### Pre-training transforms applied in `train.py`

Applied directly in `train.py` before fitting (not stored in the parquet):

- `log_OI` — reduces right skew of raw open interest
- `log_HV_20` — reduces right skew of realised volatility  
- `moneyness_sq` — allows model to capture the convex volatility smile
- Outlier clipping of `IV_relative`, `OI_normalized`, `Volume_normalized` to [1st, 99th] percentile

**Why these transforms exist:** The data visualisation audit found:
- `OI_NO_CON` skew ~5.8 — heavily right-skewed, unusable as raw feature
- `HV_20` right-skewed — log-transform normalises it
- `close` price skew ~7.5 — justifies log-price as the model target
- Volatility smile curvature — `moneyness_sq` is needed to prevent systematic OTM residual bias

---

## 9. Model

### Architecture

XGBoost gradient boosted regression tree (XGBRegressor) with log-price as the target.

**Why log-price?** Raw option prices are heavily right-skewed (skewness ~7.5). Log-transforming normalises the distribution and stabilises variance across strikes and DTE levels.

### Hyperparameters (static model)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `n_estimators` | 500 | Maximum boosting rounds |
| `max_depth` | 6 | Maximum tree depth |
| `learning_rate` | 0.05 | Step size shrinkage |
| `subsample` | 0.8 | Row sampling per tree |
| `colsample_bytree` | 0.8 | Feature sampling per tree |
| `min_child_weight` | 10 | Minimum sum of instance weight per leaf |
| `gamma` | 0.1 | Minimum loss reduction for a split |
| `reg_lambda` | 1.0 | L2 regularisation on weights |
| `early_stopping_rounds` | 50 | Stop if eval RMSE stalls for 50 rounds |
| `eval_metric` | rmse | Metric monitored for early stopping |

### Train/test split

Strictly time-based — no random split (random splits leak future data into training and produce unrealistically high test R²):

```
All dates sorted → first 70% = TRAIN,  last 30% = TEST
Train: Apr 2025 – Oct 2025    (~53,000 rows)
Test : Nov 2025 – Mar 2026    (~17,000 rows)
```

### Performance (static model)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Train R² | ~0.975 | Explains 97.5% of log-price variance in-sample |
| Test R² | ~0.952 | 95.2% out-of-sample — strong generalisation |
| Test RMSE | ~0.44 (log units) | Typical log-price prediction error |
| Price RMSE | ~₹5.84 | Average rupee prediction error |
| Price MAE | ~₹4.13 | Median rupee prediction error |
| MAPE | ~7.2% | % error on contracts priced above ₹1 |
| R² gap | ~0.023 | Train − test gap; below 0.05 threshold |

### Mispricing and z-score

```
mispricing  = close_price − predicted_price   (₹ deviation from fair value)

daily_mean  = mean(mispricing)  across all contracts on that day
daily_std   = std(mispricing)   across all contracts on that day

z_score     = (mispricing − daily_mean) / daily_std
```

The cross-sectional z-score measures whether a contract is expensive or cheap *relative to all other contracts on the same day*, not relative to its own historical pricing. This filters out market-wide moves and isolates structural mispricings.

---

## 10. Trading Signals

### Liquidity pre-filter

| Filter | Threshold | Reason |
|--------|-----------|--------|
| `OI_normalized > 0.5` | Above-median open interest | Sufficient liquidity to enter/exit |
| `Volume_normalized > 0.5` | Above-median daily volume | Active trading confirms price discovery |
| `DTE > 5` | More than 5 days to expiry | Avoids gamma/pin risk near expiry |

### Signal logic

| z_score | Signal | Interpretation |
|---------|--------|----------------|
| z > +2 | **SELL** | Contract priced ~2+ std devs above peers — overpriced |
| z < −2 | **BUY** | Contract priced ~2+ std devs below peers — underpriced |
| −2 ≤ z ≤ +2 | NEUTRAL | Within normal cross-sectional range — no signal |

The ±2 threshold corresponds to approximately the top/bottom 2.3% of a standard normal, so roughly 4.6% of liquid contracts on any given day are flagged.

### Signal file columns (`wf_trading_signals.csv`)

| Column | Type | Description |
|--------|------|-------------|
| date | date | Trading date |
| strike | int | Option strike price (₹) |
| option_type | {CE, PE} | Call or Put |
| close_price | float | Market close price (₹) |
| wf_predicted_price | float | Walk-forward model fair value (₹) |
| wf_mispricing | float | Market price − fair value (₹) |
| wf_z_score | float | Cross-sectional z-score |
| signal | {BUY, SELL} | Trading recommendation |
| DTE | int | Days to expiry |
| moneyness | float | strike / spot |
| IV | float | Implied volatility |
| ATM_IV | float | Daily ATM IV level |
| OI_normalized | float | Relative open interest |
| Volume_normalized | float | Relative trading volume |

---

## 11. Accuracy Analysis

`accuracy_analysis.py` produces a 7-layer diagnostic report. All plots are saved to `Analysis_outcomes/` with timestamps.

### Layer A — Global fit metrics

Computes RMSE, MAE, R², and MAPE on both train and test splits. Flags if the R² gap (train minus test) exceeds 0.05, which indicates overfitting.

### Layer B — Residual diagnostics

Six plots:
- Residuals vs predicted (should be a flat cloud — no funnel shape)
- Residual histogram with normal fit overlay
- Q-Q plot (normality check)
- Residuals vs moneyness (smoothed trend should be flat)
- Residuals vs DTE (should be flat)
- Daily mean residual over time (rolling average should hug zero)

Also prints Shapiro-Wilk normality test, skewness, kurtosis, and mean bias.

### Layer C — Segment analysis

Breaks down R², RMSE, MAE, MAPE, and bias by:
- **Moneyness bucket:** Deep OTM, OTM, Near ATM
- **DTE bucket:** Near (1–10d), Mid (11–30d), Far (31–90d)
- **Option type:** CE vs PE
- **Liquidity tier:** High OI (top 25%), Mid OI, Low OI (bottom 25%)

Flags segments where R² < 0.85 or bias > 0.05.

### Layer D — Feature importance and SHAP

Three importance views: Frequency (weight), Predictive power (gain), Sample impact (cover). If `shap` is installed, generates a SHAP summary plot and a moneyness interaction plot.

### Layer E — Signal quality

Six plots covering z-score distribution vs standard normal, z-score vs moneyness/DTE scatter, daily signal count timeline, z-score calibration QQ plot, and mispricing distribution by moneyness bucket. Prints signal counts and z-score statistics.

### Layer F — Temporal stability

Monthly R², RMSE, and bias over the full dataset with train/test shading. Overlays ATM_IV against RMSE to check whether high-vol regimes degrade accuracy. Flags months with R² < 0.90, |bias| > 0.02, or RMSE > 1.5× the monthly mean.

### Layer G — Model scorecard

| Check | Pass condition |
|-------|---------------|
| Global R² (test) | ≥ 0.90 — critical |
| Overfitting (R² gap) | gap < 0.05 — critical |
| Bias (mean residual) | \|bias\| < 0.01 |
| Z-score std | \|std − 1.0\| < 0.2 |
| Z-score mean | \|mean\| < 0.1 |
| ATM R² | ≥ 0.90 — critical |
| Signal rate on liquid | 1% – 15% |

**Verdict:**
- 0 failures, ≤ 2 warnings → **PRODUCTION READY**
- 0 failures, > 2 warnings → **USABLE** — review warnings before live trading
- Any failures → **NEEDS IMPROVEMENT** — address critical failures first

---

## 12. Walk-Forward Retraining

The static model is trained once on 70% of the data. When market regime shifts (as happened in early 2026), predictions drift. `retrain.py` applies four fixes:

### Fix 4 — Z-score recalibration (immediate)

Subtracts a 5-day rolling mean bias from mispricing before computing z-scores. Removes systematic under/over-prediction caused by regime shift.

**Effect:** z-score std rises from ~0.42 → ~0.85; signal count increases meaningfully.

### Fix 1+2 — Walk-forward retraining (core fix)

```
For month M (starting from month 8 of the dataset):
  Training set = all rows with date < month M start (expanding window)
  Test set     = all rows within month M
  Train a fresh XGBoost model on the training set
  Predict the test set (true out-of-sample for that month)
  Save model as models/wf_model_<YYYY-MM>.joblib
```

Walk-forward models use stricter regularisation than the static model:

| Parameter | Static | Walk-forward | Change |
|-----------|--------|--------------|--------|
| `max_depth` | 6 | 5 | Simpler trees |
| `min_child_weight` | 10 | 15 | Prevents overfitting small segments |
| `colsample_bytree` | 0.8 | 0.7 | Less feature sampling |
| `reg_alpha` | 0 | 0.05 | L1 penalty added |
| `reg_lambda` | 1.0 | 1.5 | Stronger L2 |

**Effect:** Each month is predicted by a model that has never seen that month's data. Adapts to changing spot levels and vol regimes.

### Fix 3 — IV-as-target model

An alternative model that predicts implied volatility instead of log-price. IV is dimensionless and market-level independent.

**Features used:** Same 16 minus `IV` itself and `IV_relative` (to avoid target leakage).

**Output:** `predicted_IV` and `IV_z_score` columns in `wf_predictions.parquet`. Use as a secondary signal stream or regime-robust backup.

---

## 13. Daily Pipeline

The daily pipeline allows new data to be processed after each trading day without re-running the full historical pipeline.

### Architecture

```
[After market close]
User drops:  data/option_chain_raw/option_chain_BANKNIFTY-28-Apr-2026.csv
                      │
                      ▼
          python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv
          # or: python daily_run.py --auto
                      │
          ┌───────────┼──────────────────┐
          │           │                  │
    Step A:      Step B: Format    Step C: Features
    Ingest &     (option_data_     (daily_features.py)
    Validate     formating.py)
          │           │                  │
          └───────────┴──────────────────┘
                      │
                Step D: Inference
                load latest wf_model_*.joblib
                      │
                Step E: Signal Generation
                z > +2 → SELL, z < -2 → BUY
                      │
                Step F: Persist Results
                outputs/daily/YYYY-MM-DD_predictions.parquet
                outputs/daily/YYYY-MM-DD_signals.csv
                      │
                Step G: Append to Master Outputs
                outputs/wf_predictions.parquet
                outputs/wf_trading_signals.csv
```

### Usage

```bash
# Specify a file explicitly
python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv

# Auto-pick the newest CSV in data/option_chain_raw/
python daily_run.py --auto

# Force overwrite if already processed
python daily_run.py --auto --force
```

### Input file format

Raw CSV files must be named: `option_chain_{ASSET}-{DD}-{Mon}-{YYYY}.csv`  
Example: `option_chain_BANKNIFTY-28-Apr-2026.csv`

Drop raw files into `data/option_chain_raw/`. The formatter (`option_data_formating.py`) cleans them and writes to `data/option_chain_processed/`.

### Model selection logic

`daily_run.py` loads the walk-forward model with the most recent month that is on or before the trade date. Falls back to the static model if no walk-forward model covers the period.

### Clip bounds

Outlier clip bounds for `IV_relative`, `OI_normalized`, and `Volume_normalized` are saved to `models/clip_bounds.json` during `train.py`. `daily_run.py` loads and applies these same bounds — they are **not** recomputed on single-day data (which would give different bounds and distort predictions).

### HV_20 for a new day

`daily_features.py` loads `data/features/cross_sectional.parquet` to get the recent spot price history, appends today's inferred spot, and computes a 20-day rolling realised vol. The inferred spot uses put-call parity (strike where `|call_ltp - put_ltp|` is minimised).

---

## 14. Dashboard

A Streamlit dashboard for reviewing pipeline output. It reads directly from parquet/CSV files — it never re-runs the pipeline.

### Running

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

- **Model** — toggle between Walk-Forward (recommended) and Static model; falls back to Static if WF files are absent
- **Trading Date** — date picker restricted to dates that have signals
- **Refresh Data** — clears the Streamlit data cache (use after re-running the pipeline)

### Data files read

| File | Used for |
|------|----------|
| `outputs/full_predictions.parquet` | Static model predictions and z-scores |
| `outputs/trading_signals.csv` | Static model BUY/SELL signals |
| `outputs/wf_predictions.parquet` | Walk-forward model predictions (if present) |
| `outputs/wf_trading_signals.csv` | Walk-forward BUY/SELL signals (if present) |

If walk-forward files are missing, the dashboard falls back to static model files automatically.

### File structure

```
dashboard/
├── app.py                    # Main entry point, layout assembly
├── data_loader.py            # @st.cache_data loaders + column normalization
├── components/
│   ├── kpi_cards.py          # Panel 1: KPI metric cards
│   ├── signals_table.py      # Panel 2: top 10 signals table
│   ├── zscore_chart.py       # Panel 3: z-score histogram
│   └── performance_chart.py  # Panel 4: monthly R² line chart
├── utils.py                  # Shared formatters
└── requirements.txt          # streamlit, pandas, plotly, pyarrow, scikit-learn
```

---

## 15. Output Reference

### Folder conventions

| Folder | Contents |
|--------|----------|
| `outputs/` | Data files only — parquet, CSV. No plots. |
| `Analysis_outcomes/` | Model accuracy and validation plots |
| `Visualizations/` | Exploratory data plots |
| `models/` | Serialised joblib model files |

All plot filenames include a `_YYYYMMDD_HHMMSS` timestamp so consecutive runs never overwrite each other.

### Analysis_outcomes plots

| File | Produced by | What it shows |
|------|------------|---------------|
| `val_feature_importance_<ts>.png` | train.py | Feature importance after initial training |
| `val_zscore_distribution_<ts>.png` | train.py | Z-score histogram (static model) |
| `val_zscore_vs_moneyness_<ts>.png` | train.py | Z-score scatter vs moneyness |
| `residual_diagnostics_<ts>.png` | accuracy_analysis.py | 6-panel residual analysis |
| `feature_importance_<ts>.png` | accuracy_analysis.py | 3-view importance (weight/gain/cover) |
| `shap_analysis_<ts>.png` | accuracy_analysis.py | SHAP summary + bar chart (optional) |
| `shap_moneyness_<ts>.png` | accuracy_analysis.py | SHAP moneyness interaction (optional) |
| `signal_quality_<ts>.png` | accuracy_analysis.py | Z-score calibration + signal diagnostics |
| `temporal_stability_<ts>.png` | accuracy_analysis.py | Monthly R²/RMSE/bias over time |
| `fix_comparison_zscore_<ts>.png` | retrain.py | Z-score distribution: static vs walk-forward |
| `fix_comparison_temporal_<ts>.png` | retrain.py | Monthly R²/bias: static vs walk-forward |

### Visualizations plots (data_visualization.py)

| File | What it shows |
|------|---------------|
| `01_temporal_balance_<ts>.png` | Row count per month |
| `02_balance_type_moneyness_dte_<ts>.png` | CE/PE split by moneyness and DTE |
| `03_price_distribution_raw_vs_log_<ts>.png` | Raw price vs log-price distribution |
| `04_iv_distribution_<ts>.png` | IV distribution across contracts |
| `05_feature_distributions_grid_<ts>.png` | All model features in one grid |
| `06_outlier_boxplots_<ts>.png` | Box plots for extreme values |
| `07_outlier_summary_by_feature_<ts>.png` | Count of outliers per feature |
| `08_volatility_smile_<ts>.png` | Mean IV by moneyness bucket |
| `09_timeseries_regime_features_<ts>.png` | ATM_IV, Skew, TS_Slope over time |
| `10_correlation_heatmap_<ts>.png` | Feature correlation matrix |
| `11_qq_normality_plots_<ts>.png` | Q-Q normality check for key features |
| `12_missing_values_<ts>.png` | Missing value heatmap |
| `13_oi_distribution_<ts>.png` | Open interest distribution |
| `14_iv_rank_uniformity_<ts>.png` | IV_rank uniformity check |
| `15_transformation_recommendations_<ts>.png` | Raw vs transformed feature comparison |
| `16_volume_column_audit_<ts>.png` | Volume column data quality audit |

---

## 16. Configuration Reference

### preprocess.py

```python
RISK_FREE_RATE   = 0.065     # RBI repo rate (6.5%) — update if rate changes
OI_MIN           = 50        # Minimum open interest per contract
MONEYNESS_LOW    = 0.80      # Moneyness lower bound
MONEYNESS_HIGH   = 1.20      # Moneyness upper bound
DTE_MIN          = 1         # Minimum days to expiry
DTE_MAX          = 90        # Maximum days to expiry
IV_MAX           = 2.0       # Implied volatility ceiling (200%)
MIN_ROWS_PER_DAY = 10        # Minimum contracts per date
HV_WINDOW        = 20        # Rolling window for realised volatility (trading days)
N_JOBS           = -1        # Parallelisation cores (-1 = all available)

# Study period filter (applied after feature engineering)
STUDY_START      = "2025-04-01"
STUDY_END        = "2026-03-31"
```

### train.py

```python
TRAIN_TEST_SPLIT = 0.70    # 70% of unique dates go to training
FEATURE_COLS = [           # 16 features fed to XGBoost
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness", "moneyness_sq",
    "IV", "ATM_IV", "IV_relative", "IV_rank", "IV_HV_Spread",
    "Skew", "TS_Slope", "log_HV_20",
    "OI_normalized", "Volume_normalized", "log_OI",
]
OUTLIER_CLIP = (0.01, 0.99)  # 1st–99th percentile clipping
```

### retrain.py

```python
ROLLING_BIAS_WINDOW = 5    # Days for Fix 4 rolling bias window
Z_SCORE_STD_FLOOR   = 0.5  # Minimum daily std to avoid divide-by-zero
```

---

## 17. Known Limitations

### Distribution shift in 2026

The static model was trained on 2025 data. In early 2026, BANKNIFTY's vol regime changed, causing R² to drop from 0.95 → 0.88 and systematic positive bias. The walk-forward retraining in `retrain.py` addresses this; use `wf_trading_signals.csv` in production.

### OTM accuracy

Segment analysis (Layer C) consistently shows lower R² for deep OTM contracts. OTM options have very low absolute prices and high relative noise.

### No transaction cost modelling

Signals are generated on raw mispricing. Bid-ask spreads on OTM BANKNIFTY options can be wide relative to the signal size. Any downstream strategy must account for spread, brokerage, STT, and market impact.

### Sparse signals in low-vol regimes

When the market is quiet and daily cross-sectional variance is low, z-score normalisation amplifies small differences, which can lead to unstable signals. `Z_SCORE_STD_FLOOR` in `retrain.py` provides a safety valve.

### No intraday updates

The pipeline operates on end-of-day data. Signals are generated once per day and are not updated intraday.

### Adding new monthly files

When adding a new month's `.xlsx` file to `BANKNIFTY/`, ensure the date column uses a format already handled by `fix_date_column` in `preprocess.py`. If stdout shows `"Date generic fallback parse"` for a new file, inspect its raw Date column and add a new format branch.

---

## 18. Troubleshooting

**`Volume_normalized` is all NaN after preprocess.py**  
The `TRADED_QUA` column fix is already in `preprocess.py`. Re-run `preprocess.py` to regenerate the parquet.

**R² > 0.98 on test set**  
Likely overfitting or feature leakage. Check that `strike` (raw), `log_price`, `Date`, and `close` are not in `FEATURE_COLS`.

**Z-scores all near zero / no signals**  
Model is over-fitting the price surface. Increase regularisation: raise `min_child_weight` and `gamma`, then re-run `train.py`.

**Z-score vs moneyness shows a curve (U-shape)**  
Model is not capturing the volatility smile. Ensure `moneyness_sq` is in `FEATURE_COLS`.

**`retrain.py` skips a month**  
Requires at least 5,000 training rows before a prediction month. First 7 months of data are used as the initial training window.

**SHAP plots not generated**  
Install SHAP: `pip install shap`, then re-run `accuracy_analysis.py`.

**`daily_run.py` fails with model not found**  
Walk-forward models cover months up to March 2026. If today's date is beyond that, the script falls back to the static model. Re-run `retrain.py` with newer data to produce more recent walk-forward models.

**Spot inference is wrong by a large margin on `daily_run.py`**  
High-volatility days can break the put-call parity ATM proxy. `daily_features.py` will warn if the inferred spot is more than 10% off the previous day's spot from `cross_sectional.parquet`.

**NSE added/removed a column in the daily CSV**  
`option_data_formating.py` has a column count guard (expects 21 columns after dropping empty ones). If the count changes, adjust the guard threshold or update the rename map.
