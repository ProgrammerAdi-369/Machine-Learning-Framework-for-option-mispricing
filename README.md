# Option Mispricing Pipeline

A machine learning pipeline for detecting mispriced BANKNIFTY options on the NSE. The system ingests raw monthly option chain data, engineers volatility and liquidity features, trains an XGBoost regression model, computes cross-sectional mispricing z-scores, and emits daily BUY/SELL trading signals. A walk-forward retraining layer keeps the model adapted to evolving market regimes.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Design](#2-system-design)
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
13. [Output Reference](#13-output-reference)
14. [Configuration Reference](#14-configuration-reference)
15. [Known Limitations](#15-known-limitations)

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

### What it is NOT

- Not a directional market prediction system (it does not forecast where BANKNIFTY will move)
- Not a high-frequency trading system (batch daily signals)
- Not a backtested strategy with P&L simulation (signal generation only)

### Dataset at a glance

| Attribute              | Value                                                             |
| ---------------------- | ----------------------------------------------------------------- |
| Underlying             | BANKNIFTY (NSE Bank Nifty Index)                                  |
| Instrument             | European-style index options (weekly + monthly expiry)            |
| Study period           | 1 April 2025 – 30 March 2026                                      |
| Input files            | 35 monthly `.xlsx` files spanning Apr 2023 – Mar 2026            |
| Trading days           | ~240 (within study period)                                        |
| Raw rows (post-filter) | ~70,000+ (study period only; 2023/2024 files used for HV warmup) |
| Avg contracts/day      | ~293                                                              |
| Option types           | CE (Call) and PE (Put), ~50% each                                 |
| Strike range           | 17,500 – 47,500                                                   |
| Moneyness range        | 0.80 – 1.20 (filtered)                                            |
| DTE range              | 1 – 90 days                                                       |

---

## 2. System Design

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        OPTION MISPRICING PIPELINE                          ║
║                           System Architecture                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                          DATA LAYER                                     │
 │                                                                         │
 │   BANKNIFTY/*.xlsx  ──►  data/raw/master_raw.parquet                   │
 │   (12 monthly files)     (merged, date-normalised)                      │
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
 │  data_visualization.py  →  Visualizations/*.png  (14 diagnostic plots) │
 └─────────────────────────────────────────────────────────────────────────┘
```

### Technology Stack

| Component              | Library                                         |
| ---------------------- | ----------------------------------------------- |
| Data manipulation      | pandas 2.x, numpy 2.x                           |
| IV computation         | scipy (brentq optimiser)                        |
| Machine learning       | XGBoost 2.x                                     |
| Model serialisation    | joblib                                          |
| Feature explainability | shap (optional)                                 |
| Plots                  | matplotlib 3.x (Agg backend), seaborn           |
| Data I/O               | pyarrow (parquet), openpyxl (Excel)             |
| Parallelisation        | concurrent.futures (preprocess.py, N_JOBS = −1) |

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
  │    • Validate date ranges per file — flags files with < 100 valid rows │
  │    • Infer spot price from ATM strike                                  │
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
  │      (2023/2024 data improves HV_20 warmup but is excluded from       │
  │       the final feature set)                                           │
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
  │    • Validation set = test split (monitoring only, not tuning)        │
  │    • Save: models/xgb_mispricing.joblib                                │
  │                                                                        │
  │  Phase 6 ─ Mispricing computation (full dataset)                      │
  │    • predicted_log_price  = model.predict(full feature matrix)        │
  │    • predicted_price      = exp(predicted_log_price) − 1              │
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
  │  Validation plots → Analysis_outcomes/val_*_<timestamp>.png           │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 3  │  python accuracy_analysis.py              (<1 min)         │
  │          │  (run after train.py; reads outputs/full_predictions)      │
  ├────────────────────────────────────────────────────────────────────────┤
  │                                                                        │
  │  7 diagnostic layers — see Section 11 for details                     │
  │  Prints full report to stdout + saves 6 plots to Analysis_outcomes/   │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  STEP 4  │  python retrain.py                        (2–5 min)        │
  │          │  (run after train.py; reads outputs/full_predictions)      │
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
  │         Analysis_outcomes/fix_comparison_*_<timestamp>.png            │
  │                                                                        │
  └────────────────────────────────────────────────────────────────────────┘
                                    │
                          (optional, any time)
                                    ▼
  ┌────────────────────────────────────────────────────────────────────────┐
  │  OPTIONAL │  python data_visualization.py             (~1 min)        │
  ├────────────────────────────────────────────────────────────────────────┤
  │  Reads: data/features/cross_sectional.parquet                         │
  │  Saves: 14 exploratory plots → Visualizations/*_<timestamp>.png       │
  └────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Repository Structure

```
option-mispricing-pipeline/
│
├── BANKNIFTY/                        # Raw NSE option chain data (source files)
│   │                                 # Three naming formats across two datasets:
│   │
│   │   Dataset1 — Study period (Apr 2025 – Mar 2026), string DD-MM-YYYY dates
│   ├── BANK_NIFTY_APR25.xlsx         # BANK_NIFTY_<MON><YY>.xlsx  (12 files)
│   ├── BANK_NIFTY_MAY25.xlsx
│   ├── ...
│   └── BANK_NIFTY_MAR26.xlsx
│   │
│   │   Dataset2 — Historical data (Apr 2023 – Mar 2025), Excel serial dates
│   ├── BANK_NIFTY_April2023.xlsx     # BANK_NIFTY_<Month><YYYY>.xlsx  (2023 files)
│   ├── BANK_NIFTY_Aug2024.xlsx       # BANK_NIFTY_<Mon><YYYY>.xlsx    (2024 files)
│   └── ...                           # 23 files total; used for HV_20 warmup only
│
├── data/                             # Intermediate pipeline data
│   ├── raw/
│   │   └── master_raw.parquet        # Phase 1 output: merged raw Excel
│   ├── processed/
│   │   ├── master_filtered.parquet   # Phase 2 output: liquidity + moneyness filtered
│   │   └── master_with_iv.parquet    # Phase 3 output: + implied volatility
│   └── features/
│       └── cross_sectional.parquet   # Phase 4 output: full feature matrix (main ML input)
│
├── models/                           # Serialised trained models
│   ├── xgb_mispricing.joblib         # Static model (log-price target)
│   ├── xgb_iv_target.joblib          # IV-target alternative model
│   ├── wf_model_2025-12.joblib       # Walk-forward monthly models
│   ├── wf_model_2026-01.joblib
│   ├── wf_model_2026-02.joblib
│   └── wf_model_2026-03.joblib
│
├── outputs/                          # Model predictions & trading signals (data files)
│   ├── full_predictions.parquet      # All contracts with static model predictions
│   ├── wf_predictions.parquet        # Enriched with walk-forward columns
│   ├── trading_signals.csv           # BUY/SELL signals — static model
│   └── wf_trading_signals.csv        # BUY/SELL signals — walk-forward (preferred)
│
├── Visualizations/                   # Preprocessing & data exploration plots
│   └── *_<timestamp>.png             # 14 figures (populated by data_visualization.py)
│
├── Analysis_outcomes/                # Model accuracy & validation plots
│   └── *_<timestamp>.png             # 9 figures (populated by train.py, accuracy_analysis.py, retrain.py)
│
├── preprocess.py                     # STEP 1 — Phases 1–4: data loading, IV, features
├── train.py                          # STEP 2 — Phases 5–7: training, prediction, signals
├── accuracy_analysis.py              # STEP 3 — 7-layer accuracy validation report
├── retrain.py                        # STEP 4 — Walk-forward retraining + 4 fixes
├── data_visualization.py             # OPTIONAL — 14 exploratory data plots
│
├── Blueprint.md                      # Detailed phased implementation plan
├── QuickStart.md                     # 4-step execution guide
├── Report_plan.md                    # Accuracy analysis methodology
├── fix.md                            # Problem diagnosis + fix specification
└── fix_guide_preprocess_multiformat.md  # Multi-format date parsing fix guide
```

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10 or higher
- Windows / macOS / Linux

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
pip install pandas numpy scipy scikit-learn xgboost joblib matplotlib seaborn pyarrow openpyxl
```

For optional SHAP feature analysis:

```bash
pip install shap
```

### Verify the data directory

Ensure the `BANKNIFTY/` folder contains the monthly `.xlsx` files before running `preprocess.py`. The pipeline accepts two naming formats:

- **Dataset1 (study period):** `BANK_NIFTY_APR25.xlsx` … `BANK_NIFTY_MAR26.xlsx` — 12 files, Apr 2025 – Mar 2026
- **Dataset2 (historical):** `BANK_NIFTY_April2023.xlsx`, `BANK_NIFTY_Aug2024.xlsx`, etc. — 2023/2024 files used for HV warmup

All `.xlsx` files in the folder are loaded automatically regardless of naming convention. Only the 12 study-period files are required; the older files improve `HV_20` accuracy but are optional.

---

## 6. Running the Pipeline

Run the four steps in order. Each step depends on the output of the previous one.

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

| Step                 | Key output                              | Check                                                         |
| -------------------- | --------------------------------------- | ------------------------------------------------------------- |
| preprocess.py        | `data/features/cross_sectional.parquet` | Rows within Apr 2025–Mar 2026; no NaN in IV or ATM_IV        |
| preprocess.py        | stdout date range table                 | Every file shows sensible earliest/latest dates               |
| train.py             | `outputs/full_predictions.parquet`      | Test R² printed to stdout should be > 0.90                    |
| accuracy_analysis.py | stdout scorecard                        | "0 critical failures" under Layer G                           |
| retrain.py           | `outputs/wf_trading_signals.csv`        | Walk-forward z-score std should be ~0.8–1.0                   |

### Preferred output files

After running all steps, use these files for downstream analysis:

- **Trading signals**: `outputs/wf_trading_signals.csv` (walk-forward, more robust than static)
- **Full predictions**: `outputs/wf_predictions.parquet` (contains all model variants + z-scores)
- **Plots**: `Analysis_outcomes/` for model metrics, `Visualizations/` for data distributions

---

## 7. Data

### Source

Monthly option chain files from NSE (National Stock Exchange of India) for the BANKNIFTY index. Each file covers one calendar month and contains daily end-of-day snapshots of all listed BANKNIFTY options with their strike, expiry, close price, open interest, and volume.

Two distinct file formats are present in `BANKNIFTY/`:

| Dataset   | File naming example         | Date format in file       | Coverage              | Role in pipeline             |
| --------- | --------------------------- | ------------------------- | --------------------- | ---------------------------- |
| Dataset1  | `BANK_NIFTY_APR25.xlsx`     | String `DD-MM-YYYY`       | Apr 2025 – Mar 2026   | Study period — model input   |
| Dataset2  | `BANK_NIFTY_April2023.xlsx` | Float Excel serial        | Apr 2023 – Mar 2025   | HV_20 warmup only            |

`preprocess.py` auto-detects the date format per file. The final `cross_sectional.parquet` is scoped to the study period (Apr 2025 – Mar 2026) via a post-feature date-range filter; Dataset2 rows are used to compute the 20-day rolling historical volatility for early April 2025 but are then excluded.

### Preprocessing filters

| Filter        | Condition                   | Reason                                                 |
| ------------- | --------------------------- | ------------------------------------------------------ |
| Open interest | OI ≥ 50 contracts           | Remove illiquid/phantom contracts                      |
| Moneyness     | 0.80 ≤ strike/spot ≤ 1.20   | Remove deep OTM/ITM contracts with unreliable pricing  |
| DTE           | 1 – 90 days                 | Remove contracts expiring today or beyond 3 months     |
| Sparse dates  | ≥ 10 valid rows/day         | Remove holiday/partial-session anomalies               |
| IV validity   | IV must converge via brentq | Remove contracts where option price is below intrinsic |
| IV ceiling    | IV ≤ 200%                   | Remove data errors                                     |

### Key columns in `cross_sectional.parquet`

| Column            | Description                     | Range               |
| ----------------- | ------------------------------- | ------------------- |
| Date              | Trading date                    | Apr 2025 – Mar 2026 |
| strike            | Option strike price (₹)         | 17,500 – 47,500     |
| option_type       | CE (Call) or PE (Put)           | {CE, PE}            |
| close             | Market close price (₹)          | 0.05 – 5,000        |
| spot              | BANKNIFTY spot price (₹)        | 41,000 – 47,500     |
| DTE               | Days to expiry                  | 1 – 90              |
| moneyness         | strike / spot                   | 0.80 – 1.20         |
| IV                | Implied volatility (annualised) | 0.05 – 2.00         |
| ATM_IV            | Daily ATM-level implied vol     | 0.05 – 0.50         |
| Skew              | PE_0.95 IV − CE_1.05 IV         | −0.30 – 0.50        |
| TS_Slope          | Near-expiry IV − far-expiry IV  | −0.10 – 0.40        |
| HV_20             | 20-day rolling realised vol     | 0.05 – 0.80         |
| IV_HV_Spread      | ATM_IV − HV_20                  | −0.50 – 0.80        |
| OI_normalized     | OI / daily mean OI              | 0 – 10              |
| Volume_normalized | Volume / daily mean volume      | 0 – 15              |
| IV_rank           | Within-day IV percentile        | 0 – 1               |
| log_price         | log(close + 1) — model target   | −4.6 – 8.5          |

---

## 8. Feature Engineering

The model uses 16 engineered features grouped into four categories:

### Contract structure (5 features)

| Feature               | How computed           | What it captures               |
| --------------------- | ---------------------- | ------------------------------ |
| `option_type_encoded` | 1 for CE, 0 for PE     | Call vs put premium difference |
| `DTE`                 | Days until expiry date | Time-value decay (theta)       |
| `moneyness`           | strike / spot          | Position on vol smile          |
| `abs_moneyness`       | \|moneyness − 1\|      | Distance from ATM (symmetric)  |
| `moneyness_sq`        | moneyness²             | Smile curvature (non-linear)   |

### Volatility surface (5 features)

| Feature        | How computed                     | What it captures                        |
| -------------- | -------------------------------- | --------------------------------------- |
| `IV`           | Black-Scholes inversion (brentq) | Contract-specific implied vol           |
| `ATM_IV`       | Mean IV within ±2% moneyness     | Daily market-level vol                  |
| `IV_relative`  | IV − ATM_IV                      | How far this contract's vol is from ATM |
| `IV_rank`      | Percentile of IV within the day  | Within-day vol ranking                  |
| `IV_HV_Spread` | ATM_IV − HV_20                   | Vol premium (implied vs realised)       |

### Regime (3 features)

| Feature     | How computed                   | What it captures                     |
| ----------- | ------------------------------ | ------------------------------------ |
| `Skew`      | PE_0.95_IV − CE_1.05_IV        | Put premium / tail risk demand       |
| `TS_Slope`  | Near-expiry IV − far-expiry IV | Term structure shape                 |
| `log_HV_20` | log(HV_20 + 1)                 | 20-day realised vol, log-transformed |

### Liquidity (3 features)

| Feature             | How computed              | What it captures                    |
| ------------------- | ------------------------- | ----------------------------------- |
| `OI_normalized`     | OI / daily avg OI         | Relative open interest              |
| `Volume_normalized` | Volume / daily avg volume | Relative trading activity           |
| `log_OI`            | log(OI_NO_CON + 1)        | OI on log scale (removes skew ~5.8) |

### Pre-training transforms applied in `train.py`

Three additional transforms applied directly in `train.py` before fitting:

- `log_OI` — reduces right skew of raw open interest
- `log_HV_20` — reduces right skew of realised volatility
- `moneyness_sq` — allows the model to capture the convex volatility smile
- Outlier clipping of `IV_relative`, `OI_normalized`, `Volume_normalized` to [1st, 99th] percentile

---

## 9. Model

### Architecture

XGBoost gradient boosted regression tree (XGBRegressor) with log-price as the target.

**Why log-price?** Raw option prices are heavily right-skewed (skewness ~7.5). Log-transforming normalises the distribution and stabilises variance across strikes and DTE levels, leading to significantly lower RMSE.

### Hyperparameters

| Parameter               | Value | Purpose                                          |
| ----------------------- | ----- | ------------------------------------------------ |
| `n_estimators`          | 500   | Maximum number of boosting rounds                |
| `max_depth`             | 6     | Maximum tree depth (controls complexity)         |
| `learning_rate`         | 0.05  | Step size shrinkage                              |
| `subsample`             | 0.8   | Row sampling per tree (reduces overfitting)      |
| `colsample_bytree`      | 0.8   | Feature sampling per tree                        |
| `min_child_weight`      | 10    | Minimum sum of instance weight per leaf          |
| `gamma`                 | 0.1   | Minimum loss reduction for a split               |
| `reg_lambda`            | 1.0   | L2 regularisation on weights                     |
| `early_stopping_rounds` | 50    | Stop if eval RMSE does not improve for 50 rounds |
| `eval_metric`           | rmse  | Metric monitored for early stopping              |

### Train/test split

Strictly time-based (no random split):

```
All dates sorted → first 70% = TRAIN,  last 30% = TEST
Train: Apr 2025 – Oct 2025    (~53,000 rows)
Test : Nov 2025 – Mar 2026    (~17,000 rows)
```

Random splits would leak future data into training and produce unrealistically high test R².

### Performance (static model)

| Metric     | Value             | Interpretation                                       |
| ---------- | ----------------- | ---------------------------------------------------- |
| Train R²   | ~0.975            | Model explains 97.5% of log-price variance in-sample |
| Test R²    | ~0.952            | 95.2% out-of-sample — strong generalisation          |
| Test RMSE  | ~0.44 (log units) | Typical log-price prediction error                   |
| Price RMSE | ~₹5.84            | Average rupee prediction error                       |
| Price MAE  | ~₹4.13            | Median rupee prediction error                        |
| MAPE       | ~7.2%             | % error on contracts priced above ₹1                 |
| R² gap     | ~0.023            | Train − test gap; below 0.05 threshold (no overfit)  |

### Mispricing and z-score

```
mispricing  = close_price − predicted_price   (₹ deviation from fair value)

daily_mean  = mean(mispricing)  across all contracts on that day
daily_std   = std(mispricing)   across all contracts on that day

z_score     = (mispricing − daily_mean) / daily_std
```

The cross-sectional z-score measures whether a contract is expensive or cheap _relative to all other contracts on the same day_, not relative to its own historical pricing. This filters out market-wide moves and isolates structural mispricings.

---

## 10. Trading Signals

### Liquidity pre-filter

Before any signal is generated, contracts must pass all three conditions:

| Filter                    | Threshold                  | Reason                                  |
| ------------------------- | -------------------------- | --------------------------------------- |
| `OI_normalized > 0.5`     | Above-median open interest | Sufficient liquidity to enter/exit      |
| `Volume_normalized > 0.5` | Above-median daily volume  | Active trading confirms price discovery |
| `DTE > 5`                 | More than 5 days to expiry | Avoids gamma/pin risk near expiry       |

### Signal logic

| z_score     | Signal   | Interpretation                                                    |
| ----------- | -------- | ----------------------------------------------------------------- |
| z > +2      | **SELL** | Contract priced ~2+ standard deviations above peers — overpriced  |
| z < −2      | **BUY**  | Contract priced ~2+ standard deviations below peers — underpriced |
| −2 ≤ z ≤ +2 | NEUTRAL  | Within normal cross-sectional range — no signal                   |

The ±2 threshold corresponds to approximately the top/bottom 2.3% of a standard normal distribution, so roughly 4.6% of liquid contracts on any given day are flagged.

### Signal file columns (`wf_trading_signals.csv`)

| Column             | Type        | Description                       |
| ------------------ | ----------- | --------------------------------- |
| date               | date        | Trading date                      |
| strike             | int         | Option strike price (₹)           |
| option_type        | {CE, PE}    | Call or Put                       |
| close_price        | float       | Market close price (₹)            |
| wf_predicted_price | float       | Walk-forward model fair value (₹) |
| wf_mispricing      | float       | Market price − fair value (₹)     |
| wf_z_score         | float       | Cross-sectional z-score           |
| signal             | {BUY, SELL} | Trading recommendation            |
| DTE                | int         | Days to expiry                    |
| moneyness          | float       | strike / spot                     |
| IV                 | float       | Implied volatility                |
| ATM_IV             | float       | Daily ATM IV level                |
| OI_normalized      | float       | Relative open interest            |
| Volume_normalized  | float       | Relative trading volume           |

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

- **Moneyness bucket**: Deep OTM, OTM, Near ATM
- **DTE bucket**: Near (1–10d), Mid (11–30d), Far (31–90d)
- **Option type**: CE vs PE
- **Liquidity tier**: High OI (top 25%), Mid OI, Low OI (bottom 25%)

Flags segments where R² < 0.85 or bias > 0.05.

### Layer D — Feature importance and SHAP

Three importance views: Frequency (weight), Predictive power (gain), Sample impact (cover). If `shap` is installed, also generates a SHAP summary plot and a moneyness interaction plot showing how moneyness drives predictions.

### Layer E — Signal quality

Six plots covering z-score distribution vs standard normal, z-score vs moneyness/DTE scatter, daily signal count timeline, z-score calibration QQ plot, and mispricing distribution by moneyness bucket. Prints signal counts and z-score statistics (mean should be ~0, std should be ~1).

### Layer F — Temporal stability

Monthly R², RMSE, and bias over the full dataset with train/test shading. Overlays ATM_IV against RMSE to check whether high-vol regimes degrade accuracy. Flags months with R² < 0.90, |bias| > 0.02, or RMSE > 1.5× the monthly mean.

### Layer G — Model scorecard

| Check                 | Pass condition        |
| --------------------- | --------------------- |
| Global R² (test)      | ≥ 0.90 — critical     |
| Overfitting (R² gap)  | gap < 0.05 — critical |
| Bias (mean residual)  | \|bias\| < 0.01       |
| Z-score std           | \|std − 1.0\| < 0.2   |
| Z-score mean          | \|mean\| < 0.1        |
| ATM R²                | ≥ 0.90 — critical     |
| Signal rate on liquid | 1% – 15%              |

**Verdict:**

- 0 failures, ≤ 2 warnings → **PRODUCTION READY**
- 0 failures, > 2 warnings → **USABLE** — review warnings before live trading
- Any failures → **NEEDS IMPROVEMENT** — address critical failures first

---

## 12. Walk-Forward Retraining

The static model is trained once on 70% of the data. When the market regime shifts (as happened in early 2026), its predictions drift. `retrain.py` applies four fixes:

### Fix 4 — Z-score recalibration (immediate)

Subtracts a 5-day rolling mean bias from mispricing before computing z-scores. This removes the systematic under/over-prediction caused by regime shift and restores z-score variance toward 1.0.

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

| Parameter          | Static | Walk-forward | Change                              |
| ------------------ | ------ | ------------ | ----------------------------------- |
| `max_depth`        | 6      | 5            | Simpler trees                       |
| `min_child_weight` | 10     | 15           | Prevents overfitting small segments |
| `colsample_bytree` | 0.8    | 0.7          | Less feature sampling               |
| `reg_alpha`        | 0      | 0.05         | L1 penalty added                    |
| `reg_lambda`       | 1.0    | 1.5          | Stronger L2                         |

**Effect:** Each month is predicted by a model that has never seen that month's data. Adapts to changing spot levels and vol regimes.

### Fix 3 — IV-as-target model

An alternative model that predicts implied volatility instead of log-price. IV is dimensionless and market-level independent — it means the same thing regardless of where BANKNIFTY spot is trading.

**Features used:** Same 16 minus `IV` itself (to avoid target leakage), minus `IV_relative` (derived from IV).

**Output:** `predicted_IV` and `IV_z_score` columns in `wf_predictions.parquet`. Use as a secondary signal stream or a regime-robust backup.

---

## 13. Output Reference

### Folder conventions

| Folder               | Contents                                                                                    |
| -------------------- | ------------------------------------------------------------------------------------------- |
| `outputs/`           | Data files only — parquet, CSV. No plots.                                                   |
| `Analysis_outcomes/` | Model accuracy and validation plots (from `train.py`, `accuracy_analysis.py`, `retrain.py`) |
| `Visualizations/`    | Exploratory data plots (from `data_visualization.py`)                                       |
| `models/`            | Serialised joblib model files                                                               |

All plot filenames include a `_YYYYMMDD_HHMMSS` timestamp so consecutive runs never overwrite each other.

### Analysis_outcomes plots

| File                               | Produced by          | What it shows                                  |
| ---------------------------------- | -------------------- | ---------------------------------------------- |
| `val_feature_importance_<ts>.png`  | train.py             | Feature importance after initial training      |
| `val_zscore_distribution_<ts>.png` | train.py             | Z-score histogram (static model)               |
| `val_zscore_vs_moneyness_<ts>.png` | train.py             | Z-score scatter vs moneyness                   |
| `residual_diagnostics_<ts>.png`    | accuracy_analysis.py | 6-panel residual analysis                      |
| `feature_importance_<ts>.png`      | accuracy_analysis.py | 3-view importance (weight/gain/cover)          |
| `shap_analysis_<ts>.png`           | accuracy_analysis.py | SHAP summary + bar chart (if shap installed)   |
| `shap_moneyness_<ts>.png`          | accuracy_analysis.py | SHAP moneyness interaction (if shap installed) |
| `signal_quality_<ts>.png`          | accuracy_analysis.py | Z-score calibration + signal diagnostics       |
| `temporal_stability_<ts>.png`      | accuracy_analysis.py | Monthly R²/RMSE/bias over time                 |
| `fix_comparison_zscore_<ts>.png`   | retrain.py           | Z-score distribution: static vs walk-forward   |
| `fix_comparison_temporal_<ts>.png` | retrain.py           | Monthly R²/bias: static vs walk-forward        |

### Visualizations plots (data_visualization.py)

| File                                         | What it shows                                 |
| -------------------------------------------- | --------------------------------------------- |
| `01_temporal_balance_<ts>.png`               | Row count per month (red = under-represented) |
| `02_balance_type_moneyness_dte_<ts>.png`     | CE/PE split by moneyness and DTE              |
| `03_price_distribution_raw_vs_log_<ts>.png`  | Raw price vs log-price distribution           |
| `04_iv_distribution_<ts>.png`                | IV distribution across contracts              |
| `05_feature_distributions_grid_<ts>.png`     | All 14 model features in one grid             |
| `06_outlier_boxplots_<ts>.png`               | Box plots to spot extreme values              |
| `07_outlier_summary_by_feature_<ts>.png`     | Count of outliers per feature                 |
| `08_volatility_smile_<ts>.png`               | Mean IV by moneyness bucket                   |
| `09_timeseries_regime_features_<ts>.png`     | ATM_IV, Skew, TS_Slope over time              |
| `10_correlation_heatmap_<ts>.png`            | Feature correlation matrix                    |
| `11_qq_normality_plots_<ts>.png`             | Q-Q normality check for key features          |
| `12_missing_values_<ts>.png`                 | Missing value heatmap                         |
| `13_oi_distribution_<ts>.png`                | Open interest distribution                    |
| `14_iv_rank_uniformity_<ts>.png`             | IV_rank uniformity check (should be flat)     |
| `15_transformation_recommendations_<ts>.png` | Raw vs transformed feature comparison         |
| `16_volume_column_audit_<ts>.png`            | Volume column data quality audit              |

---

## 14. Configuration Reference

### preprocess.py

```python
RISK_FREE_RATE  = 0.065    # RBI repo rate (6.5%) — update if rate changes
OI_MIN          = 50       # Minimum open interest per contract
MONEYNESS_LOW   = 0.80     # Moneyness lower bound
MONEYNESS_HIGH  = 1.20     # Moneyness upper bound
DTE_MIN         = 1        # Minimum days to expiry
DTE_MAX         = 90       # Maximum days to expiry
IV_MAX          = 2.0      # Implied volatility ceiling (200%)
MIN_ROWS_PER_DAY = 10      # Minimum contracts per date (sparse date filter)
HV_WINDOW       = 20       # Rolling window for realised volatility (trading days)
N_JOBS          = -1       # Parallelisation cores (-1 = all available)

# Study period filter (applied after feature engineering in main())
STUDY_START     = "2025-04-01"   # First date kept in cross_sectional.parquet
STUDY_END       = "2026-03-31"   # Last date kept in cross_sectional.parquet
# Files outside this range are still loaded — they improve HV_20 warmup
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

## 15. Known Limitations

### Distribution shift in 2026

The static model was trained on 2025 data. In early 2026, BANKNIFTY's vol regime changed, causing R² to drop from 0.95 → 0.88 and systematic positive bias. The walk-forward retraining in `retrain.py` addresses this; use `wf_trading_signals.csv` rather than `trading_signals.csv` in production.

### OTM accuracy

Segment analysis (Layer C) consistently shows lower R² for deep OTM contracts. OTM options have very low absolute prices and high relative noise; the model's log-price representation partially mitigates this but does not eliminate it.

### No transaction cost modelling

Signals are generated on raw mispricing. Bid-ask spreads on OTM BANKNIFTY options can be wide relative to the signal size. Any downstream strategy implementation must account for spread, brokerage, STT, and market impact.

### Sparse signals in low-vol regimes

When the market is quiet and daily cross-sectional variance is low, z-score normalisation amplifies small differences, which can lead to unstable signals. The `Z_SCORE_STD_FLOOR` in `retrain.py` provides a safety valve.

### No intraday updates

The pipeline operates on end-of-day data. Signals are generated once per day and are not updated intraday. Conditions can change significantly between the signal generation time and actual trade execution.

### Adding new monthly files

When adding a new month's `.xlsx` file to `BANKNIFTY/`, ensure the date column uses a format already handled by `fix_date_column` in `preprocess.py` (DD-MM-YYYY string, DDMMYY integer, or Excel serial float). If the console output shows `"Date generic fallback parse"` for a new file, inspect its raw Date column and add a new format branch. See `fix_guide_preprocess_multiformat.md` for the full guide.
