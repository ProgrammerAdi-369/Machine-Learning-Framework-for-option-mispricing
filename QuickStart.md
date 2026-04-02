# Option Mispricing Pipeline — Quick Start Guide

A machine-learning pipeline that detects mispriced BANKNIFTY options by training an XGBoost model to predict fair option prices, then flags contracts trading significantly above or below that fair value as BUY or SELL signals.

---

## What the pipeline does

1. **Preprocess** — loads raw NSE monthly Excel files, filters by liquidity and moneyness, computes implied volatility (Black-Scholes), and engineers cross-sectional features.
2. **Train** — trains an XGBoost model to predict `log(option price)`, computes cross-sectional z-scores of mispricing, and generates trading signals.
3. **Analyze** — runs a 7-layer accuracy report with residual diagnostics, segment breakdowns, SHAP analysis, and a model scorecard.
4. **Retrain** — applies walk-forward retraining, z-score recalibration, and an IV-as-target alternative model for improved out-of-sample reliability.

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

## Run order

Run scripts in this exact sequence. Each script depends on the output of the previous one.

### Step 1 — Preprocess raw data

```bash
python preprocess.py
```

**Reads:** `BANKNIFTY/*.xlsx` (monthly NSE option chain files)  
**Writes:** `data/features/cross_sectional.parquet`  
**Runtime:** ~15 minutes (Black-Scholes IV computation is parallelized)

What it does:

- Loads and merges all monthly Excel files, handling NSE date format quirks
- Filters to contracts with OI ≥ 50, moneyness 0.80–1.20, DTE 1–90
- Computes implied volatility per contract using Brent's method
- Builds cross-sectional features: `ATM_IV`, `IV_rank`, `IV_relative`, `Skew`, `TS_Slope`, `HV_20`, `IV_HV_Spread`, `OI_normalized`, `Volume_normalized`, `moneyness`, `log_price`

---

### Step 2 — Train model and generate signals

```bash
python train.py
```

**Reads:** `data/features/cross_sectional.parquet`  
**Writes:**

- `models/xgb_mispricing.joblib` — trained XGBoost model
- `outputs/trading_signals.csv` — BUY/SELL signals
- `outputs/full_predictions.parquet` — full dataset with predictions and z-scores
- `outcomes/val_feature_importance.png`
- `outcomes/val_zscore_distribution.png`
- `outcomes/val_zscore_vs_moneyness.png`

**Runtime:** < 2 minutes

What it does:

- Applies pre-training transformations: log-transforms OI and HV_20, adds `moneyness_sq`, clips outliers
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
**Writes:** `outputs/residual_diagnostics.png`, `outputs/feature_importance.png`, `outputs/signal_quality.png`, `outputs/temporal_stability.png`, `outputs/shap_analysis.png` (if `shap` is installed)

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

### Step 4 — Walk-forward retrain (recommended before live use)

```bash
python retrain.py
```

**Reads:** `outputs/full_predictions.parquet`  
**Writes:**

- `outputs/wf_predictions.parquet` — full dataset with walk-forward predictions
- `outputs/wf_trading_signals.csv` — signals from walk-forward model (preferred over `trading_signals.csv`)
- `outputs/fix_comparison_zscore.png` — static vs walk-forward vs IV-target z-score distributions
- `outputs/fix_comparison_temporal.png` — monthly R² and bias comparison
- `models/wf_model_<month>.joblib` — one model per month (e.g., `wf_model_2026-01.joblib`)
- `models/xgb_iv_target.joblib` — alternative model that predicts IV instead of log-price

**Runtime:** 2–5 minutes

What it does:

- **Fix 4** — Recalibrates z-scores by removing 5-day rolling model bias before normalizing, producing healthier signal variance
- **Fix 1+2** — Walk-forward retraining: trains a fresh model each month on all data before that month (expanding window), with stronger regularization (`max_depth=5`, `min_child_weight=15`, L1+L2 penalties)
- **Fix 3** — IV-as-target model: predicts implied volatility instead of log-price, robust to market regime shifts
- Prints a comparison table of static vs walk-forward R², RMSE, bias, and z-score std

---

## Key output files

| File                               | Description                                               |
| ---------------------------------- | --------------------------------------------------------- |
| `outputs/trading_signals.csv`      | BUY/SELL signals from static model                        |
| `outputs/wf_trading_signals.csv`   | BUY/SELL signals from walk-forward model (more reliable)  |
| `outputs/full_predictions.parquet` | All contracts with predicted prices, mispricing, z-scores |
| `outputs/wf_predictions.parquet`   | Same, enriched with walk-forward and IV-target columns    |
| `models/xgb_mispricing.joblib`     | Static XGBoost model                                      |
| `models/wf_model_<month>.joblib`   | Per-month walk-forward models                             |
| `models/xgb_iv_target.joblib`      | IV-as-target alternative model                            |

---

## Signal format

Both `trading_signals.csv` and `wf_trading_signals.csv` contain one row per flagged contract:

| Column                        | Description                            |
| ----------------------------- | -------------------------------------- |
| `Date` / `date`               | Trading date                           |
| `strike`                      | Option strike price                    |
| `option_type`                 | `CE` (call) or `PE` (put)              |
| `close` / `close_price`       | Market price                           |
| `predicted_price`             | Model fair value                       |
| `mispricing`                  | Market price − model price             |
| `z_score` / `wf_z_score`      | Cross-sectional z-score within the day |
| `signal`                      | `BUY` (z < −2) or `SELL` (z > +2)      |
| `DTE`                         | Days to expiry                         |
| `moneyness`                   | Strike / spot                          |
| `IV`, `ATM_IV`, `IV_relative` | Volatility context                     |

---

## Dataset facts

| Fact                     | Value                                                       |
| ------------------------ | ----------------------------------------------------------- |
| Underlying               | BANKNIFTY (NSE)                                             |
| Data range               | April 2025 – March 2026 (1 year)                            |
| Total rows (post-filter) | ~70,000                                                     |
| Trading days             | ~240                                                        |
| Avg contracts / day      | ~293                                                        |
| Train / test split       | 70% / 30% chronological                                     |
| Signal threshold         | ±2 standard deviations (cross-sectional, within-day)        |
| Liquidity filter         | `OI_normalized > 0.5`, `Volume_normalized > 0.5`, `DTE > 5` |
| Target variable          | `log_price = log(close + 1)`                                |

---

## Troubleshooting

**`Volume_normalized` is all NaN after preprocess.py**  
The `TRADED_QUA` column fix is already in `preprocess.py`. Re-run `preprocess.py` to regenerate the parquet.

**R² > 0.98 on test set**  
Likely overfitting or feature leakage. Check that `strike` (raw), `log_price`, `Date`, and `close` are not in `FEATURE_COLS`.

**Z-scores all near zero / no signals**  
Model is over-fitting the price surface. Increase regularization: raise `min_child_weight` and `gamma`, then re-run `train.py`.

**Z-score vs moneyness shows a curve (U-shape)**  
Model is not capturing the volatility smile. Ensure `moneyness_sq` is in `FEATURE_COLS`.

**`retrain.py` skips a month**  
Requires at least 5,000 training rows before a prediction month. First 7 months of data are used as the initial training window.

**SHAP plots not generated**  
Install SHAP: `pip install shap`, then re-run `accuracy_analysis.py`.
