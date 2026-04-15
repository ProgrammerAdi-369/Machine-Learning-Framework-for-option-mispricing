# Claude Session Memory — Option Mispricing Pipeline

**Read this first at the start of every new session.**

This document records the history of what has been built, problems solved, decisions made, and notes for the next Claude session. Update it whenever significant work is done.

---

## Project Summary

A BANKNIFTY options mispricing detection pipeline. Ingests raw NSE monthly Excel files, engineers 16 volatility/liquidity features, trains XGBoost to predict fair log-price, computes daily cross-sectional z-scores, emits BUY/SELL signals. Walk-forward retraining adapts to vol regime shifts. Streamlit dashboard for daily review. Daily inference pipeline for ongoing use after market close.

**Owner:** Aditya Mahamuni  
**Language:** Python 3.10+  
**Key data:** BANKNIFTY NSE options, Apr 2025 – Mar 2026 (~70,000 rows)

---

## What Has Been Built (Complete Inventory)

### Core Pipeline Scripts
| Script | Status | Purpose |
|--------|--------|---------|
| `preprocess.py` | DONE | Phases 1–4: load Excel, filter, compute IV (parallelised), build cross-sectional features |
| `train.py` | DONE | Phases 5–7: XGBoost training, mispricing z-scores, signal generation; saves `clip_bounds.json` |
| `accuracy_analysis.py` | DONE | 7-layer accuracy diagnostic report |
| `retrain.py` | DONE | Walk-forward retraining (Fix 1–4), IV-as-target model |
| `data_visualization.py` | DONE | 16 exploratory data plots |

### Daily Inference Layer
| Script | Status | Purpose |
|--------|--------|---------|
| `daily_run.py` | DONE | Single entry point for daily inference; orchestrates Steps A–G |
| `daily_features.py` | DONE | Single-day feature engineering (IV, HV_20 from history, liquidity normalisation) |
| `option_data_formating.py` | DONE | Cleans/formats raw NSE daily CSV; refactored to callable function |

### Dashboard
| File | Status | Purpose |
|------|--------|---------|
| `dashboard/app.py` | DONE | Streamlit app entry point |
| `dashboard/data_loader.py` | DONE | Cached parquet/CSV loaders with column normalisation |
| `dashboard/components/kpi_cards.py` | DONE | KPI metric cards |
| `dashboard/components/signals_table.py` | DONE | Top 10 signals table |
| `dashboard/components/zscore_chart.py` | DONE | Z-score histogram |
| `dashboard/components/performance_chart.py` | DONE | Monthly R² trend chart |
| `dashboard/utils.py` | DONE | Shared formatters |

### Documentation
| File | Purpose |
|------|---------|
| `README.md` | Professional developer README — concise, links to docs |
| `DOCUMENTATION.md` | Full technical reference — all details |
| `QuickStart.md` | Step-by-step run guide — historical + daily pipeline + dashboard |
| `CLAUDE_MEMORY.md` | This file — session history and notes for Claude |
| `Blueprint.md` | Original phased implementation plan (now historical reference) |
| `DAILY_PIPELINE_AND_DASHBOARD_PLAN.md` | Original daily pipeline design spec (now historical reference) |

---

## Key Decisions and Why

### Why log-price as the model target
Raw option prices have skewness ~7.5. Log-transforming normalises the distribution and stabilises variance across strikes and DTE levels. Without this, RMSE is dominated by expensive deep ITM contracts.

### Why no random train/test split
Random splits leak future data into training (e.g., the model learns from November rows in March context). All splits are strictly time-based: first 70% of unique dates = train.

### Why `strike` (raw) was removed from features
Including raw strike caused the model to memorise the price surface by absolute strike level (e.g., "43,000 strike always costs X"). This produces R² > 0.98 via overfitting. `moneyness` and `abs_moneyness` are the generalisable replacements.

### Why cross-sectional z-scores instead of time-series z-scores
Time-series z-scores compare a contract to its own history, which doesn't control for market-wide vol moves. Cross-sectional z-scores compare each contract to all other contracts *on the same day*, isolating structural mispricings from directional market moves.

### Why `moneyness_sq` is in FEATURE_COLS
The volatility smile is convex — OTM contracts on both sides have higher IV than ATM. Without `moneyness_sq`, the model produces a systematic U-shaped residual pattern vs moneyness. Adding it flattens this bias.

### Why clip bounds are saved to `clip_bounds.json`
Daily inference operates on a single day of data. Recomputing outlier clip bounds on 200-300 rows would give completely different thresholds than the training set. The bounds must be the same values used at training time. `train.py` saves them; `daily_run.py` loads and applies them.

### Why walk-forward models are preferred over static model
The static model was trained on 2025 data. In early 2026, BANKNIFTY's vol regime changed, causing R² to drop from 0.95 → 0.88 and systematic positive bias. Walk-forward models (Fix 1+2) are trained fresh each month on all prior data, so they adapt. Always use `wf_trading_signals.csv` in production.

### Why Fix 4 (rolling bias removal) was needed
Even within the walk-forward framework, the model accumulates a short-term bias drift over days when the market is trending. Subtracting a 5-day rolling mean of mispricing before computing z-scores removes this drift and raises z-score std from ~0.42 → ~0.85, producing healthier signal variance.

---

## Critical Bugs Fixed (With Cause and Fix)

### Bug 1 — `Volume_normalized` was 100% NaN
**Cause:** `preprocess.py` was looking for a column named `VOLUME` or similar, but the NSE Excel files use the column name `TRADED_QUA`.  
**Fix:** Added `TRADED_QUA` to the column lookup in Phase 4 of `preprocess.py`.  
**When found:** During data visualisation audit (Phase 4b).

### Bug 2 — `main()` had a broken reference to `final`
**Cause:** During refactoring, a variable named `final` was removed but still referenced in `main()`.  
**Fix:** Replaced with the correct 4-phase chain referencing the actual output variable.  
**When found:** During Phase 4 audit.

### Bug 3 — NSE FEB26 file had integer DDMMYY dates
**Cause:** One monthly file used a 6-digit integer date format (e.g., `280226` for Feb 28, 2026) instead of the usual string or Excel serial.  
**Fix:** Added a new branch to `fix_date_column()` in `preprocess.py` that detects and parses the DDMMYY integer format.  
**When found:** During initial multi-format date parsing work.

### Bug 4 — `option_data_formating.py` had hardcoded absolute Windows paths
**Cause:** Script was copied from a different project with absolute paths.  
**Fix:** Replaced with `pathlib`-based project-relative paths.  
**When found:** During daily pipeline implementation.

### Bug 5 — `option_data_formating.py` filename parser was for wrong format
**Cause:** Old parser split on `-` and extracted `parts[3]`, designed for a different naming convention. Actual files are `option_chain_BANKNIFTY-28-Apr-2026.csv`.  
**Fix:** New parser strips `option_chain_` prefix, then splits on the first `-` to get `BANKNIFTY` and `28-Apr-2026` separately.  
**When found:** During daily pipeline implementation.

### Bug 6 — Dashboard: Z-Score Distribution chart missing in Walk-Forward mode
**Cause:** `dashboard/components/zscore_chart.py` hardcoded `"z_score"` as the column name. In Walk-Forward mode, the correct column is `"wf_z_score"`. When the WF column wasn't found (or day_pred was empty), the fallback info message showed instead of the histogram.  
**Fix:** `zscore_chart.py` now accepts `model_key` and selects `wf_z_score` for WF mode, `z_score` for static, with graceful fallback. `app.py` passes `model_key` through. A subtitle was added showing which model's z-scores are displayed plus mean/std stats.  
**When found:** 2026-04-14, after dashboard was deployed.

### Bug 7 — Dashboard: Model Performance (Monthly R²) chart blank — axes visible, no line
**Cause:** `data_loader.load_monthly_r2()` computed `r2_score(close_price, predicted_price)` — R² between **raw rupee prices** (₹). Option prices have skewness ~7.5; a handful of expensive contracts dominate SS_tot. Even small percentage errors on those contracts inflate SS_res, pushing monthly R² in price space well below 0.5 (sometimes negative in 2026 regime-shift months). The chart's y-axis was fixed to `[0.5, 1.0]`, so all data points fell outside the visible range.  
**Fix:** `load_monthly_r2` now uses `log_price` vs `predicted_log_price` (static) or `wf_predicted_log_price` (WF) — the same log-price space the model was trained in. R² in log space is consistently ~0.85–0.98, well within the visible chart area. For WF mode, uses `wf_predicted_log_price` so the chart reflects actual WF performance.  
**When found:** 2026-04-14, after dashboard was deployed.

### Bug 8 — Dashboard: Top Signals table blank in Walk-Forward mode
**Cause:** `signals_table.py` looked for `"predicted_price"` and `"z_score"` — static signal column names. WF signals use `"wf_predicted_price"` and `"wf_z_score"`. Both columns were silently dropped, leaving a nearly empty table with no sort order.  
**Fix:** `signals_table.py` now normalises column names at the top of the function — renames `wf_predicted_price` → `predicted_price` and `wf_z_score` → `z_score` if the WF versions are present and static versions are absent.  
**When found:** 2026-04-14, as part of dashboard bug fix session.

---

## Data Facts (Ground Truth)

Do NOT compute these from scratch — use as reference:

| Fact | Value |
|------|-------|
| Total rows in `cross_sectional.parquet` | ~70,267 |
| Date range | 2025-04-01 → 2026-03-30 |
| Trading days | ~240 |
| Avg contracts/day | ~293 |
| CE/PE split | ~50/50 |
| ATM_IV mean | ~14.3% |
| IV range | 5% – 192% |
| OI_NO_CON skew | ~5.8 (needs log transform) |
| close price skew | ~7.5 (use log_price as target) |
| HV_20 NaN rows | ~6,959 (~10%) — first 20 trading days, expected |

**Column naming rules (always use these exact names):**
- Date: `Date` (capital D)
- Option price: `close` (never `close_price` in `cross_sectional.parquet`)
- Volume: `VOLUME` (renamed from `TRADED_QUA`)
- OI: `OI_NO_CON`
- Option type: `option_type` (values `"CE"` or `"PE"`)

---

## Model Performance (Static Model, As Of March 2026)

| Metric | Value |
|--------|-------|
| Train R² | ~0.975 |
| Test R² | ~0.952 |
| Test RMSE | ~0.44 (log units) |
| Price RMSE | ~₹5.84 |
| Price MAE | ~₹4.13 |
| MAPE | ~7.2% |
| R² gap | ~0.023 |
| Z-score std (static) | ~0.42 |
| Z-score std (walk-forward after Fix 4) | ~0.85 |

---

## File Paths That Matter

```
data/features/cross_sectional.parquet    ← main ML input, do not delete
models/xgb_mispricing.joblib             ← static model
models/clip_bounds.json                  ← outlier bounds, required for daily_run.py
models/wf_model_*.joblib                 ← walk-forward models, prefer latest
outputs/wf_trading_signals.csv           ← USE THIS for downstream analysis
outputs/wf_predictions.parquet           ← USE THIS for downstream analysis
data/option_chain_raw/                   ← drop daily CSVs here
data/option_chain_processed/             ← formatted CSVs written here
outputs/daily/                           ← per-day inference results
```

---

## What Should Come Next (Suggested Future Work)

1. **Expand to new months** — as each new month's Excel file arrives, add to `BANKNIFTY/` and re-run `preprocess.py` + `retrain.py` to get an updated walk-forward model.
2. **Backtesting with transaction costs** — build a P&L simulation layer on top of `wf_trading_signals.csv` accounting for bid-ask spreads, STT, and brokerage.
3. **Live data integration** — replace the manual CSV drop workflow with a direct NSE data API or broker feed.
4. **Dashboard Panel 5** — the `DAILY_PIPELINE_AND_DASHBOARD_PLAN.md` specifies a daily results panel for the dashboard, showing only today's `daily_run.py` output. The component (`dashboard/components/daily_panel.py`) was not yet implemented.
5. **Alert system** — email or notification when `daily_run.py` generates signals above a confidence threshold.
6. **Multi-asset expansion** — the pipeline architecture is instrument-agnostic; extending to NIFTY50 options requires new Excel files and confirming column names match.

---

## Notes for Next Claude Session

- **Always read this file first before reading any code.**
- The historical pipeline (Steps 1–4) is complete and working. Focus is on the daily pipeline and dashboard improvements.
- `Blueprint.md` and `DAILY_PIPELINE_AND_DASHBOARD_PLAN.md` are historical planning documents. They contain detailed code snippets and design specs that are still useful reference but may not reflect the final implementation exactly.
- When adding features or modifying `train.py`, always check that the same transformation is replicated in `daily_features.py` and `daily_run.py` — the three must stay in sync.
- The `clip_bounds.json` coupling between `train.py` and `daily_run.py` is intentional and critical. Never let `daily_run.py` recompute clip bounds from single-day data.
- Do not include `strike` (raw), `Date`, `close`, or `log_price` in `FEATURE_COLS`. These cause leakage or overfitting.
- Walk-forward models are named `wf_model_<YYYY-MM>.joblib`. The model for month M was trained on all data before month M — so `wf_model_2026-03.joblib` was trained on everything before March 2026 and predicts March 2026.
- Dashboard z-score chart uses `wf_z_score` for WF mode and `z_score` for static — never mix them. Both columns live in `wf_predictions.parquet`; only `z_score` lives in `full_predictions.parquet`.
- Dashboard monthly R² is computed in **log-price space** (`log_price` vs `predicted_log_price` / `wf_predicted_log_price`). Do not revert to raw price space — R² in ₹ space is unreliable for a skewed price distribution and will render outside the [0.5, 1.0] y-axis range.
- `signals_table.py` normalises WF column names (`wf_z_score` → `z_score`, `wf_predicted_price` → `predicted_price`) at render time. Do not add model-specific branches elsewhere in the table logic.

---

## Session History

| Date | What was done |
|------|---------------|
| 2026-04-01 | Phases 1–4 complete (preprocess.py). Data visualisation audit done. Volume_normalized NaN bug fixed. |
| 2026-04-01 | Phase 5–7 complete (train.py). XGBoost training, z-score computation, signal generation. |
| 2026-04-01 | accuracy_analysis.py complete. 7-layer diagnostic report. |
| 2026-04-01 | retrain.py complete. Walk-forward retraining (Fix 1–4), IV-as-target model. |
| 2026-04-XX | Daily pipeline complete. daily_run.py, daily_features.py, option_data_formating.py refactored. |
| 2026-04-XX | Dashboard complete. Streamlit app with 4 panels, walk-forward/static model toggle. |
| 2026-04-14 | Documentation reorganised. README rewritten (professional). DOCUMENTATION.md created (full detail). QuickStart.md updated (daily pipeline + dashboard). CLAUDE_MEMORY.md created. |
| 2026-04-14 | Dashboard bugs fixed: Z-Score chart now uses correct z-score column per model (wf_z_score / z_score). Monthly R² chart fixed to compute R² in log-price space (was raw ₹ space — values fell below [0.5,1.0] y-axis). Top Signals table fixed to handle WF column names. |
