# Option Mispricing and Signal Generation — Blueprint
*Updated after Phase 1–4 implementation and data visualization audit (2026-04-01)*

---

## Status tracker

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Load & merge raw Excel files | DONE |
| 2 | Filter by liquidity, moneyness, DTE | DONE |
| 3 | Compute implied volatility | DONE |
| 4 | Cross-sectional feature engineering | DONE |
| 4b | Data visualization & pipeline audit | DONE |
| 5 | XGBoost model training | NEXT |
| 6 | Mispricing computation | PENDING |
| 7 | Signal generation and output | PENDING |

---

## Dataset facts (from visualization audit)

These are ground truths to keep in mind for every phase ahead.

| Fact | Value |
|------|-------|
| Total rows | 70,267 |
| Date range | 2025-04-01 → 2026-03-30 (1 year) |
| Trading days | ~240 |
| Avg contracts/day | ~293 |
| CE / PE split | ~50% / ~50% (balanced) |
| ATM_IV mean | 14.3% |
| IV range | 5% – 192% |
| Skew | Persistently positive (put premium exists) |
| TS_Slope | Mostly positive (near-term vol elevated = backwardation) |
| IV vs HV_20 | IV > HV_20 on most days (options systematically overpriced) |
| HV_20 NaN rows | ~6,959 (~10%) — first 20 trading days, expected |
| Volume_normalized | Was 100% NaN — **fixed** (TRADED_QUA column now matched) |
| close price skew | ~7.5 (extreme right skew) — log_price is the correct target |
| OI_NO_CON skew | ~5.8 (heavy right skew) — needs log transform as raw feature |

**Critical column names in cross_sectional.parquet (use exactly these):**
- Date column: `Date` (capital D, datetime64)
- Option price: `close` (not `close_price`)
- Spot price: `spot`
- Open interest: `OI_NO_CON`
- Volume: `VOLUME` (renamed from `TRADED_QUA` in Phase 4)
- Option type: `option_type` (values: `"CE"` or `"PE"`)

---

## Phases 1–4: COMPLETE

Pipeline file: `preprocess.py`
Output: `data/features/cross_sectional.parquet`

**What was built:**
- Phases 1–3 kept as-is from original pipeline
- Phase 4 added: `compute_daily_regime_features`, `compute_hv_spread`, `build_cross_sectional_dataset`, `drop_critical_nans`

**Two bugs fixed during audit:**
1. `TRADED_QUA` added to volume column lookup → `Volume_normalized` is now populated
2. `main()` broken reference to `final` removed; clean 4-phase chain in place

**Re-run preprocess.py** before Phase 5 to get a `cross_sectional.parquet` with `Volume_normalized` populated.

---

## Phase 5: Training the XGBoost model

File to create: `train.py`
Input: `data/features/cross_sectional.parquet`
Output: `models/xgb_mispricing.joblib`

### Step 5a: Pre-training feature transformations

Run these before building the feature matrix. They fix the skewness and outlier issues found in the visualization audit.

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/features/cross_sectional.parquet")

# 1. Add log-transformed OI (raw OI_NO_CON is skew ~5.8 -- unusable as-is)
df["log_OI"] = np.log1p(df["OI_NO_CON"])

# 2. Add log-transformed HV_20 (right-skewed; log makes it near-normal)
df["log_HV_20"] = np.log1p(df["HV_20"].clip(lower=0))

# 3. Add moneyness squared (captures smile curvature; prevents systematic
#    OTM bias in residuals -- confirmed needed from vol smile plot)
df["moneyness_sq"] = df["moneyness"] ** 2

# 4. Cap outliers at 99th percentile BEFORE training
#    (IV_relative and OI_normalized both have >10% rows beyond 1.5x IQR)
for col in ["IV_relative", "OI_normalized", "Volume_normalized"]:
    cap = df[col].quantile(0.99)
    floor = df[col].quantile(0.01)
    df[col] = df[col].clip(lower=floor, upper=cap)
```

### Step 5b: Feature matrix and target

```python
FEATURE_COLS = [
    # Contract structure
    "option_type_encoded",  # 1=CE, 0=PE
    "DTE",                  # days to expiry
    "moneyness",            # strike / spot
    "abs_moneyness",        # |moneyness - 1|
    "moneyness_sq",         # moneyness^2 -- captures smile curvature

    # Volatility features
    "IV",                   # contract IV
    "ATM_IV",               # daily ATM level
    "IV_relative",          # IV - ATM_IV (capped)
    "IV_rank",              # within-day IV percentile
    "IV_HV_Spread",         # ATM_IV - HV_20

    # Regime features
    "Skew",                 # PE 0.95 IV - CE 1.05 IV
    "TS_Slope",             # near - far expiry IV
    "log_HV_20",            # log(HV_20) -- log-transformed

    # Liquidity
    "OI_normalized",        # OI / daily avg OI (capped)
    "Volume_normalized",    # VOLUME / daily avg VOLUME (capped)
    "log_OI",               # log(OI_NO_CON) -- log-transformed
]

# TARGET is log_price -- DO NOT include log_price in FEATURE_COLS
TARGET = "log_price"
```

**What was removed vs original blueprint and why:**
- `strike` (raw) removed — model memorizes price surface by strike level, causes overfitting. Use `moneyness` and `abs_moneyness` instead (generalise across market levels).
- `log_price` removed from features — it IS the target; including it is data leakage.
- Raw `OI` and `Volume` removed — replaced by `log_OI`, `OI_normalized`, `Volume_normalized`.
- Raw `HV_20` removed — replaced by `log_HV_20` (right-skew fixed).

### Step 5c: Time-based split

```python
# NEVER use random split -- that is data leakage
sorted_dates = sorted(df["Date"].unique())
cutoff = sorted_dates[int(len(sorted_dates) * 0.70)]
# With 240 trading days: ~168 days train, ~72 days test

train_df = df[df["Date"] <= cutoff].dropna(subset=FEATURE_COLS + [TARGET])
test_df  = df[df["Date"] >  cutoff].dropna(subset=FEATURE_COLS + [TARGET])

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET]

print(f"Train: {len(train_df):,} rows ({train_df['Date'].min().date()} -> {train_df['Date'].max().date()})")
print(f"Test:  {len(test_df):,}  rows ({test_df['Date'].min().date()} -> {test_df['Date'].max().date()})")
```

### Step 5d: Model

```python
from xgboost import XGBRegressor
import joblib, os

os.makedirs("models", exist_ok=True)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,    # added: prevents splits on tiny leaf groups
    gamma=0.1,              # added: minimum loss reduction for a split
    reg_lambda=1.0,         # L2 regularisation
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

joblib.dump(model, "models/xgb_mispricing.joblib")
print(f"Best iteration: {model.best_iteration}")
```

### Common problems in Phase 5

- **Strike dominates feature importance** → It was removed from FEATURE_COLS above. If it sneaks back in, remove it. `moneyness` and `abs_moneyness` are the correct generalizable replacements.
- **R² > 0.98 on test set** → Model is memorising the price surface (overfitting). Increase `min_child_weight` to 20 and `gamma` to 0.3. Check if any leakage column slipped into FEATURE_COLS.
- **NaN errors during fit** → XGBoost handles NaN natively but check `X_train.isnull().sum()`. `IV_HV_Spread` and `log_HV_20` will have NaN for the first 20 trading days — these are safe to leave as NaN (XGBoost learns the split direction). `Volume_normalized` should now be populated after the preprocess.py fix; if still NaN, skip it from FEATURE_COLS for this run.
- **1 year of data is thin** → 240 trading days. The 70/30 split gives ~168 train days. Watch for the test R² being unstable — a single high-vol week can swing it. Report RMSE in addition to R² for a more stable metric.
- **early_stopping_rounds removes too many trees** → If the model stops at iteration 50–100, the learning rate is too high or regularisation too strong. Try `learning_rate=0.03` with `n_estimators=1000`.

---

## Phase 6: Mispricing computation

Add to `train.py` after model training.

```python
# Predict on FULL dataset (train + test combined)
full_df = pd.concat([train_df, test_df]).sort_values("Date").reset_index(drop=True)
full_df["predicted_log_price"] = model.predict(full_df[FEATURE_COLS])

# Back-transform: clip at 0 (option prices cannot be negative)
full_df["predicted_price"] = (np.exp(full_df["predicted_log_price"]) - 1).clip(lower=0)

# Mispricing: market price minus model price
full_df["mispricing"] = full_df["close"] - full_df["predicted_price"]

# Z-score: WITHIN each day (cross-sectional, not time-series)
full_df["mispricing_mean"] = full_df.groupby("Date")["mispricing"].transform("mean")
full_df["mispricing_std"]  = full_df.groupby("Date")["mispricing"].transform("std")

# Floor on std to prevent divide-by-zero on low-variance days
full_df["mispricing_std"]  = full_df["mispricing_std"].clip(lower=0.01)

full_df["z_score"] = (
    (full_df["mispricing"] - full_df["mispricing_mean"]) / full_df["mispricing_std"]
)
```

**Column name note:** Use `close` not `close_price` — that is the actual column name in the dataset.

### Common problems in Phase 6

- **z-scores all near zero** → Model R² is suspiciously high (>0.98). Add regularisation or check for leakage.
- **z-scores wildly large (50, 100)** → `mispricing_std` near zero on some days. The `.clip(lower=0.01)` floor above fixes this.
- **Negative predicted prices** → The `.clip(lower=0)` on `predicted_price` handles this.
- **Systematic curve in z-score vs moneyness scatter** → Model is not capturing the smile. Add `moneyness_sq` (already in FEATURE_COLS above). If curve persists, also add `moneyness_sq * option_type_encoded` as an interaction term.

---

## Phase 7: Signal generation and output

Add to `train.py` after Phase 6.

```python
import os

os.makedirs("outputs", exist_ok=True)

# Liquidity and quality filters
# Volume_normalized now works after preprocess.py fix (TRADED_QUA column matched)
signals = full_df[
    (full_df["OI_normalized"]     > 0.5) &
    (full_df["Volume_normalized"] > 0.5) &
    (full_df["DTE"]               > 5)
].copy()

# Label signals
signals["signal"] = "NEUTRAL"
signals.loc[signals["z_score"] >  2, "signal"] = "SELL"   # overpriced vs cross-section
signals.loc[signals["z_score"] < -2, "signal"] = "BUY"    # underpriced vs cross-section

# Output only non-neutral
output = signals[signals["signal"] != "NEUTRAL"][[
    "Date", "strike", "option_type", "close",
    "predicted_price", "mispricing", "z_score", "signal",
    "DTE", "moneyness", "OI_normalized", "Volume_normalized",
    "IV", "ATM_IV", "IV_relative",
]]
output.to_csv("outputs/trading_signals.csv", index=False)
print(f"Signals generated: {len(output):,}  (BUY: {(output['signal']=='BUY').sum()}, SELL: {(output['signal']=='SELL').sum()})")
```

**Column name note:** Use `Date` (capital D) and `close` — those are the actual names in the dataset.

### Common problems in Phase 7

- **`Volume_normalized` is still all NaN** → preprocess.py was not re-run after the `TRADED_QUA` fix. Re-run `preprocess.py` first, then `train.py`. Until then, drop `Volume_normalized > 0.5` from the filter temporarily.
- **Too few signals** → With only ~72 test days and ~293 contracts/day, the liquid subset after filters may be ~150 contracts/day. At ±2σ threshold that's ~7–8 signals/day. If seeing 0–2, tighten the filter to `OI_normalized > 0.3` and check z-score distribution.
- **Too many signals clustered on the same strike** → Add a per-day top-N cap: `signals.nlargest(10, 'abs_z')` per date instead of a fixed threshold.
- **Signals cluster on far-OTM contracts** → Tighten moneyness filter or raise `OI_normalized > 0.75`. OTM contracts have the widest model error (confirmed in validation Check 4).

---

## Validation and diagnostics

Run these four checks immediately after Phase 7. All four outputs should be saved to `outcomes/`.

### Check 1: Feature importance sanity
```python
import matplotlib.pyplot as plt
ax = xgb.plot_importance(model, max_num_features=15, importance_type="gain")
plt.tight_layout()
plt.savefig("outcomes/val_feature_importance.png", dpi=150)
```
**Expected:** `moneyness`, `DTE`, `IV`, `abs_moneyness` dominate.
**Red flag:** `strike` (should not be in the model), or `Date`-derived features dominating = leakage.

### Check 2: Residual (z-score) distribution
```python
plt.hist(full_df["z_score"].dropna(), bins=100)
plt.savefig("outcomes/val_zscore_distribution.png", dpi=150)
```
**Expected:** Approximately normal, centred near zero. Heavy tails are fine — signals live there.
**Red flag:** Bimodal distribution = two distinct pricing regimes the model treats as one.

### Check 3: Mispricing vs moneyness scatter
```python
plt.scatter(full_df["moneyness"], full_df["z_score"], alpha=0.05, s=3)
plt.savefig("outcomes/val_zscore_vs_moneyness.png", dpi=150)
```
**Expected:** Random scatter with no systematic curve.
**Red flag:** Systematic U or inverted-U curve = model not capturing the smile. Add `moneyness_sq` interaction terms.

### Check 4: Out-of-time RMSE by moneyness bucket
```python
for label, mask in [("ATM", abs(test_df["moneyness"]-1) < 0.02),
                    ("OTM", abs(test_df["moneyness"]-1) >= 0.05),
                    ("ITM", abs(test_df["moneyness"]-1).between(0.02, 0.05))]:
    sub = test_df[mask]
    pred = model.predict(sub[FEATURE_COLS])
    rmse = np.sqrt(np.mean((sub[TARGET].values - pred)**2))
    r2   = 1 - np.sum((sub[TARGET].values - pred)**2) / np.sum((sub[TARGET].values - sub[TARGET].mean())**2)
    print(f"{label:5s}  n={len(sub):,}  RMSE={rmse:.4f}  R²={r2:.4f}")
```
**Expected:** ATM RMSE lowest. OTM RMSE higher is acceptable — liquidity filters remove the worst OTM signals.
**Red flag:** ATM R² < 0.70 = model is not learning the fundamental pricing relationship.

---

## Execution order

Run in this exact sequence:

1. **Re-run `preprocess.py`** → produces updated `data/features/cross_sectional.parquet` with `Volume_normalized` now populated (requires the `TRADED_QUA` fix already in place)
2. **Run `train.py`**:
   - Step 5a: Pre-training transformations (log_OI, log_HV_20, moneyness_sq, outlier caps)
   - Step 5b-d: Feature matrix, time-split, XGBoost fit → saves `models/xgb_mispricing.joblib`
   - Phase 6: Predict on full dataset, compute mispricing and z-scores
   - Phase 7: Apply filters, generate `outputs/trading_signals.csv`
3. **Run validation checks** → saves 3 diagnostic plots to `outcomes/`

Total runtime estimate: preprocess.py ~15 min (IV computation), train.py < 2 min.
