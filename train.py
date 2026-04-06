"""
train.py — Phases 5, 6, 7 of the Option Mispricing Pipeline
Reads:  data/features/cross_sectional.parquet
Writes: models/xgb_mispricing.joblib
        outputs/trading_signals.csv
        Analysis_outcomes/val_feature_importance_<timestamp>.png
        Analysis_outcomes/val_zscore_distribution_<timestamp>.png
        Analysis_outcomes/val_zscore_vs_moneyness_<timestamp>.png
        (Check 4 RMSE by moneyness bucket printed to stdout)
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb

# ---------------------------------------------------------------------------
# Phase 5a: Load data + pre-training feature transformations
# ---------------------------------------------------------------------------
print("Loading data...")
df = pd.read_parquet("data/features/cross_sectional.parquet")
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

# 1. Log-transform OI (raw skew ~5.8 — unusable as-is)
df["log_OI"] = np.log1p(df["OI_NO_CON"])

# 2. Log-transform HV_20 (right-skewed; log makes it near-normal)
df["log_HV_20"] = np.log1p(df["HV_20"].clip(lower=0))

# 3. Moneyness squared (captures smile curvature)
df["moneyness_sq"] = df["moneyness"] ** 2

# 4. Cap outliers at 99th / 1st percentile
for col in ["IV_relative", "OI_normalized", "Volume_normalized"]:
    cap   = df[col].quantile(0.99)
    floor = df[col].quantile(0.01)
    df[col] = df[col].clip(lower=floor, upper=cap)
    print(f"  Clipped {col}: [{floor:.4f}, {cap:.4f}]")

# ---------------------------------------------------------------------------
# Phase 5b: Feature matrix and target
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # Contract structure
    "option_type_encoded",
    "DTE",
    "moneyness",
    "abs_moneyness",
    "moneyness_sq",
    # Volatility
    "IV",
    "ATM_IV",
    "IV_relative",
    "IV_rank",
    "IV_HV_Spread",
    # Regime
    "Skew",
    "TS_Slope",
    "log_HV_20",
    # Liquidity
    "OI_normalized",
    "Volume_normalized",
    "log_OI",
]

TARGET = "log_price"

# ---------------------------------------------------------------------------
# Phase 5c: Time-based split (NO random split — that is data leakage)
# ---------------------------------------------------------------------------
sorted_dates = sorted(df["Date"].unique())
cutoff = sorted_dates[int(len(sorted_dates) * 0.70)]

train_df = df[df["Date"] <= cutoff].dropna(subset=FEATURE_COLS + [TARGET])
test_df  = df[df["Date"] >  cutoff].dropna(subset=FEATURE_COLS + [TARGET])

X_train = train_df[FEATURE_COLS]
y_train = train_df[TARGET]
X_test  = test_df[FEATURE_COLS]
y_test  = test_df[TARGET]

print(f"\nTrain: {len(train_df):,} rows ({train_df['Date'].min().date()} -> {train_df['Date'].max().date()})")
print(f"Test:  {len(test_df):,} rows  ({test_df['Date'].min().date()} -> {test_df['Date'].max().date()})")

# Check for NaN in feature matrices
print(f"\nNaN in X_train:\n{X_train.isnull().sum()[X_train.isnull().sum() > 0]}")
print(f"NaN in X_test:\n{X_test.isnull().sum()[X_test.isnull().sum() > 0]}")

# ---------------------------------------------------------------------------
# Phase 5d: Train XGBoost model
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)

model = XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50,
    eval_metric="rmse",
)

print("\nTraining XGBoost model...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

joblib.dump(model, "models/xgb_mispricing.joblib")
print(f"\nModel saved. Best iteration: {model.best_iteration}")

# Quick test-set metrics
y_pred_test = model.predict(X_test)
test_rmse = np.sqrt(np.mean((y_test.values - y_pred_test) ** 2))
ss_res = np.sum((y_test.values - y_pred_test) ** 2)
ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
test_r2 = 1 - ss_res / ss_tot
print(f"Test RMSE: {test_rmse:.4f}  |  Test R²: {test_r2:.4f}")

if test_r2 > 0.98:
    print("WARNING: R² > 0.98 — possible overfitting or leakage. Check FEATURE_COLS.")

# ---------------------------------------------------------------------------
# Phase 6: Mispricing computation on FULL dataset
# ---------------------------------------------------------------------------
print("\nComputing mispricing on full dataset...")
full_df = pd.concat([train_df, test_df]).sort_values("Date").reset_index(drop=True)
full_df["predicted_log_price"] = model.predict(full_df[FEATURE_COLS])

# Back-transform (option prices cannot be negative)
full_df["predicted_price"] = (np.exp(full_df["predicted_log_price"]) - 1).clip(lower=0)

# Mispricing: market price minus model price
full_df["mispricing"] = full_df["close"] - full_df["predicted_price"]

# Z-score cross-sectionally WITHIN each day
full_df["mispricing_mean"] = full_df.groupby("Date")["mispricing"].transform("mean")
full_df["mispricing_std"]  = full_df.groupby("Date")["mispricing"].transform("std")

# Floor on std to prevent divide-by-zero on low-variance days
full_df["mispricing_std"] = full_df["mispricing_std"].clip(lower=0.01)

full_df["z_score"] = (
    (full_df["mispricing"] - full_df["mispricing_mean"]) / full_df["mispricing_std"]
)

print(f"Z-score stats: mean={full_df['z_score'].mean():.3f}  "
      f"std={full_df['z_score'].std():.3f}  "
      f"min={full_df['z_score'].min():.1f}  "
      f"max={full_df['z_score'].max():.1f}")

# Save full predictions for accuracy_analysis.py
os.makedirs("outputs", exist_ok=True)
full_df.to_parquet("outputs/full_predictions.parquet", index=False)
print("Saved outputs/full_predictions.parquet")

# ---------------------------------------------------------------------------
# Phase 7: Signal generation and output
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)

signals = full_df[
    (full_df["OI_normalized"]     > 0.5) &
    (full_df["Volume_normalized"] > 0.5) &
    (full_df["DTE"]               > 5)
].copy()

signals["signal"] = "NEUTRAL"
signals.loc[signals["z_score"] >  2, "signal"] = "SELL"
signals.loc[signals["z_score"] < -2, "signal"] = "BUY"

output = signals[signals["signal"] != "NEUTRAL"][[
    "Date", "strike", "option_type", "close",
    "predicted_price", "mispricing", "z_score", "signal",
    "DTE", "moneyness", "OI_normalized", "Volume_normalized",
    "IV", "ATM_IV", "IV_relative",
]]
output.to_csv("outputs/trading_signals.csv", index=False)
print(f"\nSignals generated: {len(output):,}  "
      f"(BUY: {(output['signal']=='BUY').sum()}, "
      f"SELL: {(output['signal']=='SELL').sum()})")

# ---------------------------------------------------------------------------
# Validation checks — all outputs to Analysis_outcomes/
# ---------------------------------------------------------------------------
os.makedirs("Analysis_outcomes", exist_ok=True)
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# Check 1: Feature importance
print("\nRunning validation checks...")
fig, ax = plt.subplots(figsize=(10, 7))
xgb.plot_importance(model, ax=ax, max_num_features=15, importance_type="gain")
plt.tight_layout()
plt.savefig(f"Analysis_outcomes/val_feature_importance_{_TS}.png", dpi=150)
plt.close()
print(f"  Saved Analysis_outcomes/val_feature_importance_{_TS}.png")

# Check 2: Z-score distribution
plt.figure(figsize=(10, 5))
plt.hist(full_df["z_score"].dropna(), bins=100, edgecolor="none")
plt.xlabel("Z-score")
plt.ylabel("Count")
plt.title("Cross-sectional Z-score distribution (full dataset)")
plt.tight_layout()
plt.savefig(f"Analysis_outcomes/val_zscore_distribution_{_TS}.png", dpi=150)
plt.close()
print(f"  Saved Analysis_outcomes/val_zscore_distribution_{_TS}.png")

# Check 3: Mispricing vs moneyness scatter
plt.figure(figsize=(10, 6))
plt.scatter(full_df["moneyness"], full_df["z_score"], alpha=0.05, s=3)
plt.xlabel("Moneyness (strike / spot)")
plt.ylabel("Z-score")
plt.title("Z-score vs Moneyness — expect random scatter (no curve)")
plt.axhline(0, color="red", linewidth=0.8, linestyle="--")
plt.tight_layout()
plt.savefig(f"Analysis_outcomes/val_zscore_vs_moneyness_{_TS}.png", dpi=150)
plt.close()
print(f"  Saved Analysis_outcomes/val_zscore_vs_moneyness_{_TS}.png")

# Check 4: Out-of-time RMSE by moneyness bucket
print("\nCheck 4: Out-of-time RMSE by moneyness bucket")
print(f"{'Bucket':6s}  {'n':>7s}  {'RMSE':>8s}  {'R²':>8s}")
print("-" * 40)
for label, mask in [
    ("ATM", abs(test_df["moneyness"] - 1) < 0.02),
    ("ITM", abs(test_df["moneyness"] - 1).between(0.02, 0.05)),
    ("OTM", abs(test_df["moneyness"] - 1) >= 0.05),
]:
    sub = test_df[mask]
    if len(sub) == 0:
        print(f"{label:6s}  {'0':>7s}  {'N/A':>8s}  {'N/A':>8s}")
        continue
    pred = model.predict(sub[FEATURE_COLS])
    rmse = np.sqrt(np.mean((sub[TARGET].values - pred) ** 2))
    ss_r = np.sum((sub[TARGET].values - pred) ** 2)
    ss_t = np.sum((sub[TARGET].values - sub[TARGET].mean()) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else float("nan")
    print(f"{label:6s}  {len(sub):>7,}  {rmse:>8.4f}  {r2:>8.4f}")

print("\nDone. All outputs written.")
