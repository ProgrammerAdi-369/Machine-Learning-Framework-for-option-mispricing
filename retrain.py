"""
retrain.py  --  Model fixes as specified in fix.md

Fix 4  (run first): recalibrate_zscores    -- immediate band-aid on existing predictions
Fix 1+2             walk_forward_retrain   -- core architectural fix + regularization
Fix 3               IV-as-target model     -- regime-robust alternative target

Reads:  outputs/full_predictions.parquet
Writes: outputs/wf_predictions.parquet    (walk-forward predictions + z-scores)
        outputs/wf_trading_signals.csv    (signals from walk-forward model)
        outputs/iv_predictions.parquet    (Fix 3 IV-target predictions)
        models/wf_model_<month>.joblib    (one model per month)
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

warnings.filterwarnings("ignore")
os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------------------
# Load and normalise column names
# full_predictions.parquet has 'Date' and 'close'; fix.md expects 'date' and 'close_price'
# ---------------------------------------------------------------------------
print("Loading full_predictions.parquet ...")
df = pd.read_parquet("outputs/full_predictions.parquet")
df = df.rename(columns={"Date": "date", "close": "close_price"})
df = df.sort_values("date").reset_index(drop=True)

# Rebuild time split so we can report on the same horizon
sorted_dates = sorted(df["date"].unique())
cutoff = sorted_dates[int(len(sorted_dates) * 0.70)]
print(f"Dataset: {df['date'].min().date()} to {df['date'].max().date()}  |  "
      f"Cutoff: {cutoff.date()}")

# Feature set actually used in train.py (not the stale list in fix.md)
FEATURE_COLS = [
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness", "moneyness_sq",
    "IV", "ATM_IV", "IV_relative", "IV_rank", "IV_HV_Spread",
    "Skew", "TS_Slope", "log_HV_20",
    "OI_normalized", "Volume_normalized", "log_OI",
]
TARGET = "log_price"

# ---------------------------------------------------------------------------
# Fix 4 -- Recalibrate z-scores (band-aid on existing static predictions)
# Removes rolling 5-day bias before within-day normalisation
# ---------------------------------------------------------------------------
def recalibrate_zscores(data):
    """
    Removes rolling model bias before computing z-scores so that systematic
    under/over-prediction does not crush within-day variance to near-zero.
    """
    data = data.copy().sort_values("date")

    # Daily mean residual (bias per day)
    daily_bias = (
        data.groupby("date")
        .apply(lambda g: (g["log_price"] - g["predicted_log_price"]).mean())
        .reset_index()
    )
    daily_bias.columns = ["date", "daily_bias"]

    # 5-day rolling average of that bias
    daily_bias["rolling_bias"] = daily_bias["daily_bias"].rolling(5, min_periods=1).mean()

    data = data.merge(daily_bias[["date", "rolling_bias"]], on="date", how="left")

    # Subtract rolling bias from residual before z-scoring
    data["corrected_residual"] = (
        (data["log_price"] - data["predicted_log_price"]) - data["rolling_bias"]
    )

    # Cross-sectional z-score on corrected residual
    data["corr_misp_mean"] = data.groupby("date")["corrected_residual"].transform("mean")
    data["corr_misp_std"]  = (
        data.groupby("date")["corrected_residual"].transform("std").clip(lower=0.01)
    )
    data["corrected_z_score"] = (
        (data["corrected_residual"] - data["corr_misp_mean"]) / data["corr_misp_std"]
    )

    liquid = data[
        (data["OI_normalized"]     > 0.5) &
        (data["Volume_normalized"] > 0.5) &
        (data["DTE"]               > 5)
    ]
    print("\n--- Fix 4: Z-score recalibration ---")
    print(f"  Original  z-score std:  {liquid['z_score'].std():.3f}  "
          f"skew: {liquid['z_score'].skew():.3f}")
    print(f"  Corrected z-score std:  {liquid['corrected_z_score'].std():.3f}  "
          f"skew: {liquid['corrected_z_score'].skew():.3f}")
    sig_old = (abs(liquid["z_score"]) > 2).sum()
    sig_new = (abs(liquid["corrected_z_score"]) > 2).sum()
    print(f"  Signals before: {sig_old}  |  Signals after: {sig_new}")

    return data


df = recalibrate_zscores(df)

# ---------------------------------------------------------------------------
# Fix 1 + Fix 2 -- Walk-forward retraining with regularized hyperparameters
# ---------------------------------------------------------------------------
def walk_forward_retrain(data, feature_cols, target, initial_train_months=7):
    """
    Trains a fresh regularized model each month, always on all data before
    that month (expanding window). Returns df with walk-forward predictions
    and recalculated z-scores.
    """
    data = data.copy().sort_values("date")
    data["wf_predicted_log_price"] = np.nan
    data["wf_model_version"] = ""

    all_months = sorted(data["date"].dt.to_period("M").unique())
    print(f"\n--- Fix 1+2: Walk-forward retraining ---")
    print(f"  Months in dataset: {[str(m) for m in all_months]}")
    print(f"  First prediction month: {all_months[initial_train_months]}")
    print(f"  {'Month':<12}  {'Train rows':>11}  {'Pred rows':>10}  {'R2':>8}")
    print(f"  {'-'*48}")

    for i in range(initial_train_months, len(all_months)):
        predict_month = all_months[i]

        train_mask = data["date"].dt.to_period("M") < predict_month
        pred_mask  = data["date"].dt.to_period("M") == predict_month

        train_data = data[train_mask].dropna(subset=feature_cols + [target])
        pred_data  = data[pred_mask].dropna(subset=feature_cols)

        if len(train_data) < 5000 or len(pred_data) == 0:
            print(f"  {str(predict_month):<12}  SKIPPED (insufficient data)")
            continue

        # Fix 2: regularized hyperparameters to close R2 gap
        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=5,             # reduced from 6
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.7,    # reduced from 0.8
            min_child_weight=15,     # prevents overfitting small segments
            gamma=0.1,
            reg_alpha=0.05,          # L1 regularization
            reg_lambda=1.5,          # L2 regularization
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(train_data[feature_cols], train_data[target])

        preds = model.predict(pred_data[feature_cols])
        data.loc[pred_data.index, "wf_predicted_log_price"] = preds
        data.loc[pred_mask, "wf_model_version"] = str(predict_month)

        # Save per-month model
        joblib.dump(model, f"models/wf_model_{predict_month}.joblib")

        # Only compute R2 where we also have the target
        pred_with_target = pred_data[pred_data[target].notna()]
        if len(pred_with_target) > 0:
            preds_t = model.predict(pred_with_target[feature_cols])
            month_r2 = r2_score(pred_with_target[target], preds_t)
        else:
            month_r2 = float("nan")

        print(f"  {str(predict_month):<12}  {len(train_data):>11,}  "
              f"{len(pred_data):>10,}  {month_r2:>8.4f}")

    # Back-transform predictions
    data["wf_predicted_price"] = (np.exp(data["wf_predicted_log_price"]) - 1).clip(lower=0)
    data["wf_mispricing"] = data["close_price"] - data["wf_predicted_price"]

    # Cross-sectional z-score per day (higher std floor than original)
    data["wf_misp_mean"] = data.groupby("date")["wf_mispricing"].transform("mean")
    data["wf_misp_std"]  = (
        data.groupby("date")["wf_mispricing"].transform("std").clip(lower=0.5)
    )
    data["wf_z_score"] = (
        (data["wf_mispricing"] - data["wf_misp_mean"]) / data["wf_misp_std"]
    )

    return data


df = walk_forward_retrain(df, FEATURE_COLS, TARGET, initial_train_months=7)

# Compare static vs walk-forward on the test period (Dec 2025 onward)
# Only rows where WF predictions exist
test_wf = df[df["date"] > cutoff].dropna(subset=["wf_predicted_log_price", "log_price"])
print("\n  Static vs Walk-forward on test period:")
print(f"  {'Metric':<22}  {'Static':>10}  {'Walk-fwd':>10}")
print(f"  {'-'*48}")
static_r2   = r2_score(test_wf["log_price"], test_wf["predicted_log_price"])
wf_r2       = r2_score(test_wf["log_price"], test_wf["wf_predicted_log_price"])
static_bias = (test_wf["log_price"] - test_wf["predicted_log_price"]).mean()
wf_bias     = (test_wf["log_price"] - test_wf["wf_predicted_log_price"]).mean()
static_rmse = np.sqrt(mean_squared_error(test_wf["log_price"], test_wf["predicted_log_price"]))
wf_rmse     = np.sqrt(mean_squared_error(test_wf["log_price"], test_wf["wf_predicted_log_price"]))

liq_wf = df[
    (df["OI_normalized"]     > 0.5) &
    (df["Volume_normalized"] > 0.5) &
    (df["DTE"]               > 5) &
    df["wf_z_score"].notna()
]
static_zstd = test_wf["z_score"].std()
wf_zstd     = liq_wf["wf_z_score"].std()

print(f"  {'R2':<22}  {static_r2:>10.4f}  {wf_r2:>10.4f}")
print(f"  {'Bias':<22}  {static_bias:>10.4f}  {wf_bias:>10.4f}")
print(f"  {'RMSE':<22}  {static_rmse:>10.4f}  {wf_rmse:>10.4f}")
print(f"  {'Z-score std (liquid)':<22}  {static_zstd:>10.3f}  {wf_zstd:>10.3f}")

# Walk-forward monthly breakdown
print("\n  Walk-forward monthly breakdown (test months only):")
print(f"  {'Month':<12}  {'N':>7}  {'R2':>8}  {'RMSE':>8}  {'Bias':>8}")
print(f"  {'-'*52}")
wf_valid = df.dropna(subset=["wf_predicted_log_price", "log_price"])
wf_valid["ym"] = wf_valid["date"].dt.to_period("M")
for period, grp in wf_valid[wf_valid["date"] > cutoff].groupby("ym"):
    r2   = r2_score(grp["log_price"], grp["wf_predicted_log_price"])
    rmse = np.sqrt(mean_squared_error(grp["log_price"], grp["wf_predicted_log_price"]))
    bias = (grp["log_price"] - grp["wf_predicted_log_price"]).mean()
    print(f"  {str(period):<12}  {len(grp):>7,}  {r2:>8.4f}  {rmse:>8.4f}  {bias:>8.4f}")

# ---------------------------------------------------------------------------
# Fix 3 -- IV as target (Option A: regime-robust alternative)
# ---------------------------------------------------------------------------
FEATURE_COLS_IV = [
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness",
    "ATM_IV", "Skew", "TS_Slope", "IV_HV_Spread",
    "OI_normalized", "Volume_normalized", "IV_rank",
    # Note: IV itself excluded (it is the target here)
]
TARGET_IV = "IV"

print("\n--- Fix 3: IV-as-target model ---")
train_df_iv = df[df["date"] <= cutoff].dropna(subset=FEATURE_COLS_IV + [TARGET_IV])
test_df_iv  = df[df["date"] >  cutoff].dropna(subset=FEATURE_COLS_IV + [TARGET_IV])

model_iv = xgb.XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=15,
    gamma=0.1,
    reg_alpha=0.05,
    reg_lambda=1.5,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model_iv.fit(train_df_iv[FEATURE_COLS_IV], train_df_iv[TARGET_IV])
joblib.dump(model_iv, "models/xgb_iv_target.joblib")

# Fill NaN in features for full-dataset prediction using median
feature_medians = df[FEATURE_COLS_IV].median()
df["predicted_IV"] = model_iv.predict(df[FEATURE_COLS_IV].fillna(feature_medians))
df["IV_mispricing"] = df["IV"] - df["predicted_IV"]

# Cross-sectional z-score of IV mispricing within each day
df["IV_misp_mean"] = df.groupby("date")["IV_mispricing"].transform("mean")
df["IV_misp_std"]  = (
    df.groupby("date")["IV_mispricing"].transform("std").clip(lower=0.001)
)
df["IV_z_score"] = (
    (df["IV_mispricing"] - df["IV_misp_mean"]) / df["IV_misp_std"]
)

iv_test_r2 = r2_score(test_df_iv[TARGET_IV],
                      model_iv.predict(test_df_iv[FEATURE_COLS_IV]))
print(f"  IV model test R2: {iv_test_r2:.4f}")

liq_iv = df[
    (df["OI_normalized"]     > 0.5) &
    (df["Volume_normalized"] > 0.5) &
    (df["DTE"]               > 5)
]
print(f"  IV z-score std (liquid): {liq_iv['IV_z_score'].std():.3f}  "
      f"mean: {liq_iv['IV_z_score'].mean():.3f}")

# IV model monthly stability
print("  IV model monthly R2 (test period):")
iv_full_valid = df.dropna(subset=["predicted_IV", "IV"])
iv_full_valid["ym"] = iv_full_valid["date"].dt.to_period("M")
for period, grp in iv_full_valid[iv_full_valid["date"] > cutoff].groupby("ym"):
    r2   = r2_score(grp["IV"], grp["predicted_IV"])
    bias = (grp["IV"] - grp["predicted_IV"]).mean()
    print(f"    {str(period):<12}  R2={r2:.4f}  bias={bias:.4f}")

# ---------------------------------------------------------------------------
# Signals from walk-forward model
# ---------------------------------------------------------------------------
print("\n--- Walk-forward signals ---")
wf_signals = df[
    (df["OI_normalized"]     > 0.5) &
    (df["Volume_normalized"] > 0.5) &
    (df["DTE"]               > 5) &
    df["wf_z_score"].notna()
].copy()

wf_signals["signal"] = "NEUTRAL"
wf_signals.loc[wf_signals["wf_z_score"] >  2, "signal"] = "SELL"
wf_signals.loc[wf_signals["wf_z_score"] < -2, "signal"] = "BUY"

wf_output = wf_signals[wf_signals["signal"] != "NEUTRAL"][[
    "date", "strike", "option_type", "close_price",
    "wf_predicted_price", "wf_mispricing", "wf_z_score", "signal",
    "DTE", "moneyness", "OI_normalized", "Volume_normalized",
    "IV", "ATM_IV", "IV_relative",
]]
wf_output.to_csv("outputs/wf_trading_signals.csv", index=False)
print(f"  Walk-forward signals: {len(wf_output):,}  "
      f"(BUY: {(wf_output['signal']=='BUY').sum()}, "
      f"SELL: {(wf_output['signal']=='SELL').sum()})")

# ---------------------------------------------------------------------------
# Save enriched predictions parquet (adds all new columns)
# ---------------------------------------------------------------------------
df.to_parquet("outputs/wf_predictions.parquet", index=False)
print("\nSaved outputs/wf_predictions.parquet")

# ---------------------------------------------------------------------------
# Comparison plot: static vs walk-forward z-score distributions
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Fix comparison: Static vs Walk-forward vs IV-target", fontsize=13)

liquid_all = df[
    (df["OI_normalized"]     > 0.5) &
    (df["Volume_normalized"] > 0.5) &
    (df["DTE"]               > 5)
]

from scipy import stats as scipy_stats

for ax, col, label, color in [
    (axes[0], "z_score",           "Static model",      "steelblue"),
    (axes[1], "wf_z_score",        "Walk-forward model", "darkorange"),
    (axes[2], "IV_z_score",        "IV-target model",   "seagreen"),
]:
    data_col = liquid_all[col].dropna()
    ax.hist(data_col, bins=80, density=True, alpha=0.7, color=color)
    x = np.linspace(-6, 6, 300)
    ax.plot(x, scipy_stats.norm.pdf(x, 0, 1), "r-", linewidth=2, label="Std normal")
    ax.axvline( 2, color="black", linewidth=1, linestyle="--")
    ax.axvline(-2, color="black", linewidth=1, linestyle="--")
    ax.set_xlim(-6, 6)
    ax.set_title(f"{label}\nstd={data_col.std():.2f}  mean={data_col.mean():.2f}")
    ax.set_xlabel("Z-score")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("outputs/fix_comparison_zscore.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved outputs/fix_comparison_zscore.png")

# ---------------------------------------------------------------------------
# Temporal stability comparison plot: static vs walk-forward R2 by month
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Temporal stability: Static vs Walk-forward", fontsize=13)

# Build monthly metrics for static model
static_monthly = []
wf_monthly = []
for period, grp in df.dropna(subset=["log_price"]).assign(
    ym=df["date"].dt.to_period("M")
).groupby("ym"):
    if len(grp) < 100:
        continue
    static_r2_m = r2_score(grp["log_price"], grp["predicted_log_price"])
    static_monthly.append({"period": str(period), "r2": static_r2_m,
                            "period_dt": pd.to_datetime(str(period))})
    wf_grp = grp.dropna(subset=["wf_predicted_log_price"])
    if len(wf_grp) >= 50:
        wf_r2_m = r2_score(wf_grp["log_price"], wf_grp["wf_predicted_log_price"])
        wf_monthly.append({"period": str(period), "r2": wf_r2_m,
                            "period_dt": pd.to_datetime(str(period))})

static_mdf = pd.DataFrame(static_monthly)
wf_mdf     = pd.DataFrame(wf_monthly)

ax = axes[0]
ax.plot(static_mdf["period_dt"], static_mdf["r2"], "o-",
        color="steelblue", markersize=5, label="Static model")
ax.plot(wf_mdf["period_dt"], wf_mdf["r2"], "s-",
        color="darkorange", markersize=5, label="Walk-forward")
ax.axhline(0.90, color="red", linewidth=1, linestyle="--", label="R2=0.90 floor")
ax.axvline(pd.to_datetime(str(cutoff.date())), color="gray",
           linewidth=1.5, linestyle=":", label="Train/test split")
ax.set_title("Monthly R2")
ax.set_ylim(0.7, 1.01)
ax.legend(fontsize=9)

# Bias by month
static_bias_m = []
wf_bias_m = []
for period, grp in df.dropna(subset=["log_price"]).assign(
    ym=df["date"].dt.to_period("M")
).groupby("ym"):
    if len(grp) < 100:
        continue
    sb = (grp["log_price"] - grp["predicted_log_price"]).mean()
    static_bias_m.append({"period_dt": pd.to_datetime(str(period)), "bias": sb})
    wf_grp = grp.dropna(subset=["wf_predicted_log_price"])
    if len(wf_grp) >= 50:
        wb = (wf_grp["log_price"] - wf_grp["wf_predicted_log_price"]).mean()
        wf_bias_m.append({"period_dt": pd.to_datetime(str(period)), "bias": wb})

sb_df = pd.DataFrame(static_bias_m)
wb_df = pd.DataFrame(wf_bias_m)

ax = axes[1]
ax.plot(sb_df["period_dt"], sb_df["bias"], "o-",
        color="steelblue", markersize=5, label="Static model")
ax.plot(wb_df["period_dt"], wb_df["bias"], "s-",
        color="darkorange", markersize=5, label="Walk-forward")
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.axvline(pd.to_datetime(str(cutoff.date())), color="gray",
           linewidth=1.5, linestyle=":", label="Train/test split")
ax.set_title("Monthly Bias")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("outputs/fix_comparison_temporal.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved outputs/fix_comparison_temporal.png")

print("\nAll fixes applied. Summary of outputs:")
print("  outputs/wf_predictions.parquet    -- full dataset with all new columns")
print("  outputs/wf_trading_signals.csv    -- walk-forward signals")
print("  outputs/fix_comparison_zscore.png -- z-score distribution comparison")
print("  outputs/fix_comparison_temporal.png -- temporal stability comparison")
print("  models/wf_model_<month>.joblib    -- per-month walk-forward models")
print("  models/xgb_iv_target.joblib       -- IV-as-target model")
