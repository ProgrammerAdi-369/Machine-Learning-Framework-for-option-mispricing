"""
accuracy_analysis.py — Model accuracy analysis for the Option Mispricing Pipeline
Implements Layers A–G as specified in Report_plan.md.

Reads:  outputs/full_predictions.parquet   (saved by train.py Phase 6)
        models/xgb_mispricing.joblib        (trained model)
Writes: Analysis_outcomes/residual_diagnostics_<timestamp>.png
        Analysis_outcomes/feature_importance_<timestamp>.png
        Analysis_outcomes/shap_analysis_<timestamp>.png      (if shap installed)
        Analysis_outcomes/shap_moneyness_<timestamp>.png     (if shap installed)
        Analysis_outcomes/signal_quality_<timestamp>.png
        Analysis_outcomes/temporal_stability_<timestamp>.png

Column name mapping (actual dataset → plan names used internally):
  Date  →  Date    (kept as-is; plan used lowercase 'date' — we normalise below)
  close →  close   (plan used 'close_price' — we normalise below)
"""

import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

warnings.filterwarnings("ignore")

os.makedirs("Analysis_outcomes", exist_ok=True)
_TS = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------------------------------------------------------------------------
# Setup — load predictions parquet
# ---------------------------------------------------------------------------
print("Loading full_predictions.parquet ...")
df = pd.read_parquet("outputs/full_predictions.parquet")

# Normalise column names to match plan expectations
# Plan uses lowercase 'date' and 'close_price'; our data has 'Date' and 'close'
df = df.rename(columns={"Date": "date", "close": "close_price"})

# Time-based split using same 70/30 cutoff as train.py
sorted_dates = sorted(df["date"].unique())
cutoff = sorted_dates[int(len(sorted_dates) * 0.70)]
train_df = df[df["date"] <= cutoff].copy()
test_df  = df[df["date"] >  cutoff].copy()

print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")
print(f"Train dates: {train_df['date'].min().date()} to {train_df['date'].max().date()}")
print(f"Test  dates: {test_df['date'].min().date()} to {test_df['date'].max().date()}")

# Load model
model = joblib.load("models/xgb_mispricing.joblib")

FEATURE_COLS = [
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness", "moneyness_sq",
    "IV", "ATM_IV", "IV_relative", "IV_rank", "IV_HV_Spread",
    "Skew", "TS_Slope", "log_HV_20",
    "OI_normalized", "Volume_normalized", "log_OI",
]

# ---------------------------------------------------------------------------
# Layer A — Global fit metrics
# ---------------------------------------------------------------------------
def compute_global_metrics(sub, label=""):
    actual_log   = sub["log_price"]
    predicted_log = sub["predicted_log_price"]
    actual_price  = sub["close_price"]
    predicted_price = sub["predicted_price"]

    rmse  = np.sqrt(mean_squared_error(actual_log, predicted_log))
    mae   = mean_absolute_error(actual_log, predicted_log)
    r2    = r2_score(actual_log, predicted_log)

    mae_price  = mean_absolute_error(actual_price, predicted_price)
    rmse_price = np.sqrt(mean_squared_error(actual_price, predicted_price))

    mask = actual_price > 1.0
    mape = (np.abs((actual_price[mask] - predicted_price[mask]) / actual_price[mask])).mean() * 100

    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Log-price RMSE : {rmse:.4f}")
    print(f"  Log-price MAE  : {mae:.4f}")
    print(f"  R²             : {r2:.4f}")
    print(f"  Price RMSE     : Rs.{rmse_price:.2f}")
    print(f"  Price MAE      : Rs.{mae_price:.2f}")
    print(f"  MAPE (price>1) : {mape:.2f}%")

    return dict(label=label, rmse=rmse, mae=mae, r2=r2,
                rmse_price=rmse_price, mae_price=mae_price, mape=mape)


print("\n\n=== LAYER A: GLOBAL FIT METRICS ===")
train_metrics = compute_global_metrics(train_df, "TRAIN SET")
test_metrics  = compute_global_metrics(test_df,  "TEST SET")

r2_gap = train_metrics["r2"] - test_metrics["r2"]
print(f"\n  R² gap (train-test): {r2_gap:.4f}")
if r2_gap > 0.05:
    print("  WARNING: gap > 0.05 suggests overfitting — add regularization")
elif r2_gap < 0.0:
    print("  NOTE: test R² > train R² is unusual — check for data issues")
else:
    print("  OK: gap is within healthy range")

# ---------------------------------------------------------------------------
# Layer B — Residual diagnostics
# ---------------------------------------------------------------------------
def plot_residual_diagnostics(sub, split_label="Test set"):
    residuals = sub["log_price"] - sub["predicted_log_price"]

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Residual diagnostics — {split_label}", fontsize=14, y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Plot 1: Residuals vs predicted
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(sub["predicted_log_price"], residuals, alpha=0.05, s=4, color="steelblue")
    ax1.axhline(0, color="red", linewidth=1)
    ax1.set_xlabel("Predicted log-price")
    ax1.set_ylabel("Residual")
    ax1.set_title("Residuals vs predicted\n(should be a flat cloud)")

    # Plot 2: Residual histogram with normal fit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(residuals, bins=100, color="steelblue", alpha=0.7, density=True)
    x = np.linspace(residuals.min(), residuals.max(), 300)
    ax2.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()),
             "r-", linewidth=2, label="Normal fit")
    ax2.set_title("Residual distribution\n(should be ~normal)")
    ax2.legend(fontsize=9)

    # Plot 3: Q-Q plot
    ax3 = fig.add_subplot(gs[0, 2])
    stats.probplot(residuals, dist="norm", plot=ax3)
    ax3.set_title("Q-Q plot\n(points on line = normal errors)")

    # Plot 4: Residuals vs moneyness (with smoothed trend)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.scatter(sub["moneyness"], residuals, alpha=0.05, s=4, color="darkorange")
    ax4.axhline(0, color="red", linewidth=1)
    moneyness_sorted = sub["moneyness"].sort_values()
    res_sorted = residuals[moneyness_sorted.index]
    window = max(1, len(res_sorted) // 200)
    trend = uniform_filter1d(res_sorted.values, size=window)
    ax4.plot(moneyness_sorted.values, trend, "r-", linewidth=2)
    ax4.set_xlabel("Moneyness")
    ax4.set_ylabel("Residual")
    ax4.set_title("Residuals vs moneyness\n(red trend should be flat)")

    # Plot 5: Residuals vs DTE
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(sub["DTE"], residuals, alpha=0.05, s=4, color="purple")
    ax5.axhline(0, color="red", linewidth=1)
    ax5.set_xlabel("DTE")
    ax5.set_title("Residuals vs DTE\n(should be flat)")

    # Plot 6: Daily mean residual over time (rolling mean)
    ax6 = fig.add_subplot(gs[1, 2])
    daily_res = sub.groupby("date").apply(
        lambda g: (g["log_price"] - g["predicted_log_price"]).mean()
    )
    daily_res.plot(ax=ax6, alpha=0.4, color="steelblue", linewidth=0.8)
    daily_res.rolling(10).mean().plot(ax=ax6, color="red", linewidth=2)
    ax6.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax6.set_title("Daily mean residual over time\n(red rolling avg should hug zero)")
    ax6.set_xlabel("")

    plt.savefig(f"Analysis_outcomes/residual_diagnostics_{_TS}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Analysis_outcomes/residual_diagnostics_{_TS}.png")

    # Statistical tests
    print("\nStatistical tests on residuals:")
    stat, p = stats.shapiro(residuals.sample(min(5000, len(residuals)), random_state=42))
    print(f"  Shapiro-Wilk normality: p={p:.4f}  "
          f"{'(normal)' if p > 0.05 else '(NOT normal — heavy tails exist)'}")
    skew = residuals.skew()
    kurt = residuals.kurtosis()
    print(f"  Skewness: {skew:.3f}  "
          f"{'OK' if abs(skew) < 0.5 else 'WARNING: skewed — model biased in one direction'}")
    print(f"  Kurtosis: {kurt:.3f}  "
          f"{'OK' if kurt < 5 else 'WARNING: fat tails — extreme errors are common'}")
    bias = residuals.mean()
    print(f"  Mean residual (bias): {bias:.4f}  "
          f"{'OK' if abs(bias) < 0.01 else 'WARNING: systematic bias exists'}")


print("\n\n=== LAYER B: RESIDUAL DIAGNOSTICS ===")
plot_residual_diagnostics(test_df, "Test set")

# ---------------------------------------------------------------------------
# Layer C — Segment-level breakdown
# ---------------------------------------------------------------------------
def segment_analysis(sub, label="Test set"):
    segments = {
        "moneyness_bucket": {
            "Deep OTM": (sub["moneyness"] < 0.93) | (sub["moneyness"] > 1.07),
            "OTM":      ((sub["moneyness"] >= 0.93) & (sub["moneyness"] < 0.97)) |
                        ((sub["moneyness"] > 1.03) & (sub["moneyness"] <= 1.07)),
            "Near ATM": (sub["moneyness"] >= 0.97) & (sub["moneyness"] <= 1.03),
        },
        "DTE_bucket": {
            "Near (1-10d)": (sub["DTE"] >= 1)  & (sub["DTE"] <= 10),
            "Mid (11-30d)": (sub["DTE"] >= 11) & (sub["DTE"] <= 30),
            "Far (31-90d)": (sub["DTE"] >= 31) & (sub["DTE"] <= 90),
        },
        "option_type": {
            "CE (Call)": sub["option_type"] == "CE",
            "PE (Put)":  sub["option_type"] == "PE",
        },
        "liquidity_tier": {
            "High OI (top 25%)": sub["OI_normalized"] >= sub["OI_normalized"].quantile(0.75),
            "Mid OI":            (sub["OI_normalized"] >= sub["OI_normalized"].quantile(0.25)) &
                                  (sub["OI_normalized"] <  sub["OI_normalized"].quantile(0.75)),
            "Low OI (bot 25%)":  sub["OI_normalized"] <  sub["OI_normalized"].quantile(0.25),
        },
    }

    print(f"\n{'='*70}")
    print(f"  SEGMENT ANALYSIS — {label}")
    print(f"{'='*70}")

    results = []
    for seg_name, buckets in segments.items():
        print(f"\n  [{seg_name}]")
        print(f"  {'Bucket':<20} {'N':>7} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'MAPE%':>8} {'Bias':>8}")
        print(f"  {'-'*67}")

        for bucket_name, mask in buckets.items():
            bucket_sub = sub[mask]
            if len(bucket_sub) < 50:
                continue
            rmse = np.sqrt(mean_squared_error(bucket_sub["log_price"], bucket_sub["predicted_log_price"]))
            mae  = mean_absolute_error(bucket_sub["log_price"], bucket_sub["predicted_log_price"])
            r2   = r2_score(bucket_sub["log_price"], bucket_sub["predicted_log_price"])
            bias = (bucket_sub["log_price"] - bucket_sub["predicted_log_price"]).mean()

            price_mask = bucket_sub["close_price"] > 1.0
            mape = (np.abs(bucket_sub.loc[price_mask, "close_price"] -
                           bucket_sub.loc[price_mask, "predicted_price"]) /
                    bucket_sub.loc[price_mask, "close_price"]).mean() * 100 \
                   if price_mask.sum() > 0 else float("nan")

            flag = ""
            if r2 < 0.85:      flag += " <-- WEAK"
            if abs(bias) > 0.05: flag += " BIASED"

            print(f"  {bucket_name:<20} {len(bucket_sub):>7,} {rmse:>8.4f} {mae:>8.4f} "
                  f"{r2:>8.4f} {mape:>8.1f} {bias:>8.4f}{flag}")

            results.append(dict(seg=seg_name, bucket=bucket_name, n=len(bucket_sub),
                                rmse=rmse, mae=mae, r2=r2, mape=mape, bias=bias))

    return pd.DataFrame(results)


print("\n\n=== LAYER C: SEGMENT ANALYSIS ===")
seg_results = segment_analysis(test_df, "Test set")

# ---------------------------------------------------------------------------
# Layer D — Feature importance (three views) + SHAP
# ---------------------------------------------------------------------------
def feature_importance_analysis(mdl, feature_cols, sub, label="Test set"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Feature importance — three views", fontsize=13)

    importance_labels = {
        "weight": "Frequency\n(how often used in splits)",
        "gain":   "Gain\n(how much each feature improves predictions)",
        "cover":  "Cover\n(how many samples each feature affects)",
    }

    for ax, imp_type in zip(axes, ["weight", "gain", "cover"]):
        scores = mdl.get_booster().get_score(importance_type=imp_type)
        scores_df = pd.DataFrame({"feature": list(scores.keys()),
                                   "score":   list(scores.values())})
        scores_df = scores_df.sort_values("score", ascending=True).tail(15)
        ax.barh(scores_df["feature"], scores_df["score"], color="steelblue", alpha=0.8)
        ax.set_title(importance_labels[imp_type], fontsize=10)
        ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(f"Analysis_outcomes/feature_importance_{_TS}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Analysis_outcomes/feature_importance_{_TS}.png")

    # SHAP analysis
    try:
        import shap
        sample = sub[feature_cols].sample(min(3000, len(sub)), random_state=42)
        explainer = shap.TreeExplainer(mdl)
        shap_values = explainer.shap_values(sample)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle("SHAP analysis", fontsize=13)

        plt.sca(axes[0])
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        axes[0].set_title("SHAP summary: feature impact on predictions")

        plt.sca(axes[1])
        shap.summary_plot(shap_values, sample, plot_type="bar", show=False, max_display=15)
        axes[1].set_title("SHAP mean absolute values")

        plt.tight_layout()
        plt.savefig(f"Analysis_outcomes/shap_analysis_{_TS}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved Analysis_outcomes/shap_analysis_{_TS}.png")

        # Moneyness SHAP interaction plot
        moneyness_idx = list(feature_cols).index("moneyness")
        moneyness_shap = shap_values[:, moneyness_idx]

        plt.figure(figsize=(8, 5))
        plt.scatter(sample["moneyness"], moneyness_shap, alpha=0.1, s=4)
        plt.axhline(0, color="red", linewidth=1)
        plt.xlabel("Moneyness")
        plt.ylabel("SHAP value (impact on log-price prediction)")
        plt.title("How moneyness drives predictions\n"
                  "(should be monotone — higher moneyness = higher CE price)")
        plt.tight_layout()
        plt.savefig(f"Analysis_outcomes/shap_moneyness_{_TS}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved Analysis_outcomes/shap_moneyness_{_TS}.png")

    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        print("  Skipping SHAP analysis — native importance plots saved above.")


print("\n\n=== LAYER D: FEATURE IMPORTANCE & SHAP ===")
feature_importance_analysis(model, FEATURE_COLS, test_df)

# ---------------------------------------------------------------------------
# Layer E — Z-score and signal quality analysis
# ---------------------------------------------------------------------------
def signal_quality_analysis(sub):
    signals_df = sub[
        (sub["Volume_normalized"] > 0.5) &
        (sub["OI_normalized"]     > 0.5) &
        (sub["DTE"]               > 5)
    ].copy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Z-score and signal quality analysis", fontsize=13)

    # Plot 1: Z-score distribution vs standard normal
    ax = axes[0, 0]
    ax.hist(signals_df["z_score"], bins=80, color="steelblue", alpha=0.7, density=True)
    x = np.linspace(-5, 5, 300)
    ax.plot(x, stats.norm.pdf(x, 0, 1), "r-", linewidth=2, label="Standard normal")
    ax.axvline( 2, color="orange", linewidth=1.5, linestyle="--", label="+2 sell threshold")
    ax.axvline(-2, color="green",  linewidth=1.5, linestyle="--", label="-2 buy threshold")
    ax.set_xlabel("Z-score")
    ax.set_title("Z-score distribution (should match normal)")
    ax.legend(fontsize=8)
    ax.set_xlim(-6, 6)

    # Plot 2: Z-score vs moneyness
    ax = axes[0, 1]
    ax.scatter(signals_df["moneyness"], signals_df["z_score"],
               alpha=0.08, s=4, color="darkorange")
    ax.axhline( 2, color="red",   linewidth=1, linestyle="--")
    ax.axhline(-2, color="green", linewidth=1, linestyle="--")
    ax.axhline( 0, color="black", linewidth=0.5)
    ax.set_xlabel("Moneyness")
    ax.set_ylabel("Z-score")
    ax.set_title("Z-score vs moneyness\n(should be random scatter)")

    # Plot 3: Z-score vs DTE
    ax = axes[0, 2]
    ax.scatter(signals_df["DTE"], signals_df["z_score"],
               alpha=0.08, s=4, color="purple")
    ax.axhline( 2, color="red",   linewidth=1, linestyle="--")
    ax.axhline(-2, color="green", linewidth=1, linestyle="--")
    ax.set_xlabel("DTE")
    ax.set_title("Z-score vs DTE\n(should be random scatter)")

    # Plot 4: Signal count per day over test period
    ax = axes[1, 0]
    daily_signals = signals_df[abs(signals_df["z_score"]) > 2].groupby("date").size()
    daily_signals.plot(ax=ax, color="steelblue", alpha=0.7)
    daily_signals.rolling(5).mean().plot(ax=ax, color="red", linewidth=2)
    ax.set_title("Signals per day over time\n(red = 5-day rolling average)")
    ax.set_xlabel("")

    # Plot 5: Z-score calibration (QQ vs standard normal)
    ax = axes[1, 1]
    theoretical_quantiles = np.linspace(0.01, 0.99, 100)
    theoretical_z = stats.norm.ppf(theoretical_quantiles)
    actual_z = np.percentile(signals_df["z_score"].dropna(), np.linspace(1, 99, 100))
    ax.plot(theoretical_z, actual_z, "b-", linewidth=2, label="Actual z-scores")
    ax.plot([-4, 4], [-4, 4], "r--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Theoretical normal quantile")
    ax.set_ylabel("Actual z-score quantile")
    ax.set_title("Z-score calibration\n(blue on red line = well-calibrated)")
    ax.legend(fontsize=9)

    # Plot 6: Mispricing by moneyness bucket (boxplot)
    ax = axes[1, 2]
    bucket_masks = [
        abs(signals_df["moneyness"] - 1) > 0.07,
        (abs(signals_df["moneyness"] - 1) > 0.03) & (abs(signals_df["moneyness"] - 1) <= 0.07),
        abs(signals_df["moneyness"] - 1) <= 0.03,
    ]
    bp_data = [signals_df.loc[m, "mispricing"].dropna().values for m in bucket_masks]
    bp = ax.boxplot(bp_data, labels=["Deep OTM", "OTM", "Near ATM"],
                    patch_artist=True, medianprops=dict(color="red", linewidth=2))
    for patch, color in zip(bp["boxes"], ["#F09595", "#85B7EB", "#9FE1CB"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Mispricing distribution by bucket\n(median should be near 0)")
    ax.set_ylabel("Mispricing (Rs.)")

    plt.tight_layout()
    plt.savefig(f"Analysis_outcomes/signal_quality_{_TS}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Analysis_outcomes/signal_quality_{_TS}.png")

    # Signal statistics printout
    sell_signals = signals_df[signals_df["z_score"] >  2]
    buy_signals  = signals_df[signals_df["z_score"] < -2]

    print("\nSignal quality summary:")
    print(f"  Total liquid contracts: {len(signals_df):,}")
    print(f"  SELL signals (z>+2):    {len(sell_signals):,} ({len(sell_signals)/len(signals_df)*100:.1f}%)")
    print(f"  BUY  signals (z<-2):    {len(buy_signals):,}  ({len(buy_signals)/len(signals_df)*100:.1f}%)")
    print(f"\n  Z-score stats:")
    print(f"    Mean: {signals_df['z_score'].mean():.3f}  (should be ~0)")
    print(f"    Std:  {signals_df['z_score'].std():.3f}  (should be ~1)")
    print(f"    Skew: {signals_df['z_score'].skew():.3f}  (should be ~0)")

    sell_atm_pct = (abs(sell_signals["moneyness"] - 1) < 0.03).mean() * 100
    sell_otm_pct = (abs(sell_signals["moneyness"] - 1) > 0.05).mean() * 100
    print(f"\n  SELL signal moneyness breakdown:")
    print(f"    Near ATM: {sell_atm_pct:.1f}%  OTM: {sell_otm_pct:.1f}%")
    if sell_otm_pct > 70:
        print("    WARNING: >70% of SELL signals are deep OTM — liquidity filters may need tightening")


print("\n\n=== LAYER E: SIGNAL QUALITY ANALYSIS ===")
signal_quality_analysis(test_df)

# ---------------------------------------------------------------------------
# Layer F — Temporal stability
# ---------------------------------------------------------------------------
def temporal_stability_analysis(full_df, cutoff_date):
    full_df = full_df.copy()
    full_df["year_month"] = full_df["date"].dt.to_period("M")

    monthly_metrics = []
    for period, group in full_df.groupby("year_month"):
        if len(group) < 100:
            continue
        r2   = r2_score(group["log_price"], group["predicted_log_price"])
        rmse = np.sqrt(mean_squared_error(group["log_price"], group["predicted_log_price"]))
        bias = (group["log_price"] - group["predicted_log_price"]).mean()
        atm_iv_mean = group["ATM_IV"].mean()
        monthly_metrics.append(dict(
            period=str(period), r2=r2, rmse=rmse, bias=bias,
            atm_iv=atm_iv_mean, n=len(group)
        ))

    mdf = pd.DataFrame(monthly_metrics)
    mdf["period_dt"] = pd.to_datetime(mdf["period"])

    # Determine train/test boundary for shading
    cutoff_period = pd.to_datetime(str(
        full_df[full_df["date"] <= cutoff_date]["year_month"].max()
    ))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Temporal stability of model performance", fontsize=13)

    for ax in axes.flat:
        ax.axvspan(mdf["period_dt"].min(), cutoff_period,
                   alpha=0.08, color="steelblue", label="Train period")
        ax.axvspan(cutoff_period, mdf["period_dt"].max(),
                   alpha=0.08, color="orange", label="Test period")

    axes[0, 0].plot(mdf["period_dt"], mdf["r2"], "o-", color="steelblue", markersize=4)
    axes[0, 0].axhline(0.90, color="red", linewidth=1, linestyle="--", label="R²=0.90 floor")
    axes[0, 0].set_title("Monthly R²")
    axes[0, 0].set_ylim(0.7, 1.01)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(mdf["period_dt"], mdf["rmse"], "o-", color="darkorange", markersize=4)
    axes[0, 1].set_title("Monthly RMSE")

    axes[1, 0].plot(mdf["period_dt"], mdf["bias"], "o-", color="purple", markersize=4)
    axes[1, 0].axhline(0, color="black", linewidth=0.5, linestyle="--")
    axes[1, 0].set_title("Monthly mean bias\n(should stay near zero)")

    # ATM IV vs RMSE overlay
    ax_left = axes[1, 1]
    ax_left.plot(mdf["period_dt"], mdf["atm_iv"], "o-", color="steelblue",
                 markersize=4, label="ATM IV (left)")
    ax_right = ax_left.twinx()
    ax_right.plot(mdf["period_dt"], mdf["rmse"], "s--", color="coral",
                  markersize=4, label="RMSE (right)")
    ax_left.set_title("ATM IV vs RMSE\n(does high vol = worse accuracy?)")
    ax_left.legend(loc="upper left", fontsize=8)
    ax_right.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(f"Analysis_outcomes/temporal_stability_{_TS}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved Analysis_outcomes/temporal_stability_{_TS}.png")

    # Monthly performance table
    print("\nMonthly performance table:")
    print(f"  {'Period':<12} {'N':>6} {'R²':>8} {'RMSE':>8} {'Bias':>8}  Flag")
    print(f"  {'-'*60}")
    for _, row in mdf.iterrows():
        flag = ""
        if row["r2"] < 0.90:                       flag += " LOW_R2"
        if abs(row["bias"]) > 0.02:                flag += " BIASED"
        if row["rmse"] > mdf["rmse"].mean() * 1.5: flag += " HIGH_ERR"
        print(f"  {row['period']:<12} {row['n']:>6,} {row['r2']:>8.4f} "
              f"{row['rmse']:>8.4f} {row['bias']:>8.4f}{flag}")


print("\n\n=== LAYER F: TEMPORAL STABILITY ===")
temporal_stability_analysis(df, cutoff)

# ---------------------------------------------------------------------------
# Layer G — Model scorecard
# ---------------------------------------------------------------------------
def print_model_scorecard(t_metrics, tr_metrics, s_results, full_df):
    print("\n" + "=" * 60)
    print("  MODEL ACCURACY SCORECARD")
    print("=" * 60)

    checks = []

    def check(name, condition, good_msg, bad_msg, critical=False):
        symbol = "PASS" if condition else ("FAIL" if critical else "WARN")
        msg = good_msg if condition else bad_msg
        checks.append((symbol, name, msg))
        print(f"  [{symbol:4s}] {name:<35} {msg}")

    check("Global R² (test)", t_metrics["r2"] >= 0.90,
          f"{t_metrics['r2']:.4f}",
          f"{t_metrics['r2']:.4f} — retrain with more regularization", critical=True)

    check("Overfitting (R² gap)", (t_metrics["r2"] - tr_metrics["r2"]) > -0.05,
          "gap is acceptable",
          "gap > 0.05 — overfitting detected", critical=True)

    bias_full = (full_df["log_price"] - full_df["predicted_log_price"]).mean()
    check("Bias (mean residual)", abs(bias_full) < 0.01,
          "no systematic bias",
          "systematic bias present — check feature engineering")

    check("Z-score std", abs(full_df["z_score"].std() - 1.0) < 0.2,
          f"std={full_df['z_score'].std():.2f}",
          f"std={full_df['z_score'].std():.2f} — z-scores miscalibrated")

    check("Z-score mean", abs(full_df["z_score"].mean()) < 0.1,
          f"mean={full_df['z_score'].mean():.3f}",
          f"mean={full_df['z_score'].mean():.3f} — signal bias")

    if s_results is not None and len(s_results) > 0:
        atm_r2 = s_results[
            (s_results["seg"] == "moneyness_bucket") &
            (s_results["bucket"] == "Near ATM")
        ]["r2"].values
        if len(atm_r2):
            check("ATM R²", atm_r2[0] >= 0.90,
                  f"{atm_r2[0]:.4f}",
                  f"{atm_r2[0]:.4f} — model unreliable near ATM", critical=True)

    liquid = full_df[full_df["OI_normalized"] > 0.5]
    signal_pct = (abs(liquid["z_score"]) > 2).mean() * 100
    check("Signal rate (liquid)", 1.0 <= signal_pct <= 15.0,
          f"{signal_pct:.1f}% of liquid contracts",
          f"{signal_pct:.1f}% — {'too few' if signal_pct < 1 else 'too many'} signals")

    print("\n" + "=" * 60)
    fails = sum(1 for s, _, _ in checks if s == "FAIL")
    warns = sum(1 for s, _, _ in checks if s == "WARN")
    print(f"  Result: {fails} critical failures, {warns} warnings")
    if fails == 0 and warns <= 2:
        print("  Model is PRODUCTION READY for signal generation")
    elif fails == 0:
        print("  Model is USABLE but review warnings before live trading")
    else:
        print("  Model needs IMPROVEMENT before use — address FAIL items first")
    print("=" * 60)


print("\n\n=== LAYER G: MODEL SCORECARD ===")
print_model_scorecard(test_metrics, train_metrics, seg_results, df)

print("\nAll outputs written to Analysis_outcomes/")
