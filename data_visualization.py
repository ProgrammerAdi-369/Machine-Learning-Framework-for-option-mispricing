"""
data_visualization.py
Evaluates the cross-sectional pipeline output.

Checks:
  - Data balance (temporal, option-type)
  - Feature distributions: normality vs skewness
  - Outlier detection (box plots, scatter, IQR counts)
  - Time-series regime features
  - Correlation structure
  - Q-Q normality plots
  - Missing values
  - Transformation recommendations (raw vs log)

All figures saved to: outcomes/
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                           # non-interactive; must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
PARQUET_PATH = "data/features/cross_sectional.parquet"
OUT_DIR      = "outcomes"
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE  = "muted"
FIG_DPI  = 150
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)

# Features used by the XGBoost model
MODEL_FEATURES = [
    "strike", "DTE", "moneyness", "abs_moneyness",
    "IV", "ATM_IV", "IV_relative", "IV_rank",
    "Skew", "TS_Slope", "HV_20", "IV_HV_Spread",
    "OI_normalized", "log_price",
]


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_data():
    print("Loading cross_sectional.parquet ...")
    df = pd.read_parquet(PARQUET_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    df["YearMonth"] = df["Date"].dt.to_period("M")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    return df


# ─────────────────────────────────────────────
# FIG 01: TEMPORAL DATA BALANCE
# ─────────────────────────────────────────────

def plot_temporal_balance(df):
    """
    Row count per calendar month.
    Significance: Imbalanced months mean some periods dominate model training.
    A spike or trough flags market-closure anomalies or data gaps.
    """
    monthly = df.groupby("YearMonth").size().reset_index(name="count")
    monthly["month_str"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["#e74c3c" if c < monthly["count"].mean() * 0.7 else "#2ecc71"
              for c in monthly["count"]]
    bars = ax.bar(monthly["month_str"], monthly["count"], color=colors, edgecolor="white", linewidth=0.6)

    mean_val = monthly["count"].mean()
    ax.axhline(mean_val, color="steelblue", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:,.0f}")

    for bar, val in zip(bars, monthly["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", va="bottom", fontsize=7.5)

    ax.set_title("Fig 01 — Temporal Data Balance: Rows per Month\n"
                 "Significance: Red bars (<70% of mean) flag under-represented months that could bias the model",
                 pad=10)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Rows")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    save(fig, "01_temporal_balance.png")


# ─────────────────────────────────────────────
# FIG 02: OPTION TYPE & MONEYNESS BALANCE
# ─────────────────────────────────────────────

def plot_type_and_moneyness_balance(df):
    """
    CE vs PE split + moneyness bucket distribution.
    Significance: A strong CE/PE imbalance or OTM-heavy dataset means the model
    trains mostly on one regime — important for signal reliability.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: CE vs PE pie
    type_counts = df["option_type"].value_counts()
    axes[0].pie(type_counts, labels=type_counts.index, autopct="%1.1f%%",
                colors=["#3498db", "#e74c3c"], startangle=90,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title("(A) CE vs PE Split\nImbalance => model learns one type better")

    # Panel B: Moneyness bucket distribution
    bins  = [0.80, 0.90, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.10, 1.20]
    labels = ["0.80-0.90", "0.90-0.95", "0.95-0.97", "0.97-0.99", "0.99-1.01",
              "1.01-1.03", "1.03-1.05", "1.05-1.10", "1.10-1.20"]
    df["m_bucket"] = pd.cut(df["moneyness"], bins=bins, labels=labels)
    bucket_counts = df["m_bucket"].value_counts().sort_index()
    bucket_colors = ["#e74c3c" if "0.99-1.01" in str(k) else "#3498db" for k in bucket_counts.index]
    axes[1].bar(range(len(bucket_counts)), bucket_counts.values, color=bucket_colors, edgecolor="white")
    axes[1].set_xticks(range(len(bucket_counts)))
    axes[1].set_xticklabels(bucket_counts.index, rotation=50, ha="right", fontsize=8)
    axes[1].set_title("(B) Moneyness Bucket Distribution\nRed = ATM band (densest signal zone)")
    axes[1].set_ylabel("Row Count")

    # Panel C: DTE distribution
    dte_counts = df.groupby("DTE").size()
    axes[2].plot(dte_counts.index, dte_counts.values, color="#2ecc71", linewidth=1.5)
    axes[2].fill_between(dte_counts.index, dte_counts.values, alpha=0.25, color="#2ecc71")
    axes[2].set_title("(C) DTE Distribution\nPeaks near expiry = liquidity clustering")
    axes[2].set_xlabel("Days to Expiry")
    axes[2].set_ylabel("Row Count")

    fig.suptitle("Fig 02 — Dataset Balance: Option Type, Moneyness, DTE", fontsize=13, y=1.02)
    fig.tight_layout()
    save(fig, "02_balance_type_moneyness_dte.png")


# ─────────────────────────────────────────────
# FIG 03: PRICE DISTRIBUTION — RAW vs LOG TRANSFORM
# ─────────────────────────────────────────────

def plot_price_distribution(df):
    """
    close price is extremely right-skewed (0.05 to 13,779).
    Shows why log_price is necessary as the model target.
    Significance: Skew in target inflates RMSE for cheap OTM options.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # Raw close - histogram
    axes[0, 0].hist(df["close"], bins=200, color="#e74c3c", edgecolor="none", alpha=0.8)
    axes[0, 0].set_title("(A) Raw Close Price — Histogram\nExtremely right-skewed: bulk < 500, long tail to 13,779")
    axes[0, 0].set_xlabel("Close Price (INR)")
    axes[0, 0].set_ylabel("Frequency")
    skew_raw = df["close"].skew()
    axes[0, 0].text(0.65, 0.85, f"Skewness: {skew_raw:.2f}", transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Raw close - log-x scale for full range visibility
    axes[0, 1].hist(np.log1p(df["close"]), bins=100, color="#e74c3c", edgecolor="none", alpha=0.8)
    axes[0, 1].set_title("(B) log(1 + Close Price) — Histogram\nMore symmetric; confirms log transform is the right choice")
    axes[0, 1].set_xlabel("log(1 + Close Price)")
    axes[0, 1].set_ylabel("Frequency")
    skew_log = np.log1p(df["close"]).skew()
    axes[0, 1].text(0.65, 0.85, f"Skewness: {skew_log:.2f}", transform=axes[0, 1].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # log_price (already computed in pipeline)
    axes[1, 0].hist(df["log_price"], bins=100, color="#3498db", edgecolor="none", alpha=0.8)
    axes[1, 0].set_title("(C) log_price (pipeline column) — Distribution\nTarget variable for XGBoost; should be approximately normal")
    axes[1, 0].set_xlabel("log_price = log(close + 1)")
    axes[1, 0].set_ylabel("Frequency")
    skew_lp = df["log_price"].skew()
    kurt_lp = df["log_price"].kurtosis()
    axes[1, 0].text(0.03, 0.85,
                    f"Skewness: {skew_lp:.2f}\nKurtosis: {kurt_lp:.2f}",
                    transform=axes[1, 0].transAxes,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Percentile comparison table
    pcts = [1, 5, 25, 50, 75, 95, 99]
    raw_vals = np.percentile(df["close"], pcts)
    log_vals = np.percentile(df["log_price"], pcts)
    axes[1, 1].axis("off")
    table_data = [[f"{p}%", f"{r:,.1f}", f"{l:.3f}"] for p, r, l in zip(pcts, raw_vals, log_vals)]
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Percentile", "close (raw)", "log_price"],
        cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    axes[1, 1].set_title("(D) Percentile Comparison\nTransformation needed: raw close is unusable as-is")

    fig.suptitle("Fig 03 — Option Price Distribution: Raw vs Log Transform\n"
                 "Key finding: log transform reduces skewness from {:.1f} to {:.2f}".format(skew_raw, skew_log),
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "03_price_distribution_raw_vs_log.png")


# ─────────────────────────────────────────────
# FIG 04: IV DISTRIBUTION
# ─────────────────────────────────────────────

def plot_iv_distribution(df):
    """
    IV is the core input feature.
    Significance: Right skew from high-IV OTM options.
    Bimodal shape would indicate two regimes (e.g. before/after event).
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Overall IV histogram + KDE
    axes[0].hist(df["IV"], bins=100, density=True, color="#9b59b6", alpha=0.7, edgecolor="none", label="All")
    for opt, col in [("CE", "#3498db"), ("PE", "#e74c3c")]:
        sub = df[df["option_type"] == opt]["IV"].dropna()
        sub.plot.kde(ax=axes[0], color=col, linewidth=2, label=opt)
    axes[0].set_title("(A) IV Distribution by Option Type\n"
                      "PE typically has higher IV (put premium / skew)")
    axes[0].set_xlabel("Implied Volatility")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    skew_iv = df["IV"].skew()
    axes[0].text(0.65, 0.85, f"Skewness: {skew_iv:.2f}", transform=axes[0].transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # IV by DTE bucket
    dte_buckets = pd.cut(df["DTE"], bins=[0, 7, 14, 30, 60, 90], labels=["1-7", "8-14", "15-30", "31-60", "61-90"])
    df_temp = df.copy()
    df_temp["DTE_bucket"] = dte_buckets
    bp_data = [df_temp[df_temp["DTE_bucket"] == b]["IV"].dropna().values
               for b in ["1-7", "8-14", "15-30", "31-60", "61-90"]]
    bp = axes[1].boxplot(bp_data, patch_artist=True, notch=False,
                         medianprops=dict(color="black", linewidth=1.5))
    colors_bp = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    for patch, color in zip(bp["boxes"], colors_bp):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1].set_xticklabels(["1-7", "8-14", "15-30", "31-60", "61-90"])
    axes[1].set_title("(B) IV by DTE Bucket\nShort-DTE options have higher, more volatile IV")
    axes[1].set_xlabel("Days to Expiry")
    axes[1].set_ylabel("IV")

    # IV by moneyness bucket
    m_buckets = pd.cut(df["moneyness"],
                       bins=[0.80, 0.90, 0.95, 0.97, 0.99, 1.01, 1.03, 1.05, 1.10, 1.20],
                       labels=["0.80-0.90", "0.90-0.95", "0.95-0.97", "0.97-0.99", "0.99-1.01",
                               "1.01-1.03", "1.03-1.05", "1.05-1.10", "1.10-1.20"])
    df_temp2 = df.copy()
    df_temp2["m_bucket"] = m_buckets
    med_iv = df_temp2.groupby("m_bucket", observed=True)["IV"].median()
    axes[2].bar(range(len(med_iv)), med_iv.values,
                color=["#e74c3c" if "0.99-1.01" in str(k) else "#3498db"
                       for k in med_iv.index],
                edgecolor="white")
    axes[2].set_xticks(range(len(med_iv)))
    axes[2].set_xticklabels(med_iv.index, rotation=45, ha="right", fontsize=8)
    axes[2].set_title("(C) Median IV by Moneyness\nU-shape = vol smile; OTM options have higher IV")
    axes[2].set_xlabel("Moneyness Bucket")
    axes[2].set_ylabel("Median IV")

    fig.suptitle("Fig 04 — Implied Volatility Distribution\n"
                 "Key finding: Right-skewed (OTM spike), vol smile present, short-DTE IV is elevated",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "04_iv_distribution.png")


# ─────────────────────────────────────────────
# FIG 05: KEY FEATURE DISTRIBUTIONS GRID
# ─────────────────────────────────────────────

def plot_feature_distributions_grid(df):
    """
    Distribution of all model features.
    Significance: Identifies which features need transformation,
    which are roughly normal, and which have problematic distributions.
    """
    features = {
        "ATM_IV":       "ATM_IV — Daily regime IV\n(should be unimodal; bimodal = 2 market regimes)",
        "Skew":         "Skew (PE 0.95 IV - CE 1.05 IV)\n(positive = fear premium in puts; near-normal)",
        "TS_Slope":     "TS_Slope (near - far expiry IV)\n(negative = contango; positive = backwardation)",
        "HV_20":        "HV_20 — 20-day Realized Vol\n(right-skewed; log-transform may help)",
        "IV_HV_Spread": "IV_HV_Spread (ATM_IV - HV_20)\n(positive = expensive options; key mispricing signal)",
        "IV_relative":  "IV_relative (IV - ATM_IV)\n(centered near 0; OTM outliers visible)",
        "OI_normalized":"OI_normalized\n(heavy right-skew: high-OI contracts dominate)",
        "IV_rank":      "IV_rank (within-day percentile)\n(should be ~uniform if cross-section is balanced)",
        "abs_moneyness":"abs_moneyness |moneyness - 1|\n(right-skewed: most contracts near ATM)",
    }

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.flatten()

    for i, (col, title) in enumerate(features.items()):
        data = df[col].dropna()
        ax = axes[i]

        # Histogram + KDE
        ax.hist(data, bins=80, density=True, color="#3498db", alpha=0.5, edgecolor="none")
        try:
            data.plot.kde(ax=ax, color="#e74c3c", linewidth=2)
        except Exception:
            pass

        # Annotate skewness and kurtosis
        sk = data.skew()
        ku = data.kurtosis()
        ax.text(0.02, 0.93,
                f"n={len(data):,}\nSkew: {sk:.2f}\nKurt: {ku:.2f}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel(col, fontsize=8)
        ax.set_ylabel("Density", fontsize=8)

    fig.suptitle("Fig 05 — Key Feature Distributions\n"
                 "Significance: |Skew| > 1 flags transformation need; bimodal = regime split",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "05_feature_distributions_grid.png")


# ─────────────────────────────────────────────
# FIG 06: OUTLIER DETECTION — BOX PLOTS
# ─────────────────────────────────────────────

def plot_outlier_boxplots(df):
    """
    Box plots of model features to visualize outlier extent.
    Significance: Points beyond 3*IQR are potential data errors or
    extreme-regime observations that can destabilize XGBoost training.
    """
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    cols = ["IV", "ATM_IV", "log_price", "IV_relative",
            "Skew", "TS_Slope", "OI_normalized", "abs_moneyness"]

    for i, col in enumerate(cols):
        data = df[col].dropna()
        ax = axes[i]

        bp = ax.boxplot(data, vert=True, patch_artist=True, notch=False,
                        flierprops=dict(marker=".", markersize=2, color="#e74c3c", alpha=0.3),
                        medianprops=dict(color="black", linewidth=2),
                        boxprops=dict(facecolor="#3498db", alpha=0.6))

        # IQR-based outlier count
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        outliers = ((data < q1 - 1.5 * iqr) | (data > q3 + 1.5 * iqr)).sum()
        extreme  = ((data < q1 - 3.0 * iqr) | (data > q3 + 3.0 * iqr)).sum()

        ax.set_title(f"{col}\n1.5xIQR outliers: {outliers:,} ({100*outliers/len(data):.1f}%)\n"
                     f"3xIQR extreme: {extreme:,} ({100*extreme/len(data):.1f}%)",
                     fontsize=9)
        ax.set_xticks([])

    fig.suptitle("Fig 06 — Outlier Detection via Box Plots\n"
                 "Significance: High outlier % in IV_relative / OI_normalized warns of model instability;"
                 " consider capping at 99th percentile",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "06_outlier_boxplots.png")


# ─────────────────────────────────────────────
# FIG 07: OUTLIER SUMMARY — IQR COUNTS BAR CHART
# ─────────────────────────────────────────────

def plot_outlier_summary(df):
    """
    Horizontal bar chart: % outliers per feature at 1.5x and 3x IQR.
    Significance: Quick triage — features above 5% outlier rate need
    capping or robust scaling before model training.
    """
    numeric_cols = [c for c in MODEL_FEATURES if c in df.columns]
    results = []
    for col in numeric_cols:
        data = df[col].dropna()
        q1, q3 = data.quantile(0.25), data.quantile(0.75)
        iqr = q3 - q1
        pct_15 = 100 * ((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr)).sum() / len(data)
        pct_30 = 100 * ((data < q1 - 3.0*iqr) | (data > q3 + 3.0*iqr)).sum() / len(data)
        results.append({"feature": col, "1.5xIQR": pct_15, "3xIQR": pct_30})

    res_df = pd.DataFrame(results).sort_values("1.5xIQR", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))
    y = np.arange(len(res_df))
    ax.barh(y - 0.2, res_df["1.5xIQR"], height=0.35, color="#e67e22", label="1.5x IQR outliers", alpha=0.85)
    ax.barh(y + 0.2, res_df["3xIQR"],   height=0.35, color="#e74c3c", label="3x IQR extreme",    alpha=0.85)
    ax.axvline(5, color="black", linestyle="--", linewidth=1.2, label="5% threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(res_df["feature"])
    ax.set_xlabel("Outlier % of rows")
    ax.set_title("Fig 07 — Outlier Rate per Feature (IQR method)\n"
                 "Significance: Features above 5% line may need capping (clip at 1st/99th percentile)\n"
                 "before XGBoost training to prevent a few extreme values from dominating splits",
                 pad=10)
    ax.legend()
    fig.tight_layout()
    save(fig, "07_outlier_summary_by_feature.png")


# ─────────────────────────────────────────────
# FIG 08: VOLATILITY SMILE
# ─────────────────────────────────────────────

def plot_vol_smile(df):
    """
    IV vs moneyness scatter — the vol smile.
    Significance: A clean U-shape validates IV computation correctness.
    Outliers above the smile are candidates for mispricing signals.
    Flat or inverted shape indicates data quality problems.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Full scatter (sample for speed)
    sample = df.sample(min(15000, len(df)), random_state=42)
    for opt, col, label in [("CE", "#3498db", "Call"), ("PE", "#e74c3c", "Put")]:
        sub = sample[sample["option_type"] == opt]
        axes[0].scatter(sub["moneyness"], sub["IV"],
                        c=col, alpha=0.12, s=6, label=label)
    axes[0].set_title("(A) Vol Smile: IV vs Moneyness (sample 15k points)\n"
                      "U-shape confirms vol smile; outliers above are mispricing candidates")
    axes[0].set_xlabel("Moneyness (Strike / Spot)")
    axes[0].set_ylabel("Implied Volatility")
    axes[0].axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axes[0].legend()

    # Panel B: Median IV by moneyness + DTE bucket (smoothed smile)
    dte_labels = ["1-7", "8-14", "15-30", "31-60", "61-90"]
    dte_bins   = [0, 7, 14, 30, 60, 90]
    df_temp = df.copy()
    df_temp["DTE_bucket"] = pd.cut(df_temp["DTE"], bins=dte_bins, labels=dte_labels)
    df_temp["m_round"] = (df_temp["moneyness"] * 100).round() / 100

    colors_smile = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
    for label, color in zip(dte_labels, colors_smile):
        sub = df_temp[df_temp["DTE_bucket"] == label]
        smile = sub.groupby("m_round")["IV"].median().reset_index()
        smile = smile[(smile["m_round"] >= 0.88) & (smile["m_round"] <= 1.12)]
        if len(smile) > 3:
            axes[1].plot(smile["m_round"], smile["IV"], color=color,
                         linewidth=2, label=f"DTE {label}", marker="o", markersize=3)

    axes[1].set_title("(B) Median Vol Smile by DTE Bucket\n"
                      "Short-DTE smile is steeper (higher gamma risk near expiry)")
    axes[1].set_xlabel("Moneyness")
    axes[1].set_ylabel("Median IV")
    axes[1].axvline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].legend(fontsize=9)

    fig.suptitle("Fig 08 — Volatility Smile Analysis\n"
                 "Significance: Validates IV computation; identifies smile shape used by model",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "08_volatility_smile.png")


# ─────────────────────────────────────────────
# FIG 09: TIME-SERIES — REGIME FEATURES
# ─────────────────────────────────────────────

def plot_timeseries_regime(df):
    """
    ATM_IV, HV_20, Skew, TS_Slope over time.
    Significance: Reveals market regimes (high-vol periods, skew spikes).
    Model trained without accounting for regimes may perform poorly in unseen regimes.
    """
    daily = (df.groupby("Date")
               .agg(ATM_IV=("ATM_IV", "first"),
                    HV_20=("HV_20", "first"),
                    Skew=("Skew", "first"),
                    TS_Slope=("TS_Slope", "first"))
               .reset_index()
               .sort_values("Date"))

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    # ATM_IV vs HV_20
    axes[0].plot(daily["Date"], daily["ATM_IV"] * 100, color="#3498db", linewidth=1.5, label="ATM_IV (%)")
    axes[0].plot(daily["Date"], daily["HV_20"]  * 100, color="#e74c3c",  linewidth=1.5, label="HV_20 (%)", linestyle="--")
    axes[0].fill_between(daily["Date"],
                         daily["ATM_IV"] * 100, daily["HV_20"] * 100,
                         where=daily["ATM_IV"] >= daily["HV_20"],
                         alpha=0.15, color="#3498db", label="IV > HV (options expensive)")
    axes[0].fill_between(daily["Date"],
                         daily["ATM_IV"] * 100, daily["HV_20"] * 100,
                         where=daily["ATM_IV"] < daily["HV_20"],
                         alpha=0.15, color="#e74c3c", label="IV < HV (options cheap)")
    axes[0].set_ylabel("Volatility (%)")
    axes[0].set_title("ATM_IV vs HV_20  |  Blue fill = IV premium (options overpriced), Red fill = IV discount")
    axes[0].legend(fontsize=8)

    # IV_HV_Spread
    spread = daily["ATM_IV"] - daily["HV_20"]
    axes[1].bar(daily["Date"], spread * 100, color=np.where(spread >= 0, "#3498db", "#e74c3c"),
                width=1.0, alpha=0.7)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("IV_HV_Spread (%)")
    axes[1].set_title("IV_HV_Spread (ATM_IV - HV_20)  |  Persistent positive = systematically overpriced options")

    # Skew
    axes[2].plot(daily["Date"], daily["Skew"] * 100, color="#9b59b6", linewidth=1.5)
    axes[2].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[2].fill_between(daily["Date"], daily["Skew"] * 100, 0,
                         where=daily["Skew"] >= 0, alpha=0.2, color="#9b59b6")
    axes[2].set_ylabel("Skew (%)")
    axes[2].set_title("Skew (PE 0.95 IV - CE 1.05 IV)  |  Spikes = fear/tail-risk events; negative = rare complacency")

    # TS_Slope
    axes[3].plot(daily["Date"], daily["TS_Slope"] * 100, color="#27ae60", linewidth=1.5)
    axes[3].axhline(0, color="grey", linewidth=0.8, linestyle="--")
    axes[3].fill_between(daily["Date"], daily["TS_Slope"] * 100, 0,
                         where=daily["TS_Slope"] >= 0, alpha=0.2, color="#27ae60")
    axes[3].set_ylabel("TS_Slope (%)")
    axes[3].set_xlabel("Date")
    axes[3].set_title("TS_Slope (near - far expiry IV)  |  Positive = backwardation (near-term fear); negative = contango")

    fig.suptitle("Fig 09 — Daily Regime Features Over Time\n"
                 "Significance: Regime shifts are non-stationarity sources the model must handle",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "09_timeseries_regime_features.png")


# ─────────────────────────────────────────────
# FIG 10: CORRELATION HEATMAP
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df):
    """
    Pearson correlation among model features.
    Significance: High correlations (|r| > 0.85) indicate multicollinearity.
    While XGBoost handles this, it can cause unstable feature importances.
    """
    cols = [c for c in MODEL_FEATURES if c in df.columns]
    corr = df[cols].dropna(how="any").corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.4, ax=ax,
                annot_kws={"size": 8}, square=True)
    ax.set_title("Fig 10 — Feature Correlation Heatmap\n"
                 "Significance: |r| > 0.85 = multicollinearity risk (feature importance becomes unreliable);\n"
                 "|r| > 0.5 with log_price = strong predictor signal",
                 pad=12)
    fig.tight_layout()
    save(fig, "10_correlation_heatmap.png")


# ─────────────────────────────────────────────
# FIG 11: Q-Q PLOTS — NORMALITY CHECK
# ─────────────────────────────────────────────

def plot_qq(df):
    """
    Q-Q plots: how well does each distribution match normal?
    Significance: Heavy tails (bow outward) = fat-tailed data.
    Features that deviate from normal may need robust scaling.
    """
    cols = ["close", "log_price", "IV", "ATM_IV", "IV_relative",
            "Skew", "TS_Slope", "OI_normalized"]
    titles = [
        "close (raw price)\nHeavy right tail expected",
        "log_price (model target)\nShould be near-normal",
        "IV\nModerate right skew expected",
        "ATM_IV\nDaily regime; should be smoother",
        "IV_relative (IV - ATM_IV)\nShould center on 0",
        "Skew\nNear-normal; spikes = events",
        "TS_Slope\nShould be near-normal",
        "OI_normalized\nHeavy right tail expected",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    for i, (col, title) in enumerate(zip(cols, titles)):
        data = df[col].dropna()
        ax = axes[i]
        stats.probplot(data.sample(min(5000, len(data)), random_state=42), dist="norm", plot=ax)
        ax.set_title(title, fontsize=8.5)
        ax.get_lines()[0].set(markersize=2, alpha=0.4, color="#3498db")
        ax.get_lines()[1].set(color="#e74c3c", linewidth=1.5)

    fig.suptitle("Fig 11 — Q-Q Normality Plots (sample 5,000 per feature)\n"
                 "Significance: Deviation from red line = non-normal distribution;\n"
                 "S-curve = heavy tails; bow = skewness; both affect model calibration",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "11_qq_normality_plots.png")


# ─────────────────────────────────────────────
# FIG 12: MISSING VALUES SUMMARY
# ─────────────────────────────────────────────

def plot_missing_values(df):
    """
    Bar chart of null counts per column.
    Significance: Columns with > 5% null need imputation strategy
    (or explicit NaN handling in XGBoost).
    HV_20 nulls are expected (first 20 trading days).
    Volume nulls indicate a column-name mismatch — action needed.
    """
    nulls = df.isnull().sum()
    nulls = nulls[nulls > 0].sort_values(ascending=False)
    pct   = 100 * nulls / len(df)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if p > 5 else "#f39c12" if p > 1 else "#2ecc71" for p in pct]
    bars = ax.bar(range(len(nulls)), pct.values, color=colors, edgecolor="white")

    for bar, (col, p) in zip(bars, pct.items()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(range(len(nulls)))
    ax.set_xticklabels(nulls.index, rotation=45, ha="right")
    ax.axhline(5, color="#e74c3c", linestyle="--", linewidth=1.2, label="5% threshold (action needed)")
    ax.axhline(1, color="#f39c12", linestyle="--", linewidth=1.0, label="1% threshold (monitor)")
    ax.set_ylabel("% Null Rows")
    ax.set_title("Fig 12 — Missing Values per Column\n"
                 "Red = >5% null (needs imputation or column fix) | "
                 "Orange = 1-5% (monitor) | Green = <1% (acceptable)\n"
                 "Note: HV_20 nulls are expected (warmup period). Volume nulls = column name mismatch.",
                 pad=10)
    ax.legend()
    fig.tight_layout()
    save(fig, "12_missing_values.png")


# ─────────────────────────────────────────────
# FIG 13: OI DISTRIBUTION — RAW vs NORMALIZED
# ─────────────────────────────────────────────

def plot_oi_distribution(df):
    """
    OI_NO_CON is heavily right-skewed (50 to 109,669).
    OI_normalized corrects for daily average but may still be skewed.
    Significance: OI is a liquidity proxy — its distribution determines
    how well Volume/OI filters will remove illiquid contracts.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Raw OI
    axes[0].hist(df["OI_NO_CON"], bins=150, color="#e67e22", alpha=0.8, edgecolor="none")
    axes[0].set_title("(A) Raw OI_NO_CON\n"
                      "Extreme right skew — a few contracts have massive OI")
    axes[0].set_xlabel("Open Interest (contracts)")
    axes[0].set_ylabel("Frequency")
    axes[0].text(0.6, 0.85, f"Skew: {df['OI_NO_CON'].skew():.2f}",
                 transform=axes[0].transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Log OI
    log_oi = np.log1p(df["OI_NO_CON"])
    axes[1].hist(log_oi, bins=80, color="#e67e22", alpha=0.8, edgecolor="none")
    axes[1].set_title("(B) log(1 + OI_NO_CON)\n"
                      "Log-transform makes OI more symmetric — use for features")
    axes[1].set_xlabel("log(1 + OI)")
    axes[1].set_ylabel("Frequency")
    axes[1].text(0.02, 0.85, f"Skew: {log_oi.skew():.2f}",
                 transform=axes[1].transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # OI normalized
    oi_norm = df["OI_normalized"].dropna()
    # Cap display at 99th percentile to avoid tail distortion
    cap = oi_norm.quantile(0.99)
    axes[2].hist(oi_norm[oi_norm <= cap], bins=80, color="#27ae60", alpha=0.8, edgecolor="none")
    axes[2].set_title(f"(C) OI_normalized (capped at 99th pct = {cap:.1f})\n"
                      "Still right-skewed; values > 3 are unusually high-OI contracts")
    axes[2].set_xlabel("OI / daily average OI")
    axes[2].set_ylabel("Frequency")
    axes[2].text(0.6, 0.85, f"Skew: {oi_norm.skew():.2f}",
                 transform=axes[2].transAxes,
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    fig.suptitle("Fig 13 — OI Distribution: Raw vs Log vs Normalized\n"
                 "Key finding: OI needs log-transform before use as a raw feature;\n"
                 "OI_normalized is already a per-day ratio but still right-skewed",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "13_oi_distribution.png")


# ─────────────────────────────────────────────
# FIG 14: IV_RANK UNIFORMITY CHECK
# ─────────────────────────────────────────────

def plot_iv_rank(df):
    """
    IV_rank should be approximately uniform [0, 1] if within-day IVs
    are well-distributed across the cross-section.
    Significance: A non-uniform IV_rank (e.g., piling near 0 or 1)
    means the cross-section on many days is dominated by extreme IVs.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of IV_rank — expect uniform
    axes[0].hist(df["IV_rank"], bins=50, color="#8e44ad", alpha=0.8, edgecolor="none", density=True)
    axes[0].axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Expected (uniform)")
    axes[0].set_title("(A) IV_rank Distribution\n"
                      "Should be flat ~1.0 if IVs are well-spread each day.\n"
                      "Spikes at ends = many contracts tied for min/max IV on a day")
    axes[0].set_xlabel("IV_rank (within-day percentile)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # IV_rank by option type
    for opt, col in [("CE", "#3498db"), ("PE", "#e74c3c")]:
        sub = df[df["option_type"] == opt]["IV_rank"]
        axes[1].hist(sub, bins=50, color=col, alpha=0.5, density=True, label=opt)
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Uniform baseline")
    axes[1].set_title("(B) IV_rank by Option Type\n"
                      "CE and PE IV_rank distributions should overlap.\n"
                      "Divergence = systematic CE/PE IV ordering")
    axes[1].set_xlabel("IV_rank")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    fig.suptitle("Fig 14 — IV_rank Uniformity Check\n"
                 "Significance: Near-uniform = good cross-sectional spread; "
                 "piled distribution = rank feature may not generalize",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    save(fig, "14_iv_rank_uniformity.png")


# ─────────────────────────────────────────────
# FIG 15: TRANSFORMATION RECOMMENDATIONS SUMMARY
# ─────────────────────────────────────────────

def plot_transformation_summary(df):
    """
    Side-by-side skewness before/after log transform for right-skewed features.
    Significance: Confirms which transformations reduce skew
    and should be applied before Phase 5 model training.
    """
    candidates = {
        "close":     "log(close + 1)",
        "OI_NO_CON": "log(OI + 1)",
        "HV_20":     "log(HV_20 + 1e-6)",
    }

    fig, axes = plt.subplots(len(candidates), 2, figsize=(14, 11))

    for i, (col, transform_label) in enumerate(candidates.items()):
        raw  = df[col].dropna()
        log_ = np.log1p(raw.clip(lower=0))

        # Raw — histogram only (KDE skipped: extreme range causes slow/unstable bandwidth)
        axes[i, 0].hist(raw, bins=100, color="#e74c3c", alpha=0.75, edgecolor="none", density=True)
        axes[i, 0].set_title(f"{col} — RAW   (Skew: {raw.skew():.2f})", fontsize=10)
        axes[i, 0].set_ylabel("Density")

        # Transformed — histogram + KDE (well-behaved after log)
        axes[i, 1].hist(log_, bins=100, color="#2ecc71", alpha=0.75, edgecolor="none", density=True)
        try:
            log_.plot.kde(ax=axes[i, 1], color="#27ae60", linewidth=2)
        except Exception:
            pass
        axes[i, 1].set_title(f"{transform_label} — TRANSFORMED   (Skew: {log_.skew():.2f})", fontsize=10)

    fig.suptitle("Fig 15 — Transformation Recommendations: Raw vs Log\n"
                 "Action: Features with |Skew| > 1 should be log-transformed before model training.\n"
                 "log_price already applied for the target; apply to OI and HV_20 as features.",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    save(fig, "15_transformation_recommendations.png")


# ─────────────────────────────────────────────
# FIG 16: PIPELINE VOLUME COLUMN ISSUE
# ─────────────────────────────────────────────

def plot_volume_column_audit(df):
    """
    Volume_normalized is 100% NaN because the actual volume column
    in the NSE data is TRADED_QUA, not VOLUME/TRADED_QTY/TRDNG_VALUE.
    This plot audits what volume-related columns exist and their distributions.
    Significance: Volume_normalized is a key liquidity filter in Phase 7 signal generation.
    Without it, the filter cannot work — preprocess.py must be patched.
    """
    vol_candidates = [c for c in df.columns
                      if any(kw in c.upper() for kw in ["TRAD", "VOLUME", "QTY", "QUA", "NOTIONAL", "PREMIUM"])]

    n = len(vol_candidates)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(max(n * 4, 8), 5))
    if n == 1:
        axes = [axes]

    for i, col in enumerate(vol_candidates):
        data = df[col].dropna()
        if len(data) == 0:
            axes[i].text(0.5, 0.5, f"{col}\nAll NaN", ha="center", va="center",
                         transform=axes[i].transAxes, color="red", fontsize=12)
            axes[i].set_title(col)
            continue
        axes[i].hist(np.log1p(data.clip(lower=0)), bins=80, color="#16a085", alpha=0.8, edgecolor="none")
        axes[i].set_title(f"{col}\nn={len(data):,}  Skew: {data.skew():.2f}")
        axes[i].set_xlabel("log(1 + value)")

    fig.suptitle("Fig 16 — Volume / Traded Quantity Column Audit\n"
                 "Issue: Volume_normalized = 100% NaN because TRADED_QUA was not matched in preprocess.py.\n"
                 "Fix needed: add 'TRADED_QUA' to the vol_col lookup in build_cross_sectional_dataset()",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    save(fig, "16_volume_column_audit.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Options Mispricing Pipeline — Data Visualization")
    print(f"Output folder: {OUT_DIR}/")
    print("=" * 60)

    df = load_data()

    steps = [
        ("Fig 01 — Temporal balance",             plot_temporal_balance),
        ("Fig 02 — Option type / moneyness / DTE", plot_type_and_moneyness_balance),
        ("Fig 03 — Price distribution raw vs log", plot_price_distribution),
        ("Fig 04 — IV distribution",               plot_iv_distribution),
        ("Fig 05 — Feature distributions grid",    plot_feature_distributions_grid),
        ("Fig 06 — Outlier box plots",             plot_outlier_boxplots),
        ("Fig 07 — Outlier summary bar chart",     plot_outlier_summary),
        ("Fig 08 — Volatility smile",              plot_vol_smile),
        ("Fig 09 — Time-series regime features",   plot_timeseries_regime),
        ("Fig 10 — Correlation heatmap",           plot_correlation_heatmap),
        ("Fig 11 — Q-Q normality plots",           plot_qq),
        ("Fig 12 — Missing values",                plot_missing_values),
        ("Fig 13 — OI distribution",               plot_oi_distribution),
        ("Fig 14 — IV_rank uniformity",            plot_iv_rank),
        ("Fig 15 — Transformation recommendations",plot_transformation_summary),
        ("Fig 16 — Volume column audit",           plot_volume_column_audit),
    ]

    for label, fn in steps:
        print(f"\n{label}...")
        fn(df)

    print("\n" + "=" * 60)
    print(f"Done. {len(steps)} figures saved to: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
