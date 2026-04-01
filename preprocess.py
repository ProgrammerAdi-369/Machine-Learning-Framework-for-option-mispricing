"""
BANKNIFTY Options Data Preprocessing Pipeline
Produces a cross-sectional feature dataset ready for XGBoost options mispricing model.
Blueprint: Options Mispricing Blueprint.docx

Phases:
  1 – Load & merge raw Excel files
  2 – Filter by liquidity, moneyness, DTE
  3 – Compute implied volatility (Black-Scholes brentq)
  4 – Build cross-sectional feature dataset (contract-level, one row per option per day)
"""

import os
import re
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR = "BANKNIFTY"
OUT_DIR = "data"
RISK_FREE_RATE = 0.065          # RBI repo rate ~6.5%
OI_MIN = 50                     # minimum OI contracts
MONEYNESS_LOW = 0.80
MONEYNESS_HIGH = 1.20
DTE_MIN = 1
DTE_MAX = 90
IV_MAX = 2.0                    # cap at 200% annualized
MIN_ROWS_PER_DAY = 10
N_JOBS = -1                     # parallel jobs for IV computation
HV_WINDOW = 20                  # rolling window (trading days) for realized vol

os.makedirs(f"{OUT_DIR}/raw", exist_ok=True)
os.makedirs(f"{OUT_DIR}/processed", exist_ok=True)
os.makedirs(f"{OUT_DIR}/features", exist_ok=True)


# ─────────────────────────────────────────────
# PHASE 1: LOAD & MERGE
# ─────────────────────────────────────────────

def fix_date_column(df, fname):
    """Handle multiple date formats found across NSE monthly files:
    - datetime64: already parsed (most files)
    - int/float DDMMYY: e.g. 10226 -> 01/02/26 -> 2026-02-01  (FEB26 file)
    - int Excel serial: e.g. 44197 -> 2021-01-01 (fallback)
    """
    col = df["Date"]
    if pd.api.types.is_datetime64_any_dtype(col):
        return df

    # Try DDMMYY integer format first (NSE quirk seen in FEB26 file)
    non_null = col.dropna()
    if len(non_null) and non_null.max() <= 9999999:
        try:
            date_strs = non_null.astype(float).astype("Int64").astype(str).str.zfill(6)
            test = pd.to_datetime(date_strs, format="%d%m%y", errors="coerce")
            valid_pct = test.notna().mean()
            if valid_pct > 0.8:
                df["Date"] = pd.to_datetime(
                    col.dropna().astype(float).astype("Int64").astype(str).str.zfill(6),
                    format="%d%m%y", errors="coerce"
                ).reindex(df.index)
                sample = df["Date"].dropna().iloc[0]
                print(f"  [{fname}] Converted DDMMYY integer dates. Sample: {sample.date()}")
                return df
        except Exception:
            pass

    # Fallback: Excel serial date (days since 1899-12-30)
    try:
        df["Date"] = pd.to_datetime(col, unit="D", origin="1899-12-30", errors="coerce")
        sample = df["Date"].dropna().iloc[0]
        print(f"  [{fname}] Converted Excel serial dates. Sample: {sample.date()}")
    except Exception as e:
        df["Date"] = pd.to_datetime(col, errors="coerce")
        print(f"  [{fname}] Date fallback parse. Error was: {e}")
    return df


def load_all_files():
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".xlsx")])
    frames = []
    for f in files:
        path = os.path.join(DATA_DIR, f)
        print(f"Loading {f}...")
        df = pd.read_excel(path)
        df["source_file"] = f
        df = fix_date_column(df, f)
        print(f"  Rows: {len(df):,}  |  Cols: {list(df.columns)}")
        frames.append(df)

    # Standardize column names (strip whitespace)
    frames = [df.rename(columns=lambda c: c.strip()) for df in frames]

    # Verify all files have the same columns
    col_sets = [set(df.columns) for df in frames]
    if len(set(frozenset(s) for s in col_sets)) > 1:
        print("WARNING: Column mismatch across files!")
        for f, cs in zip(files, col_sets):
            print(f"  {f}: {cs}")

    master = pd.concat(frames, ignore_index=True)
    print(f"\nMaster shape after merge: {master.shape}")

    # Drop duplicates
    before = len(master)
    master = master.drop_duplicates(subset=["CONTRACT_D", "Date"], keep="last")
    print(f"Duplicates dropped: {before - len(master):,}")

    # Drop rows with NaN in critical columns
    master = master.dropna(subset=["Date", "UNDRLNG_ST"])
    master = master[master["CLOSE_PRIC"].notna() | master["SETTLEMENT"].notna()]

    # Fill NaN OI with 0
    master["OI_NO_CON"] = master["OI_NO_CON"].fillna(0)

    print(f"Master shape after cleaning: {master.shape}")
    return master


# ─────────────────────────────────────────────
# PHASE 1.2: PARSE CONTRACT_D
# ─────────────────────────────────────────────

CONTRACT_PATTERN = re.compile(
    r"OPTIDXBANKNIFTY(\d{2}-[A-Z]{3}-\d{4})(CE|PE)(\d+)"
)


def parse_contract(contract_str):
    m = CONTRACT_PATTERN.match(str(contract_str).strip())
    if not m:
        return None, None, None
    expiry_str, opt_type, strike = m.groups()
    try:
        expiry_date = pd.to_datetime(expiry_str, format="%d-%b-%Y")
        strike = int(strike)
    except Exception:
        return None, None, None
    return expiry_date, opt_type, strike


def parse_contracts(df):
    print("\nParsing CONTRACT_D...")
    parsed = df["CONTRACT_D"].map(parse_contract)
    df["expiry_date"] = parsed.map(lambda x: x[0])
    df["option_type"] = parsed.map(lambda x: x[1])
    df["strike"] = parsed.map(lambda x: x[2])

    failed = df["expiry_date"].isna().sum()
    print(f"  Failed to parse: {failed:,} rows -- dropping them")
    df = df.dropna(subset=["expiry_date", "option_type", "strike"])
    df["strike"] = df["strike"].astype(int)
    return df


# ─────────────────────────────────────────────
# PHASE 1.3: DAYS TO EXPIRY
# ─────────────────────────────────────────────

def compute_dte(df):
    df["DTE"] = (df["expiry_date"] - df["Date"]).dt.days
    neg = (df["DTE"] < 0).sum()
    print(f"  Negative DTE rows (data errors): {neg:,} -- dropping")
    df = df[df["DTE"] >= 0]
    expiry_day = (df["DTE"] == 0).sum()
    print(f"  Expiry-day rows (DTE=0): {expiry_day:,} -- flagged but kept")
    df["is_expiry_day"] = (df["DTE"] == 0).astype(int)
    return df


# ─────────────────────────────────────────────
# PHASE 2: FILTERING
# ─────────────────────────────────────────────

def apply_filters(df):
    print("\nApplying liquidity filters...")
    n = len(df)

    # Use CLOSE_PRIC; fallback to SETTLEMENT if NaN
    df["close"] = df["CLOSE_PRIC"].where(df["CLOSE_PRIC"].notna(), df["SETTLEMENT"])
    df["spot"] = df["UNDRLNG_ST"]

    # Filter 1: remove zero close price
    df = df[df["close"] > 0]
    print(f"  After zero-close filter: {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 2: OI >= 50
    df = df[df["OI_NO_CON"] >= OI_MIN]
    print(f"  After OI filter (>={OI_MIN}): {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 3: moneyness 0.80-1.20
    df["moneyness"] = df["strike"] / df["spot"]
    df = df[(df["moneyness"] >= MONEYNESS_LOW) & (df["moneyness"] <= MONEYNESS_HIGH)]
    print(f"  After moneyness filter: {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Filter 4: DTE 1-90
    df = df[(df["DTE"] >= DTE_MIN) & (df["DTE"] <= DTE_MAX)]
    print(f"  After DTE filter ({DTE_MIN}-{DTE_MAX}): {len(df):,}  (removed {n - len(df):,})")
    n = len(df)

    # Flag sparse dates
    rows_per_day = df.groupby("Date").size()
    sparse_dates = rows_per_day[rows_per_day < MIN_ROWS_PER_DAY].index
    print(f"  Sparse dates (< {MIN_ROWS_PER_DAY} rows): {len(sparse_dates)}")
    df = df[~df["Date"].isin(sparse_dates)]
    print(f"  After dropping sparse dates: {len(df):,}")

    # Validate spot price consistency per day
    spot_std = df.groupby("Date")["spot"].std()
    bad_days = spot_std[spot_std > 100].index
    if len(bad_days):
        print(f"  WARNING: {len(bad_days)} days with inconsistent spot price (std > 100). Investigate!")
        for d in bad_days[:5]:
            print(f"    {d.date()}: std={spot_std[d]:.2f}")

    return df


# ─────────────────────────────────────────────
# PHASE 3: IMPLIED VOLATILITY
# ─────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, option_type):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def compute_all_iv(df):
    print("\nComputing implied volatility (this may take 5-15 minutes)...")
    rows = df.to_dict("records")

    def iv_for_record(rec):
        S = rec["spot"]
        K = rec["strike"]
        T = rec["DTE"] / 365.0
        r = RISK_FREE_RATE
        market_price = rec["close"]
        opt_type = rec["option_type"]

        if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
            return np.nan
        intrinsic = max(0, S - K) if opt_type == "CE" else max(0, K - S)
        if market_price < intrinsic - 0.01:
            return np.nan

        def objective(sigma):
            return bs_price(S, K, T, r, sigma, opt_type) - market_price

        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=1e-6, maxiter=100)
            return np.nan if iv > IV_MAX else iv
        except Exception:
            return np.nan

    ivs = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(iv_for_record)(rec) for rec in rows
    )
    df["IV"] = ivs

    total = len(df)
    valid = df["IV"].notna().sum()
    print(f"  IV computed: {valid:,}/{total:,} rows ({100*valid/total:.1f}%)")

    if valid / total < 0.95:
        print("  WARNING: more than 5% of rows have NaN IV -- check upstream filters")

    df = df.dropna(subset=["IV"])

    atm = df[df["moneyness"].between(0.98, 1.02)]
    print(f"\n  ATM IV stats (moneyness 0.98-1.02):")
    print(f"    Mean: {atm['IV'].mean()*100:.1f}%  |  Median: {atm['IV'].median()*100:.1f}%")
    print(f"    Min:  {atm['IV'].min()*100:.1f}%   |  Max:    {atm['IV'].max()*100:.1f}%")

    return df


# ─────────────────────────────────────────────
# PHASE 4A: DAILY REGIME FEATURES
# ─────────────────────────────────────────────

def compute_daily_regime_features(df):
    """
    Compute ATM_IV, Skew, TS_Slope as daily scalars and broadcast
    back to every contract row on that date via a left-merge.
    """
    print("\nComputing daily regime features (ATM_IV, Skew, TS_Slope)...")
    results = []

    for date, day_df in df.groupby("Date"):
        # ATM_IV: mean IV within +-2% moneyness; widen band if empty
        atm_iv = np.nan
        for band in [0.02, 0.03, 0.05]:
            near_atm = day_df[abs(day_df["moneyness"] - 1.0) < band]
            if len(near_atm) > 0:
                atm_iv = near_atm["IV"].mean()
                break

        # Skew: 0.95-moneyness PE IV minus 1.05-moneyness CE IV
        # Fall back to nearest contract when exact band is empty
        puts  = day_df[day_df["option_type"] == "PE"].copy()
        calls = day_df[day_df["option_type"] == "CE"].copy()

        put_iv = np.nan
        if len(puts) > 0:
            puts["dist_95"] = abs(puts["moneyness"] - 0.95)
            near_95 = puts[puts["dist_95"] < 0.05]
            put_iv = (near_95["IV"].mean() if len(near_95) > 0
                      else puts.loc[puts["dist_95"].idxmin(), "IV"])

        call_iv = np.nan
        if len(calls) > 0:
            calls["dist_105"] = abs(calls["moneyness"] - 1.05)
            near_105 = calls[calls["dist_105"] < 0.05]
            call_iv = (near_105["IV"].mean() if len(near_105) > 0
                       else calls.loc[calls["dist_105"].idxmin(), "IV"])

        skew = (put_iv - call_iv
                if not (np.isnan(put_iv) or np.isnan(call_iv)) else np.nan)

        # TS_Slope: ATM IV of nearest expiry minus next expiry
        expiries = sorted(day_df["expiry_date"].unique())
        ts_slope = np.nan
        if len(expiries) >= 2:
            iv_near = day_df[day_df["expiry_date"] == expiries[0]]["IV"].mean()
            iv_far  = day_df[day_df["expiry_date"] == expiries[1]]["IV"].mean()
            ts_slope = iv_near - iv_far

        results.append({"Date": date, "ATM_IV": atm_iv, "Skew": skew, "TS_Slope": ts_slope})

    daily_feats = pd.DataFrame(results)
    print(f"  Days with NaN ATM_IV:  {daily_feats['ATM_IV'].isna().sum()}")
    print(f"  Days with NaN Skew:    {daily_feats['Skew'].isna().sum()}")
    print(f"  Days with NaN TS_Slope:{daily_feats['TS_Slope'].isna().sum()}")

    df = df.merge(daily_feats, on="Date", how="left")
    return df


# ─────────────────────────────────────────────
# PHASE 4B: HV_20 AND IV_HV_SPREAD
# ─────────────────────────────────────────────

def compute_hv_spread(df):
    """
    Build daily spot series -> rolling 20-day realized vol (annualized) -> HV_20.
    Merge back to contract rows, then compute IV_HV_Spread = ATM_IV - HV_20.
    """
    print("\nComputing HV_20 and IV_HV_Spread...")

    spot_series = df.groupby("Date")["spot"].first().sort_index()
    log_ret = np.log(spot_series / spot_series.shift(1))
    hv20 = log_ret.rolling(HV_WINDOW).std() * np.sqrt(252)
    hv20 = hv20.rename("HV_20").reset_index()   # columns: Date, HV_20

    df = df.merge(hv20, on="Date", how="left")
    df["IV_HV_Spread"] = df["ATM_IV"] - df["HV_20"]

    valid = df["HV_20"].notna().sum()
    print(f"  HV_20 valid rows: {valid:,}/{len(df):,} "
          f"(first {HV_WINDOW} trading days will be NaN -- expected)")
    return df


# ─────────────────────────────────────────────
# PHASE 4C: CONTRACT-LEVEL FEATURES
# ─────────────────────────────────────────────

def build_cross_sectional_dataset(df):
    """
    Add all contract-level features needed by the XGBoost model.
    One row per contract per day -- nothing collapsed to daily summaries.
    """
    print("\nBuilding cross-sectional feature dataset...")

    df["option_type_encoded"] = (df["option_type"] == "CE").astype(int)
    df["log_price"]    = np.log(df["close"] + 1)
    df["IV_relative"]  = df["IV"] - df["ATM_IV"]
    df["abs_moneyness"] = abs(df["moneyness"] - 1.0)

    # OI normalization: per-day average
    df["avg_OI_day"]    = df.groupby("Date")["OI_NO_CON"].transform("mean")
    df["OI_normalized"] = df["OI_NO_CON"] / df["avg_OI_day"].replace(0, np.nan)

    # Volume normalization: per-day average (column name may vary)
    vol_col = next((c for c in df.columns if c.upper() in ("VOLUME", "TRADED_QTY", "TRDNG_VALUE", "TRADED_QUA")), None)
    if vol_col:
        df["avg_vol_day"]       = df.groupby("Date")[vol_col].transform("mean")
        df["Volume_normalized"] = df[vol_col] / df["avg_vol_day"].replace(0, np.nan)
        df = df.rename(columns={vol_col: "VOLUME"})
    else:
        print("  WARNING: No volume column found -- Volume_normalized will be NaN")
        df["avg_vol_day"]       = np.nan
        df["Volume_normalized"] = np.nan

    # IV rank: within-day percentile of this contract's IV
    df["IV_rank"] = df.groupby("Date")["IV"].rank(pct=True)

    print(f"  Cross-sectional dataset shape: {df.shape}")
    return df


# ─────────────────────────────────────────────
# PHASE 4D: DROP NaN ON CRITICAL COLUMNS
# ─────────────────────────────────────────────

CRITICAL_COLS = ["IV", "ATM_IV", "moneyness", "log_price"]


def drop_critical_nans(df):
    before = len(df)
    df = df.dropna(subset=CRITICAL_COLS)
    dropped = before - len(df)
    print(f"\nDropped {dropped:,} rows with NaN in critical columns {CRITICAL_COLS}")
    print(f"Final cross-sectional dataset: {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BANKNIFTY Options Mispricing Pipeline")
    print("=" * 60)

    # -- Phase 1: Load, parse, DTE --
    master = load_all_files()
    master.to_parquet(f"{OUT_DIR}/raw/master_raw.parquet", index=False)
    print(f"\nSaved: {OUT_DIR}/raw/master_raw.parquet")

    master = parse_contracts(master)
    master = compute_dte(master)

    # -- Phase 2: Filter --
    master = apply_filters(master)
    master.to_parquet(f"{OUT_DIR}/processed/master_filtered.parquet", index=False)
    print(f"Saved: {OUT_DIR}/processed/master_filtered.parquet")

    # -- Phase 3: Implied Volatility --
    master = compute_all_iv(master)
    master.to_parquet(f"{OUT_DIR}/processed/master_with_iv.parquet", index=False)
    print(f"Saved: {OUT_DIR}/processed/master_with_iv.parquet")

    # -- Phase 4: Cross-sectional features --
    master = compute_daily_regime_features(master)
    master = compute_hv_spread(master)
    master = build_cross_sectional_dataset(master)
    master = drop_critical_nans(master)
    master.to_parquet(f"{OUT_DIR}/features/cross_sectional.parquet", index=False)
    print(f"Saved: {OUT_DIR}/features/cross_sectional.parquet")

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Final dataset shape: {master.shape}")
    print(f"Date range: {master['Date'].min().date()} -> {master['Date'].max().date()}")
    print("=" * 60)

    return master


if __name__ == "__main__":
    final = main()
    print("\nSample output (last 5 rows):")
    preview_cols = [
        "Date", "strike", "option_type", "close", "IV", "ATM_IV",
        "IV_relative", "Skew", "TS_Slope", "HV_20", "IV_HV_Spread",
        "OI_normalized", "IV_rank"
    ]
    print(final[[c for c in preview_cols if c in final.columns]].tail().to_string())
