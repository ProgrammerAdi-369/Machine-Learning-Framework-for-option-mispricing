"""
daily_features.py — Single-day feature engineering for the daily inference pipeline.

Public API:
    compute_daily_features(processed_csv_path, trade_date) -> pd.DataFrame

All 16 model features are computed here using today's processed option chain
plus historical spot data from data/features/cross_sectional.parquet for HV_20.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date
from scipy.optimize import brentq
from scipy.stats import norm

PROJECT_ROOT   = Path(__file__).resolve().parent
CROSS_SEC_PATH = PROJECT_ROOT / "data" / "features" / "cross_sectional.parquet"
RISK_FREE_RATE = 0.065   # RBI repo rate — same as preprocess.py
IV_MAX         = 2.0     # cap at 200%


# ── Black-Scholes helpers (copied from preprocess.py) ────────────────────────

def _bs_price(S: float, K: float, T: float, r: float,
              sigma: float, option_type: str) -> float:
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "CE":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _compute_iv(S: float, K: float, T: float, r: float,
                market_price: float, option_type: str) -> float:
    """Compute implied volatility via brentq. Returns NaN on failure."""
    if T <= 0 or S <= 0 or K <= 0 or market_price <= 0:
        return np.nan
    intrinsic = max(0.0, S - K) if option_type == "CE" else max(0.0, K - S)
    if market_price < intrinsic - 0.01:
        return np.nan
    try:
        iv = brentq(
            lambda sigma: _bs_price(S, K, T, r, sigma, option_type) - market_price,
            1e-6, 10.0, xtol=1e-6, maxiter=100
        )
        return np.nan if iv > IV_MAX else iv
    except Exception:
        return np.nan


# ── Main public function ──────────────────────────────────────────────────────

def compute_daily_features(processed_csv_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Compute all 16 model features for a single trading day.

    Parameters
    ----------
    processed_csv_path : Path
        Path to the cleaned option chain CSV from option_data_formating.py.
    trade_date : date
        Today's actual trading date (when the script is run), NOT the expiry
        date from the filename. DTE = expiry_date - trade_date.

    Returns
    -------
    pd.DataFrame with one row per contract (CE + PE), containing all 16
    feature columns plus side columns kept for signal generation.
    """
    raw = pd.read_csv(processed_csv_path)

    # ── 1. Parse expiry date ─────────────────────────────────────────────────
    if 'expiry_date' in raw.columns:
        expiry_str = raw['expiry_date'].iloc[0]
    else:
        # Fall back: derive from filename  e.g. option_chain_BANKNIFTY-28-Apr-2026.csv
        stem = Path(processed_csv_path).stem        # option_chain_BANKNIFTY-28-Apr-2026
        expiry_str = '-'.join(stem.split('-')[-3:]) # 28-Apr-2026
    expiry_dt = pd.to_datetime(expiry_str, format='%d-%b-%Y')

    # ── 2. Infer spot from put-call parity ────────────────────────────────────
    valid_mask = raw['call_ltp'].notna() & raw['put_ltp'].notna()
    if valid_mask.sum() == 0:
        raise ValueError("No valid call/put LTP pairs — cannot infer spot price.")

    raw_valid = raw[valid_mask].copy()
    raw_valid['parity_diff'] = (raw_valid['call_ltp'] - raw_valid['put_ltp']).abs()
    atm_strike = raw_valid.loc[raw_valid['parity_diff'].idxmin(), 'strike']
    spot = float(atm_strike)

    # Sanity check: spot must be within ±15% of last known spot
    try:
        cs_hist = pd.read_parquet(CROSS_SEC_PATH)
        last_spot = float(cs_hist.groupby('Date')['spot'].first().sort_index().iloc[-1])
        if abs(spot - last_spot) / last_spot > 0.15:
            print(f"WARNING: Inferred spot {spot:.0f} differs >15% from last known "
                  f"spot {last_spot:.0f}. Verify the option chain data.")
    except Exception:
        pass

    # ── 3. Compute HV_20 from historical spot series ─────────────────────────
    try:
        cs_hist = pd.read_parquet(CROSS_SEC_PATH)
        spot_series = cs_hist.groupby('Date')['spot'].first().sort_index()
        # Append today's inferred spot
        spot_series[pd.Timestamp(trade_date)] = spot
        log_ret = np.log(spot_series / spot_series.shift(1)).dropna()
        hv_20 = float(log_ret.rolling(20).std().iloc[-1] * np.sqrt(252))
        if np.isnan(hv_20):
            raise ValueError("rolling window too short")
    except Exception as e:
        hv_20 = 0.20   # sensible fallback
        print(f"WARNING: Could not compute HV_20 from history ({e}). Using fallback {hv_20}.")

    # ── 4. Expand to per-contract rows (one row per strike × option_type) ────
    calls = raw[['strike', 'call_oi', 'call_volume', 'call_iv', 'call_ltp']].copy()
    calls.columns = ['strike', 'oi', 'volume', 'iv_pct', 'ltp']
    calls['option_type']         = 'CE'
    calls['option_type_encoded'] = 1

    puts = raw[['strike', 'put_oi', 'put_volume', 'put_iv', 'put_ltp']].copy()
    puts.columns = ['strike', 'oi', 'volume', 'iv_pct', 'ltp']
    puts['option_type']         = 'PE'
    puts['option_type_encoded'] = 0

    df = pd.concat([calls, puts], ignore_index=True)
    df = df[df['ltp'].notna() & (df['ltp'] > 0)].copy()

    # ── 5. Contract structure features ───────────────────────────────────────
    df['Date']      = pd.Timestamp(trade_date)
    df['expiry']    = expiry_dt
    df['DTE']       = (expiry_dt - pd.Timestamp(trade_date)).days
    df['spot']      = spot
    df['close']     = df['ltp']
    df['moneyness'] = df['strike'] / spot
    df['abs_moneyness'] = (df['moneyness'] - 1).abs()
    df['moneyness_sq']  = df['moneyness'] ** 2

    # Moneyness filter (same as historical pipeline)
    before_filter = len(df)
    df = df[(df['moneyness'] >= 0.80) & (df['moneyness'] <= 1.20)].copy()
    dropped_pct = (before_filter - len(df)) / max(before_filter, 1) * 100
    if dropped_pct > 80:
        print(f"WARNING: Moneyness filter removed {dropped_pct:.1f}% of contracts.")

    if df['DTE'].max() == 0:
        print("NOTE: All contracts have DTE=0 (expiry-day data). "
              "BS IV will be unavailable; using NSE-reported IV. "
              "Signal generation will produce no signals (DTE>5 filter).")

    # ── 6. Compute per-contract IV via Black-Scholes ──────────────────────────
    T_years = df['DTE'] / 365.0
    # For DTE=0 (expiry day), BS IV is undefined — use NSE-reported IV directly
    if df['DTE'].max() > 0:
        df['IV'] = [
            _compute_iv(spot, K, T, RISK_FREE_RATE, P, ot)
            for K, T, P, ot in zip(df['strike'], T_years, df['close'], df['option_type'])
        ]
    else:
        df['IV'] = np.nan  # will be filled from NSE IV below

    # Fall back to NSE-reported IV (converted from %) where BS fails
    nse_iv_fallback = df['iv_pct'] / 100.0
    df['IV'] = df['IV'].fillna(nse_iv_fallback)
    # Drop rows where IV is still NaN after both attempts
    df = df.dropna(subset=['IV']).copy()
    print(f"  IV computed: {len(df)} rows")

    # ── 7. ATM IV and relative features ──────────────────────────────────────
    atm_mask  = df['moneyness'].between(0.98, 1.02)
    atm_iv    = df.loc[atm_mask, 'IV'].mean() if atm_mask.sum() > 0 else df['IV'].mean()
    df['ATM_IV']      = atm_iv
    df['IV_relative'] = df['IV'] - atm_iv
    df['IV_rank']     = df['IV'].rank(pct=True)
    df['IV_HV_Spread'] = atm_iv - hv_20
    df['HV_20']       = hv_20
    df['log_HV_20']   = np.log1p(max(hv_20, 0))

    # ── 8. Regime features ────────────────────────────────────────────────────
    # Skew: PE at ~0.95 moneyness IV minus CE at ~1.05 moneyness IV
    pe_095 = df[(df['option_type'] == 'PE') & df['moneyness'].between(0.93, 0.97)]['IV'].mean()
    ce_105 = df[(df['option_type'] == 'CE') & df['moneyness'].between(1.03, 1.07)]['IV'].mean()
    skew   = (pe_095 - ce_105) if (not np.isnan(pe_095) and not np.isnan(ce_105)) else 0.0
    df['Skew'] = skew

    # TS_Slope: near-expiry IV - far-expiry IV (within today's data)
    near_iv = df[df['DTE'] <= 14]['IV'].mean()
    far_iv  = df[df['DTE'] >  14]['IV'].mean()
    ts_slope = 0.0
    if not np.isnan(near_iv) and not np.isnan(far_iv):
        ts_slope = near_iv - far_iv
    df['TS_Slope'] = ts_slope

    # ── 9. Liquidity features ─────────────────────────────────────────────────
    avg_oi  = df['oi'].mean()
    avg_vol = df['volume'].mean()
    df['OI_normalized']     = df['oi']     / avg_oi  if avg_oi  > 0 else 0.0
    df['Volume_normalized'] = df['volume'] / avg_vol if avg_vol > 0 else 0.0
    df['log_OI']            = np.log1p(df['oi'].clip(lower=0))

    # Keep raw OI/volume for liquidity filter in signal generation
    df['call_oi']     = df['oi']
    df['call_volume'] = df['volume']

    print(f"  Features computed: {len(df)} contracts | "
          f"DTE={df['DTE'].iloc[0]} | spot={spot:.0f} | "
          f"ATM_IV={atm_iv:.4f} | HV_20={hv_20:.4f} | Skew={skew:.4f}")

    return df.reset_index(drop=True)
