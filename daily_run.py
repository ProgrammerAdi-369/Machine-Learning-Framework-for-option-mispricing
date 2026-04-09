"""
daily_run.py — Daily inference pipeline entry point.

Run after market close each day:
    python daily_run.py --file option-chain-ED-BANKNIFTY-28-Apr-2026.csv
    python daily_run.py --auto          # picks newest CSV in option_chain_raw/

Steps:
    A. Validate input file
    B. Format raw CSV via option_data_formating.py
    C. Compute single-day features via daily_features.py
    D. Run model inference
    E. Generate BUY/SELL signals
    F. Save daily results to outputs/daily/
    G. Append to master wf_predictions.parquet + wf_trading_signals.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_FOLDER   = PROJECT_ROOT / "data" / "option_chain_raw"
DAILY_OUT    = PROJECT_ROOT / "outputs" / "daily"
MODELS_DIR   = PROJECT_ROOT / "models"

FEATURE_COLS = [
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness", "moneyness_sq",
    "IV", "ATM_IV", "IV_relative", "IV_rank", "IV_HV_Spread",
    "Skew", "TS_Slope", "log_HV_20",
    "OI_normalized", "Volume_normalized", "log_OI",
]

Z_SCORE_STD_FLOOR = 0.5


# ── A. Validation ─────────────────────────────────────────────────────────────

_DATE_RE = re.compile(
    r'(?:option.chain.(?:ED.)?)?'   # optional prefix (NSE or clean)
    r'([A-Z]+)'                     # asset name
    r'-(\d{1,2}-[A-Za-z]{3}-\d{4})',  # DD-Mon-YYYY
    re.IGNORECASE
)


def _parse_expiry_date(filename: str) -> tuple[str, date]:
    """
    Extract (asset, expiry_date) from the NSE filename.
    The date in the filename is the CONTRACT EXPIRY date, not the trade date.
    e.g. option-chain-ED-BANKNIFTY-26-May-2026.csv → ('BANKNIFTY', date(2026, 5, 26))
    """
    stem = Path(filename).stem
    m = _DATE_RE.search(stem)
    if not m:
        raise ValueError(
            f"Cannot parse date from filename: '{filename}'.\n"
            f"Expected pattern: option-chain-ED-BANKNIFTY-26-May-2026.csv"
        )
    asset      = m.group(1).upper()
    date_str   = m.group(2)           # e.g. '26-May-2026'
    try:
        expiry_date = datetime.strptime(date_str, "%d-%b-%Y").date()
    except ValueError as e:
        raise ValueError(f"Invalid date '{date_str}' in filename: {e}")
    return asset, expiry_date


def validate_input(filename: str, force: bool = False,
                   trade_date_override: date | None = None) -> tuple[Path, date]:
    """
    Validate raw input file. Returns (full_path, trade_date).

    trade_date is today's date by default (the day you run this script).
    Pass trade_date_override (via --date flag) when processing a historical file.
    """
    filename  = Path(filename).name   # strip any leading path
    file_path = RAW_FOLDER / filename

    # 1. Must exist
    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}\n"
            f"Drop the CSV into: {RAW_FOLDER}"
        )

    # 2. Must be a CSV
    if file_path.suffix.lower() != '.csv':
        raise ValueError(
            "Input must be a .csv file. "
            "NSE exports must be saved as CSV before running this script."
        )

    # 3. Validate filename is parseable (just to catch bad filenames early)
    _parse_expiry_date(filename)

    # 4. Trade date = today unless overridden via --date
    trade_date = trade_date_override or datetime.today().date()
    if trade_date.weekday() >= 5:
        print(f"WARNING: {trade_date} is a {trade_date.strftime('%A')}. "
              f"Proceeding — some expiry-day runs occur on weekends.")

    # 5. Must have at least 10 rows
    with open(file_path) as fh:
        row_count = sum(1 for _ in fh)
    if row_count < 12:
        raise ValueError(f"File appears truncated ({row_count} lines).")

    # 6. Check for existing output (idempotency guard)
    sig_path = DAILY_OUT / f"{trade_date}_signals.csv"
    if sig_path.exists() and not force:
        raise FileExistsError(
            f"Output already exists for {trade_date}: {sig_path}\n"
            f"Re-run with --force to overwrite."
        )

    return file_path, trade_date


def _auto_pick() -> str:
    """Return the filename of the newest option_chain_*.csv in RAW_FOLDER."""
    candidates = sorted(RAW_FOLDER.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {RAW_FOLDER}")
    return candidates[-1].name


# ── D. Model loading & inference ─────────────────────────────────────────────

def _load_best_model(trade_date: date):
    """Load the most recent walk-forward model on or before trade_date."""
    wf_models = sorted(MODELS_DIR.glob("wf_model_*.joblib"))
    trade_period = f"{trade_date.year}-{trade_date.month:02d}"
    eligible = [m for m in wf_models if m.stem.split('_', 2)[2] <= trade_period]
    if eligible:
        chosen = eligible[-1]
        print(f"  Using model: {chosen.name}")
        return joblib.load(chosen)
    print("WARNING: No walk-forward model found for this date. Using static model.")
    return joblib.load(MODELS_DIR / "xgb_mispricing.joblib")


def _load_clip_bounds() -> dict | None:
    """Load training percentile clip bounds. Returns None if file missing."""
    path = MODELS_DIR / "clip_bounds.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    print("WARNING: clip_bounds.json not found. Skipping feature clipping.")
    return None


def run_inference(features_df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    """Add predicted_price and mispricing columns to features_df."""
    model       = _load_best_model(trade_date)
    clip_bounds = _load_clip_bounds()

    # Build feature matrix in exact column order
    X = features_df[FEATURE_COLS].copy()

    # Apply training-time clip bounds
    if clip_bounds:
        for col, (lo, hi) in clip_bounds.items():
            if col in X.columns:
                X[col] = X[col].clip(lo, hi)

    # Drop rows with any NaN in features
    valid_mask = X.notna().all(axis=1)
    if not valid_mask.all():
        n_dropped = (~valid_mask).sum()
        print(f"WARNING: Dropping {n_dropped} rows with NaN features before inference.")

    X_clean = X[valid_mask]
    idx     = X_clean.index

    pred_log_price = model.predict(X_clean)

    df = features_df.copy()
    df['predicted_price'] = np.nan
    df['mispricing']      = np.nan
    df.loc[idx, 'predicted_price'] = (np.exp(pred_log_price) - 1).clip(min=0)
    df.loc[idx, 'mispricing']      = (
        df.loc[idx, 'close'] - df.loc[idx, 'predicted_price']
    )
    return df.dropna(subset=['predicted_price'])


# ── E. Signal generation ─────────────────────────────────────────────────────

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply liquidity filter, compute z-scores, assign BUY/SELL/HOLD."""
    liquid = (
        (df['OI_normalized']     > 0.5) &
        (df['Volume_normalized'] > 0.5) &
        (df['DTE']               > 5)
    )
    df_liquid = df[liquid].copy()
    print(f"  Liquid contracts: {len(df_liquid)} / {len(df)}")

    daily_mean = df_liquid['mispricing'].mean()
    daily_std  = df_liquid['mispricing'].std()

    if daily_std < Z_SCORE_STD_FLOOR:
        print(f"WARNING: daily_std={daily_std:.3f} floored to {Z_SCORE_STD_FLOOR}. "
              f"Signals may be less reliable on quiet days.")
    daily_std = max(daily_std, Z_SCORE_STD_FLOOR)

    df_liquid['z_score'] = (df_liquid['mispricing'] - daily_mean) / daily_std
    df_liquid['signal']  = 'HOLD'
    df_liquid.loc[df_liquid['z_score'] >  2, 'signal'] = 'SELL'
    df_liquid.loc[df_liquid['z_score'] < -2, 'signal'] = 'BUY'
    df_liquid['signal'] = df_liquid['signal'].str.upper()

    return df_liquid


# ── F. Persist daily results ──────────────────────────────────────────────────

def save_daily_results(df_pred: pd.DataFrame, df_signals: pd.DataFrame, trade_date: date):
    """
    Save daily results to outputs/daily/.
    df_pred    : all predictions with features (for vol smile / scatter charts)
    df_signals : liquid contracts with z_score/signal (for signal table)
    """
    DAILY_OUT.mkdir(parents=True, exist_ok=True)
    date_str = trade_date.strftime("%Y-%m-%d")

    # All predictions parquet (feeds dashboard charts)
    pred_path = DAILY_OUT / f"{date_str}_predictions.parquet"
    df_pred.to_parquet(pred_path, index=False)

    # BUY/SELL signals CSV (actionable output)
    signals_only = df_signals[df_signals['signal'].isin(['BUY', 'SELL'])].copy() \
        if not df_signals.empty and 'signal' in df_signals.columns else pd.DataFrame()
    sig_path = DAILY_OUT / f"{date_str}_signals.csv"
    signals_only.to_csv(sig_path, index=False)

    buy_n  = (df_signals['signal'] == 'BUY').sum()  if not df_signals.empty else 0
    sell_n = (df_signals['signal'] == 'SELL').sum() if not df_signals.empty else 0

    print(f"\n--- Daily Run Complete ---")
    print(f"Date         : {date_str}")
    print(f"Contracts    : {len(df_pred)}")
    print(f"Liquid rows  : {len(df_signals)}")
    print(f"BUY signals  : {buy_n}")
    print(f"SELL signals : {sell_n}")
    print(f"Saved to     : {pred_path}")
    print(f"             : {sig_path}")

    return pred_path, sig_path


# ── G. Append to master outputs ───────────────────────────────────────────────

def append_to_master(df_pred: pd.DataFrame, df_signals: pd.DataFrame, trade_date: date):
    """
    Idempotently append today's rows to the master output files.
    df_pred    : all predictions (used for wf_predictions.parquet — feeds R² chart)
    df_signals : liquid contracts with z_score/signal (used for wf_trading_signals.csv)
    """
    ts = pd.Timestamp(trade_date)

    def _normalise_date(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'date'}, inplace=True)
        if 'date' not in df.columns:
            df['date'] = ts
        return df

    pred_out = _normalise_date(df_pred)
    sig_out  = _normalise_date(df_signals)

    # ── wf_predictions.parquet — all predictions (needed for R² chart) ──
    pred_master = PROJECT_ROOT / "outputs" / "wf_predictions.parquet"
    if pred_master.exists():
        master = pd.read_parquet(pred_master)
        date_col = 'date' if 'date' in master.columns else 'Date'
        master = master[pd.to_datetime(master[date_col]) != ts]

        # Column alignment
        master_cols = set(master.columns)
        daily_cols  = set(pred_out.columns)
        for col in master_cols - daily_cols:
            pred_out[col] = np.nan
        extra = daily_cols - master_cols
        if extra:
            pred_out = pred_out.drop(columns=list(extra))

        updated = pd.concat([master, pred_out], ignore_index=True)
    else:
        updated = pred_out

    # Atomic write
    tmp = pred_master.with_suffix('.tmp.parquet')
    updated.to_parquet(tmp, index=False)
    os.replace(tmp, pred_master)
    print(f"  Appended to {pred_master.name} ({len(updated):,} total rows)")

    # ── wf_trading_signals.csv ──
    sig_master  = PROJECT_ROOT / "outputs" / "wf_trading_signals.csv"
    signals_out = sig_out[sig_out['signal'].isin(['BUY', 'SELL'])].copy() if 'signal' in sig_out.columns else pd.DataFrame()

    if sig_master.exists():
        master_sig = pd.read_csv(sig_master)
        date_col_s = 'date' if 'date' in master_sig.columns else 'Date'
        master_sig[date_col_s] = pd.to_datetime(master_sig[date_col_s])
        master_sig = master_sig[master_sig[date_col_s] != ts]
        updated_sig = pd.concat([master_sig, signals_out], ignore_index=True)
    else:
        updated_sig = signals_out

    tmp_csv = sig_master.with_suffix('.tmp.csv')
    updated_sig.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, sig_master)
    print(f"  Appended to {sig_master.name} ({len(updated_sig):,} total signal rows)")


# ── Main orchestrator ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Daily option mispricing inference pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--file', metavar='FILENAME',
                       help='Raw CSV filename (in data/option_chain_raw/)')
    group.add_argument('--auto', action='store_true',
                       help='Auto-pick the newest CSV in data/option_chain_raw/')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output for this date')
    parser.add_argument('--date', metavar='YYYY-MM-DD',
                        help='Override trade date (default: today). '
                             'Use when processing a historical file.')
    args = parser.parse_args()

    filename = _auto_pick() if args.auto else args.file
    print(f"\n=== Daily Run: {filename} ===")

    # Parse optional date override
    trade_date_override = None
    if args.date:
        try:
            trade_date_override = datetime.strptime(args.date, "%Y-%m-%d").date()
        except ValueError:
            print(f"ERROR: --date must be YYYY-MM-DD, got '{args.date}'")
            sys.exit(1)

    # A. Validate
    print("\n[Step A] Validating input...")
    file_path, trade_date = validate_input(filename, force=args.force,
                                           trade_date_override=trade_date_override)
    print(f"  Trade date : {trade_date}")

    # B. Format
    print("\n[Step B] Formatting raw CSV...")
    from option_data_formating import format_option_chain
    processed_path = format_option_chain(filename)

    # C. Features
    print("\n[Step C] Computing features...")
    from daily_features import compute_daily_features
    features_df = compute_daily_features(processed_path, trade_date)

    if len(features_df) < 10:
        print("ERROR: Too few contracts after feature computation. Aborting.")
        sys.exit(1)

    # D. Inference
    print("\n[Step D] Running model inference...")
    df_pred = run_inference(features_df, trade_date)

    # E. Signals
    print("\n[Step E] Generating signals...")
    df_signals = generate_signals(df_pred)

    # F. Save daily
    print("\n[Step F] Saving daily results...")
    save_daily_results(df_pred, df_signals, trade_date)

    # G. Append to master
    print("\n[Step G] Appending to master outputs...")
    append_to_master(df_pred, df_signals, trade_date)

    print("\nDone.")


if __name__ == "__main__":
    main()
