"""
option_data_formating.py — Format & clean a raw NSE option chain CSV.

Supports two filename patterns:
  NSE download : option-chain-ED-BANKNIFTY-28-Apr-2026.csv
  Clean format : option_chain_BANKNIFTY-28-Apr-2026.csv

Usage (standalone):
    python option_data_formating.py option-chain-ED-BANKNIFTY-28-Apr-2026.csv
"""

import sys
import pandas as pd
from pathlib import Path

PROJECT_ROOT     = Path(__file__).resolve().parent
RAW_FOLDER       = PROJECT_ROOT / "data" / "option_chain_raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "option_chain_processed"


def _parse_filename(name_without_ext: str) -> tuple[str, str]:
    """
    Parse asset name and date string from NSE or clean filename stem.

    NSE format  : option-chain-ED-BANKNIFTY-28-Apr-2026  → ('BANKNIFTY', '28-Apr-2026')
    Clean format: option_chain_BANKNIFTY-28-Apr-2026     → ('BANKNIFTY', '28-Apr-2026')
    """
    # NSE format: split on '-', asset is 4th token (index 3)
    # option | chain | ED | BANKNIFTY | 28 | Apr | 2026
    parts = name_without_ext.replace('_', '-').split('-')
    # Find the asset token: first token that is all-uppercase letters (length > 3)
    asset_idx = None
    for i, p in enumerate(parts):
        if p.isupper() and len(p) > 3:
            asset_idx = i
            break

    if asset_idx is None:
        raise ValueError(
            f"Could not parse asset name from filename: '{name_without_ext}'. "
            f"Expected format: option-chain-ED-BANKNIFTY-28-Apr-2026.csv"
        )

    asset_name = parts[asset_idx]
    date_str   = '-'.join(parts[asset_idx + 1:])  # e.g. '28-Apr-2026'

    if not date_str:
        raise ValueError(f"Could not parse date from filename: '{name_without_ext}'")

    return asset_name, date_str


def format_option_chain(input_filename: str) -> Path:
    """
    Format and clean a raw NSE option chain CSV.
    Returns the path to the processed output file.

    Parameters
    ----------
    input_filename : str
        Bare filename (not full path) inside data/option_chain_raw/.
    """
    # Always use just the filename, in case user passed a full path
    input_filename = Path(input_filename).name

    file_path = RAW_FOLDER / input_filename
    if not file_path.exists():
        raise FileNotFoundError(f"Raw file not found: {file_path}")

    # Parse asset + date from filename
    name_without_ext = input_filename.replace('.csv', '')
    asset_name, date_str = _parse_filename(name_without_ext)

    # Output filename
    output_filename = f'option_chain_{asset_name}-{date_str}.csv'
    output_path = PROCESSED_FOLDER / output_filename
    PROCESSED_FOLDER.mkdir(parents=True, exist_ok=True)

    # === LOAD DATA ===
    df = pd.read_csv(file_path, header=1)

    # Drop columns that are entirely empty
    df = df.dropna(axis=1, thresh=3)

    # Column count guard
    expected_cols = 21
    if len(df.columns) != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns after dropping empty ones, "
            f"got {len(df.columns)}. Actual columns: {list(df.columns)}. "
            f"Check if NSE changed the export format."
        )

    # Rename columns
    df.columns = [
        "call_oi", "call_chng_oi", "call_volume", "call_iv", "call_ltp",
        "call_net_chng", "call_bid_qty", "call_bid_price",
        "call_ask_price", "call_ask_qty",
        "strike",
        "put_bid_qty", "put_bid_price", "put_ask_price", "put_ask_qty",
        "put_net_chng", "put_ltp", "put_iv", "put_volume",
        "put_chng_oi", "put_oi"
    ]

    # Replace '-' with NaN
    df = df.replace('-', pd.NA)

    # Clean numeric columns (remove commas, convert to float)
    numeric_cols = df.columns.drop('strike')
    df[numeric_cols] = (
        df[numeric_cols]
        .astype(str)
        .apply(lambda col: col.str.replace(',', '', regex=False))
    )
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Drop rows without strike
    df = df.dropna(subset=['strike'])

    # Step 1A: Clean strike — remove commas, convert to int
    df['strike'] = (
        df['strike'].astype(str)
        .str.replace(',', '', regex=False)
    )
    df['strike'] = pd.to_numeric(df['strike'], errors='coerce')
    df = df.dropna(subset=['strike'])
    df['strike'] = df['strike'].astype(float).astype(int)

    # Sort by strike
    df = df.sort_values('strike').reset_index(drop=True)

    # Step 1B: Interpolate IV columns (vol smile is smooth)
    if df['call_iv'].notna().sum() >= 3:
        df['call_iv'] = df['call_iv'].interpolate(method='linear', limit_direction='both')
    else:
        print("WARNING: Too few call_iv values to interpolate — expiry-day or data error.")

    if df['put_iv'].notna().sum() >= 3:
        df['put_iv'] = df['put_iv'].interpolate(method='linear', limit_direction='both')
    else:
        print("WARNING: Too few put_iv values to interpolate — expiry-day or data error.")

    # Step 1C: Interpolate LTP columns
    df['call_ltp'] = df['call_ltp'].interpolate(method='linear', limit_direction='both')
    df['put_ltp']  = df['put_ltp'].interpolate(method='linear', limit_direction='both')

    # Step 1D: Fill OI and volume NaN with 0
    oi_vol_cols = ['call_oi', 'call_chng_oi', 'call_volume',
                   'put_oi',  'put_chng_oi',  'put_volume']
    df[oi_vol_cols] = df[oi_vol_cols].fillna(0)

    # Step 1E: Drop strikes where both IV columns are still NaN after interpolation
    df = df.dropna(subset=['call_iv', 'put_iv'], how='all')

    # Step 1F: Add metadata columns extracted from filename
    df['asset']       = asset_name   # e.g. 'BANKNIFTY'
    df['expiry_date'] = date_str      # e.g. '28-Apr-2026'

    # Step 1G: Print quality report
    print(f"\n--- Cleaning Report ---")
    print(f"Total strikes after cleaning : {len(df)}")
    print(f"Strike range                 : {df['strike'].min()} to {df['strike'].max()}")
    print(f"Strikes with call_iv         : {df['call_iv'].notna().sum()}")
    print(f"Strikes with put_iv          : {df['put_iv'].notna().sum()}")
    print(f"Asset                        : {asset_name}")
    print(f"Expiry date in file          : {date_str}")

    # === SAVE OUTPUT ===
    df.to_csv(output_path, index=False)
    print(f"\nProcessed file saved at:\n{output_path}")

    return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python option_data_formating.py <filename>")
        sys.exit(1)
    out = format_option_chain(sys.argv[1])
    print(f"Done: {out}")
