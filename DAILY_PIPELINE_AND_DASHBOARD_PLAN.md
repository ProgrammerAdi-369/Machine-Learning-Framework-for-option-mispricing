# Daily Computation & Dashboard — Complete Implementation Guide
### Option Mispricing Pipeline — Absolute Reference for Claude Code

---

## 0. What This Document Covers

The existing pipeline (`preprocess.py → train.py → accuracy_analysis.py → retrain.py`) is a one-time batch operation over a fixed historical study period (Apr 2025 – Mar 2026). The trained walk-forward models are saved in `models/`. That work is done.

This document plans two new things that together form the "daily use" layer:

1. **`daily_run.py`** — A single script the user runs after market close each day. It takes a raw NSE option chain CSV, runs it through formatting → feature engineering → inference → signal generation, and saves the results.
2. **Dashboard** — A Streamlit app (`dashboard/`) that displays historical pipeline results (already planned in `dashboard_plan.md`) **plus** a new daily results panel that shows today's output from `daily_run.py`.

**What is NOT changing:** The trained models, `preprocess.py`, `train.py`, `retrain.py`, and `accuracy_analysis.py` are untouched. This is purely an inference + display layer built on top of them.

---

## 1. Full Architecture

```
[After market close]
User drops:  data/option_chain_raw/option_chain_BANKNIFTY-28-Apr-2026.csv
                          │
                          ▼
              python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv
                          │
          ┌───────────────┼────────────────────────────┐
          │               │                            │
    STEP A: Ingest   STEP B: Format            STEP C: Features
    & Validate       option_data_formatting.py  daily_features.py
    raw CSV          → option_chain_processed/  → single-day DataFrame
          │               │                            │
          └───────────────┴────────────────────────────┘
                          │
                    STEP D: Inference
                    load latest wf_model_*.joblib
                    → predicted_price, mispricing, z_score per row
                          │
                    STEP E: Signal Generation
                    z > +2 → SELL, z < -2 → BUY
                          │
                    STEP F: Persist Results
                    outputs/daily/YYYY-MM-DD_predictions.parquet
                    outputs/daily/YYYY-MM-DD_signals.csv
                          │
                    STEP G: Append to Master Outputs
                    outputs/wf_predictions.parquet   ← append today's rows
                    outputs/wf_trading_signals.csv   ← append today's rows
                          │
                          ▼
              streamlit run dashboard/app.py
                          │
          ┌───────────────┼─────────────────────────────────┐
          │               │               │                  │
     Panel 1:        Panel 2:        Panel 3:          Panel 4:
     KPI Cards      Top Signals    Z-Score Dist.    Monthly R² Trend
    (existing)      (existing)      (existing)        (existing)
                                                            +
                                                      Panel 5 (NEW):
                                                      Daily Results
                                                      (today only)
```

---

## 2. Repository Changes

These are the only new files and folders that need to be created. Nothing existing is modified except Step G (appending to master outputs) and adding Panel 5 to the dashboard.

```
option-mispricing-pipeline/
│
├── data/
│   ├── option_chain_raw/         ← user drops raw CSV here (already exists)
│   ├── option_chain_processed/   ← formatted CSV written here (already exists)
│   └── daily_features/           ← NEW: single-day feature parquets
│       └── YYYY-MM-DD_features.parquet
│
├── outputs/
│   └── daily/                    ← NEW: daily inference results
│       ├── YYYY-MM-DD_predictions.parquet
│       └── YYYY-MM-DD_signals.csv
│
├── daily_run.py                  ← NEW: single entry-point script
├── option_data_formatting.py     ← MODIFIED: fix paths + refactor to function
├── daily_features.py             ← NEW: single-day feature engineering module
│
└── dashboard/
    ├── app.py                    ← MODIFIED: add Panel 5 import + layout
    ├── data_loader.py            ← MODIFIED: add daily loaders
    ├── components/
    │   ├── kpi_cards.py          (existing)
    │   ├── signals_table.py      (existing)
    │   ├── zscore_chart.py       (existing)
    │   ├── performance_chart.py  (existing)
    │   └── daily_panel.py        ← NEW: Panel 5 component
    ├── utils.py                  (existing)
    └── requirements.txt          (existing)
```

---

## 3. Step A — Raw File Ingestion & Validation

**File:** `daily_run.py` (partial — the entry point and validation logic)

### What to build

`daily_run.py` is the only script the user ever calls. It orchestrates all steps A through G. At Step A it resolves and validates the input file.

```python
# CLI interface
python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv
python daily_run.py --auto   # picks the newest .csv in option_chain_raw/
```

**Validation checks to run before doing any processing:**

1. File exists at `data/option_chain_raw/<filename>`
2. File is a `.csv` (not `.xlsx` or other)
3. Filename matches the expected pattern: `option_chain_{ASSET}-{DD}-{Mon}-{YYYY}.csv`
4. Parse `trade_date` from the filename (`28-Apr-2026` → `datetime(2026, 4, 28)`) and confirm it is a valid calendar date, not a weekend, and not already processed (check if `outputs/daily/{date}_signals.csv` already exists — warn and ask for `--force` flag if so)
5. The CSV has at least 10 rows (not empty or truncated)

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| User passes filename with full path instead of just the filename | CLI confusion | In `daily_run.py`, use `Path(args.file).name` to always extract just the filename, and join with the known `RAW_FOLDER` |
| `--auto` picks a non-option-chain CSV | Other files sitting in the folder | Filter with `glob("option_chain_*.csv")`, sort by `mtime`, take the newest |
| File is for a date that was already processed | Ran script twice | Check for existing output file; print a clear warning; require `--force` flag to overwrite |
| File is an `.xlsx` not a `.csv` | User downloaded the wrong format | Detect extension; print a specific message: "NSE exports must be saved as CSV before running this script" |
| Weekend / holiday date in filename | Data entry error | Warn but do not block — NSE sometimes releases expiry-day data on days that are technically weekends in some calendar libraries |

---

## 4. Step B — Format & Clean (`option_data_formatting.py`)

This is the existing script from the other project. It needs three changes before it can be used here.

### Change 1: Fix hardcoded paths

Replace the two absolute Windows paths at the top with project-relative paths using `pathlib`:

```python
# REMOVE this:
RAW_FOLDER = r'C:\Users\Aditya\Downloads\...'
PROCESSED_FOLDER = r'C:\Users\Aditya\...'

# ADD this:
from pathlib import Path
PROJECT_ROOT     = Path(__file__).resolve().parent
RAW_FOLDER       = PROJECT_ROOT / "data" / "option_chain_raw"
PROCESSED_FOLDER = PROJECT_ROOT / "data" / "option_chain_processed"
```

### Change 2: Fix the filename parser

The current script splits on `-` and extracts `parts[3]` and `parts[4:]`, which was designed for the old project's filename format. The actual file here is named `option_chain_BANKNIFTY-28-Apr-2026.csv`. The correct parser:

```python
# REMOVE this:
parts = name_without_ext.split('-')
asset_name = parts[3]
date = '-'.join(parts[4:])

# ADD this:
# Strip "option_chain_" prefix, then split on first dash
remainder  = name_without_ext.removeprefix('option_chain_')
# remainder = "BANKNIFTY-28-Apr-2026"
first_dash = remainder.index('-')
asset_name = remainder[:first_dash]        # "BANKNIFTY"
date_str   = remainder[first_dash + 1:]   # "28-Apr-2026"
```

**Note:** The output filename remains `option_chain_{asset_name}-{date_str}.csv` which is the same as the input — that is intentional. The processed file in `option_chain_processed/` is the cleaned version of the same file.

### Change 3: Refactor to a callable function

The script currently runs top-to-bottom on import. `daily_run.py` needs to call it as a function. Wrap everything in a function and add a CLI guard:

```python
def format_option_chain(input_filename: str) -> Path:
    """
    Formats and cleans a raw option chain CSV.
    Returns the path to the processed output file.
    """
    # ... all existing logic here ...
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python option_data_formatting.py <filename>")
        sys.exit(1)
    out = format_option_chain(sys.argv[1])
    print(f"Done: {out}")
```

### Change 4: Add a column count guard

After `df = df.dropna(axis=1, how='all')`, assert the column count before renaming, since NSE occasionally adds/removes a blank column:

```python
expected_cols = 21
if len(df.columns) != expected_cols:
    raise ValueError(
        f"Expected {expected_cols} columns after dropping empty ones, "
        f"got {len(df.columns)}. Actual columns: {list(df.columns)}. "
        f"Check if NSE changed the export format."
    )
```

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| `removeprefix` not available | Python < 3.9 | Use `name_without_ext.replace('option_chain_', '', 1)` instead |
| Column count is 22 not 21 | NSE added an extra blank column that wasn't fully empty | Change the `dropna` threshold: `df = df.dropna(axis=1, thresh=3)` — drop columns with fewer than 3 non-null values |
| Strike column still has `.00` suffix after cleaning | Some NSE exports use `"43000.00"` not `"43,000.00"` | The existing `.str.replace('.00', '')` handles this, but it will also corrupt strikes like `43500.00` — use `astype(float).astype(int)` instead of string replacement |
| `interpolate` crashes on all-NaN IV column | Expiry day data — all contracts ITM/ATM, no IV data for OTM | Add `if df['call_iv'].notna().sum() < 3:` guard before interpolating; fill with ATM IV or raise an informative error |
| Output file already exists | Re-run of same day | Overwrite silently — the processed CSV is always deterministic from the raw CSV |

---

## 5. Step C — Single-Day Feature Engineering (`daily_features.py`)

This is the most technically complex step. `preprocess.py` builds features using the full historical dataset (needed for things like `HV_20` rolling window and cross-sectional normalisation). For a single new day, those same features need to be computed using today's data + recent history.

### What to build

A new module `daily_features.py` with a single public function:

```python
def compute_daily_features(processed_csv_path: Path, trade_date: date) -> pd.DataFrame:
    """
    Takes a single day's processed option chain CSV.
    Returns a DataFrame with all 16 model features ready for inference.
    """
```

### How each feature group is computed for a single new day

**Contract structure features** — computed purely from today's data, no history needed:
- `option_type_encoded`: 1 for CE, 0 for PE (from `asset` column and row structure)
- `strike`: already in processed CSV
- `DTE`: `(expiry_date - trade_date).days`
- `moneyness`: `strike / spot`. Spot must be inferred (see below)
- `abs_moneyness`: `abs(moneyness - 1)`
- `moneyness_sq`: `moneyness ** 2`

**How to infer spot price from the processed CSV:** The processed CSV does not have a spot column. Replicate the logic from `preprocess.py`: find the strike where `abs(call_ltp - put_ltp)` is minimised (put-call parity ATM proxy), or use the strike with highest combined OI as the ATM proxy. This will not be exact but will be within ±50 points.

```python
# ATM proxy via put-call parity
df['parity_diff'] = abs(df['call_ltp'] - df['put_ltp'])
atm_strike = df.loc[df['parity_diff'].idxmin(), 'strike']
spot = atm_strike  # close enough for moneyness computation
```

**IV features** — computed from today's data:
- `IV` (per contract): Black-Scholes brentq inversion. This exact logic is already in `preprocess.py` inside the `compute_iv_for_row` function. Extract it into a shared utility (see Note below).
- `ATM_IV`: mean IV of contracts within ±2% of ATM moneyness
- `IV_relative`: `IV - ATM_IV`
- `IV_rank`: percentile rank of IV within today's data (`df['IV'].rank(pct=True)`)
- `IV_HV_Spread`: `ATM_IV - HV_20` (HV_20 from history — see below)

**Regime features:**
- `Skew`: `PE_0.95_IV - CE_1.05_IV` — computed from today's data, same logic as `preprocess.py`
- `TS_Slope`: `near_expiry_IV - far_expiry_IV` — computed from today's data
- `log_HV_20`: requires the last 20 trading days of spot prices

**How to get HV_20 for a new day:** Load `data/features/cross_sectional.parquet`, filter for the most recent dates, extract unique daily spot prices, compute 20-day rolling std, annualise. Today's spot (inferred above) can be appended to extend the series by one day.

```python
cs = pd.read_parquet("data/features/cross_sectional.parquet")
spot_series = cs.groupby('Date')['spot'].first().sort_index()
# Append today
spot_series[pd.Timestamp(trade_date)] = spot
log_returns = np.log(spot_series / spot_series.shift(1)).dropna()
hv_20 = log_returns.rolling(20).std().iloc[-1] * np.sqrt(252)
```

**Liquidity features:**
- `OI_normalized`: `call_oi / daily_mean_oi` and `put_oi / daily_mean_oi`. Daily mean is from today's data only — `df['call_oi'].mean()`
- `Volume_normalized`: same pattern
- `log_OI`: `np.log(call_oi + 1)` / `np.log(put_oi + 1)`

### Note on code reuse

`preprocess.py` contains the Black-Scholes IV computation logic inside it. Rather than duplicating it, create a shared file `bs_utils.py` (or `option_math.py`) that both `preprocess.py` and `daily_features.py` import from. If modifying `preprocess.py` is out of scope (to avoid breaking the historical pipeline), copy the relevant functions into `daily_features.py` with a comment marking the source.

### Output format

The function returns a DataFrame with these columns (matching `cross_sectional.parquet` exactly):
```
Date, strike, expiry, option_type, close, spot, DTE, moneyness,
abs_moneyness, moneyness_sq, IV, ATM_IV, IV_relative, IV_rank,
IV_HV_Spread, Skew, TS_Slope, HV_20, log_HV_20, OI_normalized,
Volume_normalized, log_OI, option_type_encoded
```

It also keeps `call_oi`, `put_oi`, `call_volume`, `put_volume` as side columns (not fed to the model) for the liquidity filter in Step E.

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| `brentq` fails to converge for some contracts | Option price below intrinsic value (data error or deep ITM) | Wrap each IV computation in try/except; set IV = NaN on failure; drop those rows before inference |
| `expiry_date` column in processed CSV is a string `"28-Apr-2026"` | Formatting script stores it as string | Parse with `pd.to_datetime(df['expiry_date'], format='%d-%b-%Y')` |
| HV_20 is NaN because cross_sectional.parquet is empty or missing | Pipeline was never run / first use | Fall back to a hardcoded recent value (e.g. 0.15) with a clear warning; or read HV_20 from the last row of cross_sectional.parquet |
| Spot inference is wrong by a large margin | High-volatility day where put-call parity breaks down | Add a sanity check: inferred spot must be within 10% of the previous day's spot (read from cross_sectional.parquet). If not, warn the user |
| `IV_rank` is computed only on today's data, not cross-sectional vs historical | Single day = compressed rank range | This is correct and intentional — IV_rank is within-day percentile in the original pipeline too. No issue. |
| Moneyness filter (0.80–1.20) removes too many contracts | Far expiry contracts or extreme market day | Apply filter after feature computation; log how many rows were dropped. If > 80% dropped, warn user |
| `log_OI` computed on CE/PE separately vs combined | Original pipeline may combine them | Check `preprocess.py`'s `log_OI` computation. If it uses `OI_NO_CON` (total OI regardless of type), replicate that. |

---

## 6. Step D — Model Inference

**File:** `daily_run.py` (the inference section)

### What to build

Load the appropriate model and run predictions. The walk-forward model should always be preferred.

```python
import joblib
import numpy as np

FEATURE_COLS = [
    "option_type_encoded", "DTE", "moneyness", "abs_moneyness", "moneyness_sq",
    "IV", "ATM_IV", "IV_relative", "IV_rank", "IV_HV_Spread",
    "Skew", "TS_Slope", "log_HV_20",
    "OI_normalized", "Volume_normalized", "log_OI",
]

def run_inference(features_df: pd.DataFrame, trade_date: date) -> pd.DataFrame:
    model = load_best_model(trade_date)   # see below

    # Apply same pre-training transforms as train.py
    X = features_df[FEATURE_COLS].copy()
    X['log_OI']    = np.log(features_df['call_oi'].clip(0) + 1)   # recompute from raw
    X['log_HV_20'] = np.log(features_df['HV_20'].clip(0) + 1)
    # Clip outliers at 1st/99th percentile using TRAINING distribution
    # (See note on percentile clipping below)

    pred_log_price     = model.predict(X)
    features_df['predicted_price'] = np.exp(pred_log_price) - 1
    features_df['mispricing']      = features_df['close'] - features_df['predicted_price']
    return features_df
```

### How to select the best walk-forward model

Walk-forward models are named `wf_model_{YYYY-MM}.joblib` (e.g. `wf_model_2026-03.joblib`). Load the one with the most recent month that is on or before `trade_date`:

```python
def load_best_model(trade_date: date) -> object:
    model_dir = PROJECT_ROOT / "models"
    wf_models = sorted(model_dir.glob("wf_model_*.joblib"))
    # Filter to models trained on data up to trade_date's month
    trade_period = f"{trade_date.year}-{trade_date.month:02d}"
    eligible = [m for m in wf_models if m.stem.split('_', 2)[2] <= trade_period]
    if eligible:
        return joblib.load(eligible[-1])   # most recent
    # Fallback to static model
    print("Warning: No walk-forward model found. Using static model.")
    return joblib.load(model_dir / "xgb_mispricing.joblib")
```

### Note on percentile clipping

`train.py` clips `IV_relative`, `OI_normalized`, `Volume_normalized` at the training set's 1st/99th percentiles. These clip bounds must be the same values used during training — they must not be recomputed on today's single-day data (which would give different bounds and distort predictions).

**Solution:** At the end of `train.py` (after training), save the clip bounds to a small JSON file:

```python
# Add to train.py, after computing training percentiles:
import json
clip_bounds = {
    "IV_relative":       [float(X_train["IV_relative"].quantile(0.01)),
                          float(X_train["IV_relative"].quantile(0.99))],
    "OI_normalized":     [float(X_train["OI_normalized"].quantile(0.01)),
                          float(X_train["OI_normalized"].quantile(0.99))],
    "Volume_normalized": [float(X_train["Volume_normalized"].quantile(0.01)),
                          float(X_train["Volume_normalized"].quantile(0.99))],
}
with open(PROJECT_ROOT / "models" / "clip_bounds.json", "w") as f:
    json.dump(clip_bounds, f, indent=2)
```

Then in `daily_run.py`, load and apply these bounds:

```python
with open(PROJECT_ROOT / "models" / "clip_bounds.json") as f:
    clip_bounds = json.load(f)
for col, (lo, hi) in clip_bounds.items():
    X[col] = X[col].clip(lo, hi)
```

If `clip_bounds.json` doesn't exist (pipeline ran before this change), fall back to clipping at the column's own 1st/99th percentile with a warning.

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| `wf_model_*.joblib` was trained on an older XGBoost version | XGBoost version mismatch between training env and daily env | Pin `xgboost` version in requirements. If loading fails with pickle error, retrain or use `model.save_model('model.json')` format which is version-independent |
| Feature column order doesn't match what the model expects | DataFrame column order varies | Always select with `X = df[FEATURE_COLS]` in the exact FEATURE_COLS order defined in train.py |
| NaN in feature matrix causes XGBoost to output NaN predictions | IV failed to converge for some rows | Drop rows where ANY feature in FEATURE_COLS is NaN before predict; re-attach to full df with `merge` for signal output |
| `clip_bounds.json` doesn't exist | Old pipeline run | Warn and proceed with no clipping; this is the fallback and produces slightly noisier but not broken predictions |

---

## 7. Step E — Signal Generation

**File:** `daily_run.py` (signal section)

### What to build

Z-score computation and signal assignment for the single day.

```python
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    # Liquidity filter (same as retrain.py)
    liquid = (
        (df['OI_normalized']     > 0.5) &
        (df['Volume_normalized'] > 0.5) &
        (df['DTE']               > 5)
    )
    df_liquid = df[liquid].copy()

    # Z-score: cross-sectional on today's data only
    daily_mean = df_liquid['mispricing'].mean()
    daily_std  = df_liquid['mispricing'].std()

    # Floor std to avoid divide-by-zero on quiet days (from retrain.py: Z_SCORE_STD_FLOOR = 0.5)
    Z_SCORE_STD_FLOOR = 0.5
    daily_std = max(daily_std, Z_SCORE_STD_FLOOR)

    df_liquid['z_score'] = (df_liquid['mispricing'] - daily_mean) / daily_std

    # Signals
    df_liquid['signal'] = 'HOLD'
    df_liquid.loc[df_liquid['z_score'] >  2, 'signal'] = 'SELL'
    df_liquid.loc[df_liquid['z_score'] < -2, 'signal'] = 'BUY'

    return df_liquid
```

**Important:** The z-score here is computed cross-sectionally across all liquid contracts on today's single day. This is the same methodology as the historical pipeline. Because we only have today's ~150–300 contracts (not 70,000 historical rows), the distribution will naturally be tighter. This is expected and correct.

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| All z-scores are near zero, no BUY/SELL signals generated | Low-volatility day with very small mispricing spread | This is real — the model says everything is fairly priced today. Check `daily_std` in the console output. If it's below 1.0, that's the reason. |
| `daily_std` hits the floor (0.5) and forces artificial signals | Very quiet market | Add a note in the console output: "WARNING: daily_std = {x:.3f} floored to 0.5. Signals may be less reliable." |
| Liquidity filter removes too many contracts | Gap-up/down day where volume is concentrated in few strikes | Consider loosening the filter for expiry-day data (`DTE == 0` contracts) or log the liquid contract count |
| Signal column in output conflicts with historical signal format | Historical uses uppercase BUY/SELL, new might differ | Always uppercase: `df['signal'] = df['signal'].str.upper()` |

---

## 8. Step F — Persist Daily Results

**File:** `daily_run.py` (save section)

### What to build

Save two files per trading day to `outputs/daily/`:

```python
def save_daily_results(df_signals: pd.DataFrame, trade_date: date):
    daily_dir = PROJECT_ROOT / "outputs" / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    date_str = trade_date.strftime("%Y-%m-%d")

    # Full predictions parquet (all liquid contracts, with z_score)
    pred_path = daily_dir / f"{date_str}_predictions.parquet"
    df_signals.to_parquet(pred_path, index=False)

    # Signals CSV (BUY/SELL only — the actionable output)
    signals_only = df_signals[df_signals['signal'].isin(['BUY', 'SELL'])]
    sig_path = daily_dir / f"{date_str}_signals.csv"
    signals_only.to_csv(sig_path, index=False)

    print(f"\n--- Daily Run Complete ---")
    print(f"Date         : {date_str}")
    print(f"Liquid rows  : {len(df_signals)}")
    print(f"BUY signals  : {(df_signals['signal']=='BUY').sum()}")
    print(f"SELL signals : {(df_signals['signal']=='SELL').sum()}")
    print(f"Saved to     : {pred_path}")
    print(f"             : {sig_path}")
```

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| `outputs/daily/` doesn't exist | First run | `mkdir(parents=True, exist_ok=True)` handles this |
| Parquet write fails | `pyarrow` not installed | Add `pyarrow` to requirements; for fallback, save as CSV instead |
| Date in filename and date in DataFrame don't match | Off-by-one if running after midnight | Parse `trade_date` strictly from the filename in Step A, not from `datetime.today()` |

---

## 9. Step G — Append to Master Outputs

**File:** `daily_run.py` (append section)

### What to build

Append today's rows to the master output files so the existing dashboard panels automatically include today's data.

```python
def append_to_master(df_signals: pd.DataFrame, trade_date: date):
    # --- wf_predictions.parquet ---
    pred_master = PROJECT_ROOT / "outputs" / "wf_predictions.parquet"
    if pred_master.exists():
        master = pd.read_parquet(pred_master)
        # Remove any existing rows for this date (idempotent)
        master = master[master['Date'] != pd.Timestamp(trade_date)]
        updated = pd.concat([master, df_signals], ignore_index=True)
    else:
        updated = df_signals
    updated.to_parquet(pred_master, index=False)

    # --- wf_trading_signals.csv ---
    sig_master = PROJECT_ROOT / "outputs" / "wf_trading_signals.csv"
    signals_only = df_signals[df_signals['signal'].isin(['BUY', 'SELL'])]
    if sig_master.exists():
        master_sig = pd.read_csv(sig_master)
        master_sig['date'] = pd.to_datetime(master_sig['date'])
        master_sig = master_sig[master_sig['date'] != pd.Timestamp(trade_date)]
        updated_sig = pd.concat([master_sig, signals_only], ignore_index=True)
    else:
        updated_sig = signals_only
    updated_sig.to_csv(sig_master, index=False)
```

**Critical: column alignment.** The `df_signals` DataFrame from Step E must have column names that exactly match the schema of the existing master files. Before appending, check column names against the master file and rename/drop as needed.

```python
# Column name check
master_cols = set(pd.read_parquet(pred_master).columns)
daily_cols  = set(df_signals.columns)
missing = master_cols - daily_cols
extra   = daily_cols - master_cols
if missing:
    print(f"WARNING: daily data missing columns: {missing}. Filling with NaN.")
    for col in missing:
        df_signals[col] = np.nan
if extra:
    print(f"INFO: dropping extra daily columns not in master: {extra}")
    df_signals = df_signals.drop(columns=list(extra))
```

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| Column `Date` in master is uppercase, `date` in daily is lowercase | Inconsistent naming across pipeline scripts | Normalise to lowercase in `daily_run.py` before appending: `df_signals.rename(columns={'Date': 'date'}, inplace=True)` |
| Master parquet grows without bound over months | Daily appends accumulate | This is intentional and fine — 300 rows/day × 250 trading days = 75,000 rows/year, tiny for parquet |
| Append corrupts the master file if the script crashes mid-write | No atomic write | Write to a temp file first, then `os.replace(temp, master)` — atomic on all OS |
| `wf_predictions.parquet` missing columns that daily data adds | Schema evolves | The column alignment check above (with NaN fill) handles this |

---

## 10. Dashboard — Panel 5: Daily Results

**File:** `dashboard/components/daily_panel.py`

This is the new panel added to the existing dashboard. It reads exclusively from `outputs/daily/` (not the master files) and shows a deep-dive on today's single-day run.

### Data loader additions to `dashboard/data_loader.py`

```python
@st.cache_data
def load_daily_predictions(trade_date: str) -> pd.DataFrame:
    """trade_date: 'YYYY-MM-DD' string"""
    path = BASE / "outputs" / "daily" / f"{trade_date}_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_data
def load_daily_signals(trade_date: str) -> pd.DataFrame:
    path = BASE / "outputs" / "daily" / f"{trade_date}_signals.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_daily_available_dates() -> list[str]:
    daily_dir = BASE / "outputs" / "daily"
    if not daily_dir.exists():
        return []
    files = sorted(daily_dir.glob("*_signals.csv"), reverse=True)
    return [f.stem.replace("_signals", "") for f in files]
```

### Panel 5 component: `dashboard/components/daily_panel.py`

The daily panel contains 5 sub-sections:

**Sub-section 1: Day Summary Banner**

```
┌────────────────────────────────────────────────────────────┐
│  📅 Daily Run: 28 Apr 2026   │  Model: Walk-Forward       │
│  Liquid Contracts: 241       │  Run at: 16:42             │
│  BUY: 7   SELL: 11   HOLD: 223                            │
└────────────────────────────────────────────────────────────┘
```
Show using `st.info()` or `st.container()` with columns.

**Sub-section 2: Full Signals Table (not just top 10)**

Unlike the existing Panel 2 which shows top 10 from the full study period, this shows ALL BUY and SELL signals from today sorted by abs(z_score). Use `st.dataframe` with the full `daily_signals` DataFrame.

Columns to show:
`strike | expiry | type | close_price | predicted_price | mispricing | z_score | signal`

Color: green rows for BUY, red for SELL. Use the same `color_signal` styling from `signals_table.py`.

**Sub-section 3: Volatility Smile Plot**

Plot `IV` vs `moneyness` for today's data, split by option type (CE vs PE). This shows whether today's vol surface looks normal or skewed.

```python
fig = go.Figure()
for opt_type, color in [('CE', '#4a9eff'), ('PE', '#ff9e4a')]:
    subset = day_pred[day_pred['option_type'] == opt_type].sort_values('moneyness')
    fig.add_trace(go.Scatter(
        x=subset['moneyness'],
        y=subset['IV'],
        mode='lines+markers',
        name=opt_type,
        line=dict(color=color)
    ))
fig.add_vline(x=1.0, line_dash="dot", line_color="#888", annotation_text="ATM")
```

**Sub-section 4: Mispricing Scatter Plot**

X-axis: moneyness. Y-axis: mispricing. Color: z_score (diverging colorscale centred at 0). Size: OI_normalized. This visually shows where on the vol smile the mispricing is concentrated.

```python
fig = go.Figure(go.Scatter(
    x=day_pred['moneyness'],
    y=day_pred['mispricing'],
    mode='markers',
    marker=dict(
        color=day_pred['z_score'],
        colorscale='RdYlGn_r',   # red = high z (SELL), green = low z (BUY)
        cmin=-3, cmax=3,
        size=day_pred['OI_normalized'].clip(1, 8) * 2,
        colorbar=dict(title='Z-Score'),
    ),
))
fig.add_hline(y=0, line_color="#555")
```

**Sub-section 5: DTE Breakdown**

A bar chart showing how many BUY/SELL signals fall into DTE buckets: `[1–7], [8–14], [15–30], [31–60], [61–90]`. This helps the user understand whether the mispricing is concentrated in near-term or far-term expiries.

```python
bins   = [0, 7, 14, 30, 60, 90]
labels = ['1–7', '8–14', '15–30', '31–60', '61–90']
day_signals['dte_bucket'] = pd.cut(day_signals['DTE'], bins=bins, labels=labels)
breakdown = day_signals.groupby(['dte_bucket', 'signal']).size().unstack(fill_value=0)
```

### Adding Panel 5 to `dashboard/app.py`

The daily panel is shown as a new section below the existing four panels. Add a tab layout to keep the UI clean:

```python
# Replace the flat layout in app.py with tabs
tab_hist, tab_daily = st.tabs(["📈 Historical Analysis", "📅 Today's Run"])

with tab_hist:
    render_kpi_cards(day_signals)
    st.divider()
    col_left, col_right = st.columns([6, 4])
    with col_left:
        render_signals_table(day_signals)
    with col_right:
        render_zscore_chart(day_pred)
    st.divider()
    render_performance_chart(model_key)

with tab_daily:
    render_daily_panel(model_key)   # new component
```

`render_daily_panel` internally calls `get_daily_available_dates()`, shows a date selector for daily results (separate from the historical date selector in the sidebar), and renders the 5 sub-sections.

### Potential Problems & Solutions

| Problem | Cause | Fix |
|---|---|---|
| Daily panel shows no data on first load before `daily_run.py` has been run | No files in `outputs/daily/` | Show `st.info("No daily results yet. Run daily_run.py after market close to populate this panel.")` |
| `load_daily_predictions` is called with the current calendar date but that date's file doesn't exist | Weekend or holiday | `get_daily_available_dates()` lists only dates with actual files — use this list to populate the selector, not a calendar picker |
| IV smile plot looks jagged | Some IV values are interpolated, others are model-computed | This is cosmetic; the existing `option_data_formatting.py` interpolates IV for missing values, so the chart reflects that |
| Mispricing scatter has a few extreme outliers that compress the chart | Deep OTM contracts slipping through filters | Clip mispricing for display: `day_pred['mispricing'].clip(-500, 500)` — do not clip the actual data |
| Tab layout breaks existing dashboard | Streamlit tabs is available in ≥ 1.12 | Add `streamlit>=1.32.0` to requirements (already in dashboard_plan.md) |

---

## 11. Complete File Checklist for Claude Code

### New files to create

| File | Purpose |
|---|---|
| `daily_run.py` | Main entry point orchestrating Steps A–G |
| `daily_features.py` | Single-day feature engineering (Step C) |
| `dashboard/components/daily_panel.py` | Panel 5 component |

### Files to modify

| File | What changes |
|---|---|
| `option_data_formatting.py` | Fix paths, fix filename parser, refactor to function, add column guard |
| `train.py` | Add 3 lines to save `models/clip_bounds.json` after training |
| `dashboard/app.py` | Wrap existing panels in `tab_hist`, add `tab_daily` |
| `dashboard/data_loader.py` | Add `load_daily_predictions`, `load_daily_signals`, `get_daily_available_dates` |
| `dashboard/requirements.txt` | Confirm `scikit-learn` is present (needed for R² computation) |

### New folders to create

| Folder | Created by |
|---|---|
| `data/daily_features/` | `daily_run.py` via `mkdir(parents=True, exist_ok=True)` |
| `outputs/daily/` | `daily_run.py` via `mkdir(parents=True, exist_ok=True)` |

---

## 12. Daily User Workflow (End State)

Once everything is built, the user's daily workflow is:

```
1. Download option chain CSV from NSE after 15:30 IST
   → Save as: option_chain_BANKNIFTY-{DD}-{Mon}-{YYYY}.csv
   → Drop into: data/option_chain_raw/

2. Open terminal in project root, activate .venv:
   > .venv\Scripts\activate  (Windows)
   > source .venv/bin/activate  (Mac/Linux)

3. Run:
   > python daily_run.py --auto
   (or: python daily_run.py --file option_chain_BANKNIFTY-28-Apr-2026.csv)

4. See console output:
   --- Daily Run Complete ---
   Date         : 2026-04-28
   Liquid rows  : 241
   BUY signals  : 7
   SELL signals : 11
   Saved to     : outputs/daily/2026-04-28_predictions.parquet
                : outputs/daily/2026-04-28_signals.csv

5. Open dashboard (if not already running):
   > streamlit run dashboard/app.py

6. Click "Today's Run" tab in the dashboard.
   → See full signal table, vol smile, mispricing scatter, DTE breakdown.
```

Total time from CSV download to dashboard view: **under 2 minutes**.

---

## 13. Order of Implementation

Implement strictly in this order. Each step depends on the previous one being testable.

1. **Fix `option_data_formatting.py`** — test it standalone: `python option_data_formatting.py option_chain_BANKNIFTY-28-Apr-2026.csv` and verify the processed CSV in `option_chain_processed/`
2. **Write `daily_features.py`** — test it standalone with the processed CSV, print the feature DataFrame, verify all 16 feature columns exist and have no unexpected NaNs
3. **Write Steps D + E in `daily_run.py`** — test inference and signal generation with the feature DataFrame, print signal counts to console
4. **Write Steps F + G in `daily_run.py`** — test file saving and master append, verify the parquet and CSV files are written correctly
5. **Write `daily_panel.py`** — test it in isolation with `st.write(load_daily_predictions("2026-04-28"))` first
6. **Update `dashboard/data_loader.py`** — add the three new loaders and test them
7. **Update `dashboard/app.py`** — integrate the tabs and verify the existing panels still work
8. **End-to-end test** — run the full `daily_run.py` then open the dashboard and verify Panel 5 shows correct data

---

## 14. Known Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| NSE changes the option chain CSV column order | Step B breaks | The column count guard in `option_data_formatting.py` will catch this immediately with an informative error |
| `wf_predictions.parquet` schema changes after a future pipeline rerun | Step G append produces misaligned columns | The column alignment check in Step G fills missing columns with NaN and drops extra columns |
| Model becomes stale as 2026 progresses beyond the training data | Signals become less reliable | This is an existing known limitation (README §15). The dashboard's R² trend chart (Panel 4) will visually show model degradation. When monthly R² drops below 0.80, it's time to retrain. |
| Two runs on the same day produce duplicate rows in master files | Step G idempotency issue | The `master = master[master['Date'] != pd.Timestamp(trade_date)]` line before concat makes every run idempotent |
| `clip_bounds.json` doesn't exist (train.py never updated) | Features clipped inconsistently | Graceful fallback: warn and use column's own percentiles. This is the worst-case fallback, not a crash. |
