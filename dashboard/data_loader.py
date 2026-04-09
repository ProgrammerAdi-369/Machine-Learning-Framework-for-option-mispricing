import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

BASE = Path(__file__).resolve().parent.parent

# Maps actual column names in files → dashboard expected names
COLUMN_MAP = {
    "Date": "date",
    "close": "close_price",
    "expiry_date": "expiry",
    "Close": "close_price",
    "Predicted": "predicted_price",
    "ZScore": "z_score",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})


@st.cache_data
def load_predictions(model: str = "static") -> pd.DataFrame:
    """
    model: "static" -> full_predictions.parquet
           "wf"     -> wf_predictions.parquet (falls back to static if missing)
    """
    fname = "wf_predictions.parquet" if model == "wf" else "full_predictions.parquet"
    path = BASE / "outputs" / fname

    if not path.exists():
        if model == "wf":
            # Graceful fallback to static
            path = BASE / "outputs" / "full_predictions.parquet"
            if not path.exists():
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    try:
        df = pd.read_parquet(path)
        df = normalize_columns(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["date"])
        dropped = before - len(df)
        if dropped:
            print(f"[data_loader] Dropped {dropped} rows with invalid date")
        # Normalize signal column if present
        if "signal" in df.columns:
            df["signal"] = df["signal"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        print(f"[data_loader] Failed to load predictions: {e}")
        return pd.DataFrame()


@st.cache_data
def load_signals(model: str = "static") -> pd.DataFrame:
    fname = "wf_trading_signals.csv" if model == "wf" else "trading_signals.csv"
    path = BASE / "outputs" / fname

    if not path.exists():
        if model == "wf":
            path = BASE / "outputs" / "trading_signals.csv"
            if not path.exists():
                return pd.DataFrame()
        else:
            return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        df = normalize_columns(df)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["date"])
        dropped = before - len(df)
        if dropped:
            print(f"[data_loader] Dropped {dropped} signal rows with invalid date")
        # Normalize signal values
        if "signal" in df.columns:
            df["signal"] = df["signal"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        print(f"[data_loader] Failed to load signals: {e}")
        return pd.DataFrame()


@st.cache_data
def load_monthly_r2(model: str = "static") -> pd.DataFrame:
    df = load_predictions(model)
    if df.empty or "close_price" not in df.columns or "predicted_price" not in df.columns:
        return pd.DataFrame(columns=["month", "r2"])

    df = df.dropna(subset=["close_price", "predicted_price"])
    df["month"] = df["date"].dt.to_period("M")

    def safe_r2(g):
        g = g.dropna(subset=["close_price", "predicted_price"])
        if len(g) < 10:
            return None
        try:
            return r2_score(g["close_price"], g["predicted_price"])
        except Exception:
            return None

    result = (
        df.groupby("month")
        .apply(safe_r2)
        .dropna()
        .reset_index()
    )
    result.columns = ["month", "r2"]
    result["month"] = result["month"].astype(str)
    return result


def get_available_dates(df: pd.DataFrame) -> list:
    if df.empty or "date" not in df.columns:
        return []
    return sorted(df["date"].dt.date.unique().tolist(), reverse=True)


# ── Daily run loaders ─────────────────────────────────────────────────────────

@st.cache_data
def load_daily_predictions(trade_date: str) -> pd.DataFrame:
    """trade_date: 'YYYY-MM-DD' string"""
    path = BASE / "outputs" / "daily" / f"{trade_date}_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        df = normalize_columns(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "signal" in df.columns:
            df["signal"] = df["signal"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        print(f"[data_loader] Failed to load daily predictions: {e}")
        return pd.DataFrame()


@st.cache_data
def load_daily_signals(trade_date: str) -> pd.DataFrame:
    """trade_date: 'YYYY-MM-DD' string"""
    path = BASE / "outputs" / "daily" / f"{trade_date}_signals.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return pd.DataFrame()
        df = normalize_columns(df)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if "signal" in df.columns:
            df["signal"] = df["signal"].astype(str).str.upper().str.strip()
        return df
    except Exception as e:
        print(f"[data_loader] Failed to load daily signals: {e}")
        return pd.DataFrame()


def get_daily_available_dates() -> list[str]:
    """Return list of 'YYYY-MM-DD' strings for dates with daily signal files."""
    daily_dir = BASE / "outputs" / "daily"
    if not daily_dir.exists():
        return []
    files = sorted(daily_dir.glob("*_predictions.parquet"), reverse=True)
    return [f.stem.replace("_predictions", "") for f in files]
