# src/data_loader.py
# ─────────────────────────────────────────────
# All data loading goes here. Other modules
# call these functions — never pd.read_parquet
# directly in analysis files.
# ─────────────────────────────────────────────
import pandas as pd
from src.config import DATA_PROCESSED


def load_spot_rates() -> pd.DataFrame:
    """Monthly ECB AAA spot rates. Columns: spot_1y ... spot_30y"""
    return pd.read_parquet(DATA_PROCESSED / "ecb_spot_rates.parquet")


def load_forward_regression_data() -> pd.DataFrame:
    """Forward rates aligned with realized future spot rates. Built by Q2."""
    return pd.read_parquet(DATA_PROCESSED / "forward_regression_data.parquet")


def load_etf_prices() -> pd.DataFrame:
    """Monthly ETF price levels. Columns: govt_short, govt_mid, govt_long, corp_ig, equity"""
    return pd.read_parquet(DATA_PROCESSED / "etf_prices.parquet")


def load_etf_returns() -> pd.DataFrame:
    """
    Monthly bond log returns.
    Prefers Refinitiv iBoxx indices (2000-2025) if available.
    Falls back to iShares ETF proxies (2008/2009-2025) if not.
    Columns: govt_short_logret, govt_mid_logret, govt_long_logret,
             corp_ig_logret
    """
    refinitiv_govt = DATA_PROCESSED / "refinitiv_govt_returns.parquet"
    refinitiv_corp = DATA_PROCESSED / "refinitiv_corp_returns.parquet"

    if refinitiv_govt.exists() and refinitiv_corp.exists():
        print("  Using Refinitiv iBoxx data")
        govt = pd.read_parquet(refinitiv_govt)
        corp = pd.read_parquet(refinitiv_corp)
        return pd.concat([govt, corp], axis=1)
    else:
        print("  Using ETF proxy data (Refinitiv not available)")
        return pd.read_parquet(DATA_PROCESSED / "etf_returns.parquet")


def load_corp_oas() -> pd.Series:
    """
    Monthly corporate bond OAS spread (iBoxx EUR Corp IG).
    Only available when Refinitiv data has been fetched.
    Returns None if not available.
    Coverage: Jan 2000 – Dec 2025
    """
    path = DATA_PROCESSED / "refinitiv_corp_oas.parquet"
    if path.exists():
        return pd.read_parquet(path)["corp_oas"]
    else:
        print("  OAS data not available — run fetch_and_save_refinitiv()")
        return None


def load_msci_europe() -> pd.Series:
    """
    Monthly MSCI Europe log returns as a Series.
    Source: data/processed/msci_europe.parquet
    Built by data_import.py from MSCI_Europe_daily.csv (daily → month-end resample)
    Coverage: Jan 2000 – Dec 2025 (312 observations)
    """
    df = pd.read_parquet(DATA_PROCESSED / "msci_europe.parquet")
    return df["equity_logret"]


def load_all_asset_returns() -> pd.DataFrame:
    etf  = load_etf_returns()
    msci = load_msci_europe()

    # Safety check — print available columns so mismatches are visible
    print(f"  ETF columns available: {list(etf.columns)}")

    etf.index  = etf.index.to_period("M").to_timestamp("M")
    msci.index = msci.index.to_period("M").to_timestamp("M")

    df = pd.DataFrame({
        "equity":     msci,
        "govt_short": etf["govt_short_logret"],
        "govt_mid":   etf["govt_mid_logret"],
        "govt_long":  etf["govt_long_logret"],
        "corp_ig":    etf["corp_ig_logret"],
    }).dropna()

    print(f"  Common sample: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Observations: {len(df)}")
    return df

def load_govt_returns_refinitiv() -> pd.Series | None:
    """
    Monthly iBoxx EUR Sovereigns (all maturities) log returns.
    Coverage: Jan 2000 – Dec 2025
    Returns None if Refinitiv data not available.
    """
    # Try separate file first (used by load_etf_returns)
    path = DATA_PROCESSED / "refinitiv_govt_returns.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Return first column as Series
        return df.iloc[:, 0]

    # Try combined file (from data_import.py)
    path2 = DATA_PROCESSED / "refinitiv_returns.parquet"
    if path2.exists():
        df = pd.read_parquet(path2)
        if "govt_all_logret" in df.columns:
            return df["govt_all_logret"]

    return None


def load_corp_returns_refinitiv() -> pd.Series | None:
    """
    Monthly iBoxx EUR Corporate IG log returns.
    Coverage: Jan 2000 – Dec 2025
    Returns None if Refinitiv data not available.
    """
    path = DATA_PROCESSED / "refinitiv_corp_returns.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        return df.iloc[:, 0]

    path2 = DATA_PROCESSED / "refinitiv_returns.parquet"
    if path2.exists():
        df = pd.read_parquet(path2)
        if "corp_ig_logret" in df.columns:
            return df["corp_ig_logret"]


    return None

def load_govt_duration_returns_refinitiv() -> pd.DataFrame | None:
    """
    Monthly iBoxx EUR Sovereign maturity sub-index log returns.
    Columns: govt_short_logret, govt_mid_logret, govt_long_logret
    Coverage: ~2000–2025
    Returns None if not available.
    """
    path = DATA_PROCESSED / "refinitiv_returns.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        needed = ["govt_short_logret", "govt_mid_logret", "govt_long_logret"]
        if all(c in df.columns for c in needed):
            return df[needed]
    return None
