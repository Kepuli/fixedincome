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
    Monthly ETF log returns.
    Columns: govt_short_logret, govt_mid_logret, govt_long_logret,
             corp_ig_logret, equity_logret
    Coverage: govt/equity ~2008, corp_ig ~2009
    """
    return pd.read_parquet(DATA_PROCESSED / "etf_returns.parquet")


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
    """
    Convenience loader: all ETF log returns + MSCI Europe aligned to common sample.
    Normalizes both indices to month-end before joining to avoid date mismatch.
    Columns: equity, govt_short, govt_mid, govt_long, corp_ig
    Coverage: common sample only (rows with any NaN dropped, starts ~2009-05)
    """
    etf  = load_etf_returns()
    msci = load_msci_europe()

    # Normalize to month-end — ETF dates are month-start, MSCI dates are month-end
    etf.index  = etf.index.to_period("M").to_timestamp("M")
    msci.index = msci.index.to_period("M").to_timestamp("M")

    df = pd.DataFrame({
        "equity":     msci,
        "govt_short": etf["govt_short_logret"],
        "govt_mid":   etf["govt_mid_logret"],
        "govt_long":  etf["govt_long_logret"],
        "corp_ig":    etf["corp_ig_logret"],
    }).dropna()

    return df