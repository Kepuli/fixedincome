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

def load_govt_returns() -> pd.DataFrame:
    """Monthly govt bond ETF log returns by maturity bucket."""
    return pd.read_parquet(DATA_PROCESSED / "govt_bond_returns.parquet")

def load_corp_returns() -> pd.DataFrame:
    """Monthly corporate bond ETF log returns."""
    return pd.read_parquet(DATA_PROCESSED / "corp_bond_returns.parquet")

def load_equity_returns() -> pd.DataFrame:
    """Monthly MSCI Europe log returns."""
    return pd.read_parquet(DATA_PROCESSED / "equity_returns.parquet")

def load_forward_regression_data() -> pd.DataFrame:
    """Forward rates aligned with realized future spot rates."""
    return pd.read_parquet(DATA_PROCESSED / "forward_regression_data.parquet")

def load_etf_prices() -> pd.DataFrame:
    """Monthly ETF price levels. Columns: govt_short, govt_mid, govt_long, corp_ig, equity"""
    return pd.read_parquet(DATA_PROCESSED / "etf_prices.parquet")

def load_etf_returns() -> pd.DataFrame:
    """Monthly ETF log returns. Columns: *_logret"""
    return pd.read_parquet(DATA_PROCESSED / "etf_returns.parquet")


### FOR q5, DOSENT WORK
def load_msci_world() -> pd.Series:
    """
    Monthly MSCI World log returns.
    Source: data/processed/msci_world.parquet
    Built by data_import.py from MSCIworld_raw_2000_to_2025.xlsx
    Coverage: 2000-01 to 2025-12
    """
    import pandas as pd
    from src.config import DATA_PROCESSED
    df = pd.read_parquet(DATA_PROCESSED / "msci_world.parquet")
    return df["equity_logret"]


def load_all_asset_returns() -> pd.DataFrame:
    """
    Convenience loader: all ETF log returns + MSCI World in one DataFrame.
    Columns: equity, govt_short, govt_mid, govt_long, corp_ig
    Common sample only (rows with any NaN dropped).
    """
    import pandas as pd
    etf  = load_etf_returns()
    msci = load_msci_world()

    df = pd.DataFrame({
        "equity":     msci,
        "govt_short": etf["govt_short_logret"],
        "govt_mid":   etf["govt_mid_logret"],
        "govt_long":  etf["govt_long_logret"],
        "corp_ig":    etf["corp_ig_logret"],
    }).dropna()

    return df
