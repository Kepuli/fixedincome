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


