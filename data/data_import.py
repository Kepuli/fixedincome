import refinitiv.data as rd
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

DATA_PROCESSED = Path("C:/Users/tuoma/PycharmProjectsfixedincome/data/processed")
DATA_PROCESSED.mkdir(exist_ok=True)  # creates the folder if it doesn't exist


##########################
### SELECT WHAT TO IMPORT!
##########################

import_ecb_spot = False
import_etf_data = False
import_msci = True

##########################



# ── ECB Spot Rates ────────────────────────────────────────────

def fetch_and_save_ecb_spot(rating: str = "A"):
    """
    Fetch ECB AAA spot rates for all maturities and save to parquet.
    rating: A=AAA, C=all ratings
    Maturities: 1Y, 2Y, 3Y, 4Y, 5Y, 10Y, 30Y
    NOTE: Data starts Sep 2004 — ECB limitation.
    """
    def _fetch_single(maturity: str) -> pd.Series:
        series_key = f"YC/B.U2.EUR.4F.G_N_{rating}.SV_C_YM.SR_{maturity}"
        url = f"https://data-api.ecb.europa.eu/service/data/{series_key}?format=csvdata"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text), parse_dates=["TIME_PERIOD"])
        series = df.set_index("TIME_PERIOD")["OBS_VALUE"]
        series.name = f"spot_{maturity.lower()}"
        return series

    print("Fetching ECB spot rates...")
    maturities = ["1Y", "2Y", "3Y", "4Y", "5Y", "10Y", "30Y"]
    df_rates = pd.concat([_fetch_single(m) for m in maturities], axis=1).sort_index()
    df_rates = df_rates.resample("ME").last().loc["2004-01":"2025-12"]

    print(df_rates.head())
    print("Shape:", df_rates.shape)
    print("NaNs:\n", df_rates.isna().sum())

    df_rates.to_parquet(DATA_PROCESSED / "ecb_spot_rates.parquet")
    print("✓ Saved ecb_spot_rates.parquet")

import yfinance as yf
import numpy as np


# ── ETF Data ──────────────────────────────────────────────────

def fetch_etf_data():
    tickers = {
        "govt_short":    "IBGS.AS",   # iShares Euro Govt 1-3Y
        "govt_mid":      "IBGM.AS",   # iShares Euro Govt 7-10Y
        "govt_long":     "IBGL.AS",   # iShares Euro Govt 15-30Y
        "corp_ig":       "IEAC.AS",   # iShares EUR Corporate IG
        "equity":        "IMEU.AS",   # iShares MSCI Europe
    }

    print("Fetching ETF data...")
    raw = yf.download(
        list(tickers.values()),
        start="2000-01-01",
        end="2025-12-31",
        interval="1mo",
        auto_adjust=True
    )["Close"]

    raw.columns = list(tickers.keys())

    # Laske log-tuotot total return indexistä
    returns = np.log(raw / raw.shift(1))
    returns.columns = [c + "_logret" for c in raw.columns]

    raw.to_parquet(DATA_PROCESSED / "etf_prices.parquet")
    returns.to_parquet(DATA_PROCESSED / "etf_returns.parquet")
    print(f"✓ Saved ETF data. Coverage:")
    print(raw.apply(lambda c: f"{c.first_valid_index().date()} → {c.last_valid_index().date()}"))

fetch_etf_data()



### MSCI for Q5, MIGHT NOT WORK RIGHT

def fetch_and_save_msci():
    """
    Load MSCI Europe daily index from CSV, resample to month-end,
    compute monthly log returns, save to parquet.

    Source file:  data/raw/MSCI_Europe_daily.csv
    Columns:      Date (YYYY-MM-DD), MSCI Europe Index (float)
    Coverage:     2000-01-03 to 2025-12-xx (daily → resampled to month-end)

    Resampling:   .resample('ME').last() — takes last trading day of each month
    Log return:   r_t = ln(P_t / P_{t-1})  on monthly series
    Saved to:     data/processed/msci_europe.parquet
    """
    print("Loading MSCI Europe from CSV...")

    BASE = Path(__file__).parent.parent
    raw_path = BASE / "data" / "raw" / "MSCI_Europe_daily.csv"

    df = pd.read_csv(
        raw_path,
        usecols=[0, 1],                    # skip the empty trailing columns
        names=["date", "price"],           # rename on load
        header=0,                          # skip original header row
        parse_dates=["date"],
    )

    df = df.set_index("date").sort_index()

    # Resample daily → month-end (last trading day of each month)
    monthly = df.resample("ME").last()
    monthly = monthly.loc["2000-01":"2025-12"]

    # Compute log returns
    monthly["equity_logret"] = np.log(monthly["price"] / monthly["price"].shift(1))

    print(f"  Shape:    {monthly.shape}")
    print(f"  Coverage: {monthly.index.min().date()} → {monthly.index.max().date()}")
    print(f"  NaNs:     {monthly['equity_logret'].isna().sum()} (first row expected)")
    print(monthly.head(3))

    processed_path = BASE / "data" / "processed" / "msci_europe.parquet"
    monthly[["price", "equity_logret"]].to_parquet(processed_path)
    print("✓ Saved msci_europe.parquet")





### MAIN LOOP

if import_ecb_spot:
    fetch_and_save_ecb_spot()
else:
    print("⚠ Skipped ECB spot rates (import_ecb_spot = False)")

if import_etf_data:
    fetch_etf_data()
else:
    print("⚠ Skipped ETF data (import_etf_data = False)")

if import_msci:
    fetch_and_save_msci()