import refinitiv.data as rd
import pandas as pd


# EI KÄYTÖSSÄ VIELÄ / EI PÄÄSYÄ REFINITIV MULLA


# 1. Open a session using your App Key
# Make sure your Refinitiv Workspace/Eikon app is OPEN and LOGGED IN
#rd.open_session(app_key='99ebe5da8d12486a84333454b6a51a0d704cec10')

# 2. Pull some basic snapshot data (Fundamental/Price data)
#df = rd.get_data(
#    universe=['AAPL.O', 'MSFT.O', 'GOOGL.O'],
#    fields=['TR.PriceClose', 'TR.Volume', 'TR.CompanyMarketCap']
#)

#print(df)

# 3. Pull historical time-series data
#history = rd.get_history(
#   universe='AAPL.O',
#    fields=['TR.PriceClose'],
#    interval='daily',
#    start='2023-01-01',
#    end='2023-12-31'
#)

#print(history.head())

# 4. Close the session when done
#rd.close_session()

# main.py
# ─────────────────────────────────────────────
# Master script. Runs all analysis modules and
# saves outputs. Run this to reproduce all charts.
#
# Usage: python main.py
# ─────────────────────────────────────────────

# data/data_import.py
# PURPOSE: Fetch all raw data from ECB API and save to data/processed/
# Run this ONCE before running main.py or any Q files.

import requests
import pandas as pd
from io import StringIO
from pathlib import Path

DATA_PROCESSED = Path("C:/Users/tuoma/PycharmProjectsfixedincome/data/processed")
DATA_PROCESSED.mkdir(exist_ok=True)  # creates the folder if it doesn't exist

# ── ECB Spot Rates ────────────────────────────────────────────

def fetch_ecb_spot(maturity: str, rating: str = "A") -> pd.Series:
    """
    Fetch ECB spot rate using new ECB Data Portal API.
    rating: A=AAA, C=all ratings
    maturity: '1Y', '2Y', '3Y', '4Y', '5Y', '10Y', '30Y'
    """
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
series = [fetch_ecb_spot(m) for m in maturities]

df_rates = pd.concat(series, axis=1).sort_index()
df_rates = df_rates.resample("ME").last()
df_rates = df_rates.loc["2004-01":"2025-12"]

print(df_rates.head())
print("Shape:", df_rates.shape)
print("NaNs:\n", df_rates.isna().sum())

df_rates.to_parquet(DATA_PROCESSED / "ecb_spot_rates.parquet")
print("✓ Saved ecb_spot_rates.parquet")