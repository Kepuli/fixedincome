from idlelib.autocomplete import TRY_A

import refinitiv.data as rd
import requests
import pandas as pd
from io import StringIO
from pathlib import Path

# dynamic path
BASE = Path(__file__).parent.parent  # goes up from data/ to project root
DATA_RAW       = BASE / "data" / "raw"
DATA_PROCESSED = BASE / "data" / "processed"
DATA_PROCESSED.mkdir(exist_ok=True)
DATA_RAW.mkdir(exist_ok=True)


##########################
### SELECT WHAT TO IMPORT!
##########################

# ── SELECT WHAT TO IMPORT ─────────────────────────────────────
import_ecb_spot     = False
import_etf_data     = False
import_msci         = True
import_refinitiv    = False   # ← new flag
import_msci_refinitiv = False

##########################

def fetch_and_save_msci_refinitiv():
    """Fetch MSCI Europe Net Return Index (EUR) from Refinitiv."""
    import refinitiv.data as rd
    rd.open_session()

    print("Fetching MSCI Europe Net Return (EUR) from Refinitiv...")
    df = rd.get_history(
        universe=".dMIEU00000NEU",
        fields=["TRDPRC_1"],
        interval="daily",
        start="2000-01-01",
        end="2025-12-31",
    )

    print(f"  Raw rows: {len(df)}, columns: {list(df.columns)}")

    # Resample daily → month-end, compute log returns
    monthly = df.resample("ME").last()
    monthly["equity_logret"] = np.log(monthly.iloc[:, 0] / monthly.iloc[:, 0].shift(1))
    monthly = monthly[["equity_logret"]].dropna()

    monthly.to_parquet(DATA_PROCESSED / "msci_europe.parquet")
    print(f"✓ MSCI Europe saved: {monthly.index.min().date()} → {monthly.index.max().date()}"
          f"  ({len(monthly)} observations)")

    rd.close_session()


# ── Refinitiv fetch function ──────────────────────────────────
def fetch_and_save_refinitiv():
    rd.open_session()
    print("✓ Refinitiv session opened")

    def fetch_series(tickers: dict) -> pd.DataFrame:
        frames = {}
        for name, ric in tickers.items():
            print(f"  Fetching {name} ({ric})...")
            try:
                df = rd.get_history(
                    universe=ric,
                    fields=["TR.DAILYTOTALRETURN"],
                    interval="monthly",
                    start="2000-01-01",
                    end="2025-12-31",
                )
                if df is None or df.empty:
                    print(f"  ✗ {name}: empty response")
                    continue

                # Print actual columns so we know what Refinitiv returned
                print(f"  Columns returned: {list(df.columns)}")
                print(df.head(3))

                # Take the first non-index column regardless of name
                series = df.iloc[:, 0].rename(name)
                frames[name] = series
                print(f"  ✓ {name}: {series.index.min().date()} → {series.index.max().date()}")
            except Exception as e:
                print(f"  ✗ {name}: {e}")
        return pd.DataFrame(frames)

    # ── Fetch ─────────────────────────────────────────────────
    govt_prices = fetch_series({
        "govt_all": ".IBBEU0144",  # iBoxx EUR Sovereigns (all maturities)
        "govt_short": ".IBBEU0147",  # iBoxx EUR Sovereigns 1-3Y
        "govt_mid": ".IBBEU0154",  # iBoxx EUR Sovereigns 7-10Y
        "govt_long": ".IBBEU0145",  # iBoxx EUR Sovereigns 10+
    })

    corp_prices = fetch_series({
        "corp_ig":  ".IBBEU003D",   # iBoxx EUR Corporates IG
    })

    if govt_prices.empty or corp_prices.empty:
        print("✗ No data retrieved — check RICs and session")
        rd.close_session()
        return

    # ── Compute log returns ───────────────────────────────────
    all_prices  = pd.concat([govt_prices, corp_prices], axis=1)
    all_returns = np.log(all_prices / all_prices.shift(1))
    all_returns.columns = [c + "_logret" for c in all_prices.columns]

    print(f"\nCoverage:")
    print(all_prices.apply(lambda c:
          f"{c.first_valid_index().date()} → {c.last_valid_index().date()}"))

    # ── Save ──────────────────────────────────────────────────
    all_prices.to_parquet(DATA_PROCESSED / "refinitiv_prices.parquet")
    all_returns.to_parquet(DATA_PROCESSED / "refinitiv_returns.parquet")
    print("✓ Saved refinitiv_prices.parquet and refinitiv_returns.parquet")

    rd.close_session()
    print("✓ Session closed")

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

    # Nimeä sarakkeet tickerin perusteella — EI järjestyksen perusteella
    # yfinance saattaa palauttaa sarakkeet eri järjestyksessä kuin latasit
    reverse_map = {v: k for k, v in tickers.items()}  # {"IBGS.AS": "govt_short", ...}
    raw = raw.rename(columns=reverse_map)

    # Laske log-tuotot total return indexistä
    returns = np.log(raw / raw.shift(1))
    returns.columns = [c + "_logret" for c in raw.columns]

    # Tarkistus — tulosta sarakkeiden järjestys jotta näkee onko oikein
    print(f"  Columns in raw: {list(raw.columns)}")

    raw.to_parquet(DATA_PROCESSED / "etf_prices.parquet")
    returns.to_parquet(DATA_PROCESSED / "etf_returns.parquet")
    print(f"✓ Saved ETF data. Coverage:")
    print(raw.apply(lambda c: f"{c.first_valid_index().date()} → {c.last_valid_index().date()}"))

#fetch_etf_data()



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
else:
    print("⚠ Skipped msci data (import_msci_data = False)")

if import_refinitiv:
    fetch_and_save_refinitiv()
else:
    print("⚠ Skipped msci data (import_refinitiv_data = False)")
if import_msci_refinitiv:
    fetch_and_save_msci_refinitiv()
else:
    print("⚠ Skipped MSCI Europe Refinitiv (import_msci_refinitiv = False)")