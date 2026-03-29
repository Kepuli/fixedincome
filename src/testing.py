import pandas as pd
import numpy as np

from pathlib import Path
BASE = Path(__file__).parent.parent

etf = pd.read_parquet(BASE / "data/processed/etf_returns.parquet")
ref = pd.read_parquet(BASE / "data/processed/refinitiv_returns.parquet")


# Align to same index format
etf.index = etf.index.to_period("M").to_timestamp("M")
ref.index = ref.index.to_period("M").to_timestamp("M")

# Compare overlapping period (2008-2025) for govt_short
overlap = pd.DataFrame({
    "etf":  etf["govt_short_logret"],
    "iboxx": ref["govt_short_logret"],
}).dropna()

print(f"Overlapping months: {len(overlap)}")
print(f"\n── Govt short ──")
print(f"\nCorrelation: {overlap['etf'].corr(overlap['iboxx']):.4f}")
print(f"\nETF  ann return: {overlap['etf'].mean()*12*100:.2f}%")
print(f"iBoxx ann return: {overlap['iboxx'].mean()*12*100:.2f}%")
print(f"Difference:       {(overlap['iboxx'].mean()-overlap['etf'].mean())*12*100:.2f}% p.a.")
print(f"\nETF  ann vol: {overlap['etf'].std()*np.sqrt(12)*100:.2f}%")
print(f"iBoxx ann vol: {overlap['iboxx'].std()*np.sqrt(12)*100:.2f}%")

# Same for govt_mid
overlap_mid = pd.DataFrame({
    "etf":  etf["govt_mid_logret"],
    "iboxx": ref["govt_mid_logret"],
}).dropna()

print(f"\n── Govt Mid ──")
print(f"Correlation: {overlap_mid['etf'].corr(overlap_mid['iboxx']):.4f}")
print(f"ETF  ann return: {overlap_mid['etf'].mean()*12*100:.2f}%")
print(f"iBoxx ann return: {overlap_mid['iboxx'].mean()*12*100:.2f}%")
print(f"Difference:       {(overlap_mid['iboxx'].mean()-overlap_mid['etf'].mean())*12*100:.2f}% p.a.")

# Same for govt_long
overlap_long = pd.DataFrame({
    "etf":  etf["govt_long_logret"],
    "iboxx": ref["govt_long_logret"],
}).dropna()

print(f"\n── Govt Long ──")
print(f"Correlation: {overlap_long['etf'].corr(overlap_long['iboxx']):.4f}")
print(f"ETF  ann return: {overlap_long['etf'].mean()*12*100:.2f}%")
print(f"iBoxx ann return: {overlap_long['iboxx'].mean()*12*100:.2f}%")
print(f"Difference:       {(overlap_long['iboxx'].mean()-overlap_long['etf'].mean())*12*100:.2f}% p.a.")