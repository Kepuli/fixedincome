# src/Q3_duration_risk.py
# ─────────────────────────────────────────────
# Q3: Have investors historically been compensated for taking duration risk?
#
# UPDATE (March 2026):
#   Now uses Refinitiv iBoxx EUR Sovereign sub-indices (2000–2025)
#   when available. Falls back to ETF data (2008–2025) if not.
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.config import SUBPERIODS, COLORS
from src.data_loader import (
    load_etf_prices,
    load_etf_returns,
    load_govt_duration_returns_refinitiv,
)


# ── 0. Data loading — Refinitiv preferred, ETF fallback ──────

def load_q3_data() -> tuple[pd.DataFrame, pd.DataFrame | None, str]:
    """
    Load duration bucket data.

    Returns
    -------
    returns : DataFrame with govt_short_logret, govt_mid_logret, govt_long_logret
    prices  : DataFrame with govt_short, govt_mid, govt_long (ETF only, None for Refinitiv)
    source  : 'refinitiv' or 'etf'
    """
    ref = load_govt_duration_returns_refinitiv()

    if ref is not None and len(ref) > 0:
        source = "refinitiv"
        print(f"\n  Q3 data source: Refinitiv iBoxx sub-indices")
        print(f"  Sample: {ref.index.min().date()} → {ref.index.max().date()}  "
              f"({len(ref)} months)")
        return ref, None, source
    else:
        returns = load_etf_returns()
        prices  = load_etf_prices()
        source = "etf"
        print(f"\n  Q3 data source: ETF proxies")
        print(f"  Sample: {returns.index.min().date()} → {returns.index.max().date()}")
        return returns, prices, source


def _full_sample_dates(returns: pd.DataFrame) -> tuple[str, str]:
    start = returns.index.min().strftime("%Y-%m")
    end   = returns.index.max().strftime("%Y-%m")
    return start, end


def _bucket_labels(source: str) -> dict:
    if source == "refinitiv":
        return {
            "Short (1-3Y)":  "govt_short_logret",
            "Mid (7-10Y)":   "govt_mid_logret",
            "Long (10+Y)":   "govt_long_logret",
        }
    return {
        "Short (1-3Y)":  "govt_short_logret",
        "Mid (7-10Y)":   "govt_mid_logret",
        "Long (15-30Y)": "govt_long_logret",
    }


# ── 1. Summary statistics ─────────────────────────────────────

def compute_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    ann_return = returns.mean() * periods_per_year
    ann_vol    = returns.std() * np.sqrt(periods_per_year)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum = np.exp(returns.cumsum())
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "Ann. Return (%)": round(ann_return * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe":          round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
    }


def compute_summary_table(returns: pd.DataFrame, source: str) -> pd.DataFrame:
    buckets = _bucket_labels(source)
    full_start, full_end = _full_sample_dates(returns)

    rows = []
    periods = {f"Full Sample ({full_start[:4]}–{full_end[:4]})": (full_start, full_end)}
    periods.update(SUBPERIODS)

    for period_label, (start, end) in periods.items():
        for bucket_label, col in buckets.items():
            if col not in returns.columns:
                continue
            r = returns.loc[start:end, col].dropna()
            if len(r) < 6:
                continue
            stats = compute_stats(r)
            rows.append({"Period": period_label, "Bucket": bucket_label, **stats})

    return pd.DataFrame(rows)


# ── 2. Plots ──────────────────────────────────────────────────

def plot_cumulative_returns(returns: pd.DataFrame, prices: pd.DataFrame | None,
                            source: str) -> plt.Figure:
    """Cumulative total return for short, mid, long govt bond buckets."""
    buckets = _bucket_labels(source)
    bucket_colors = ["steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(14, 6))

    if source == "refinitiv":
        # Build cumulative from log returns
        for (label, col), color in zip(buckets.items(), bucket_colors):
            if col not in returns.columns:
                continue
            r = returns[col].dropna()
            cum = np.exp(r.cumsum())
            cum = cum / cum.iloc[0] * 100
            ax.plot(cum.index, cum, label=label, color=color, linewidth=2)
    else:
        # Use ETF prices
        price_cols = {
            "Short (1-3Y)":  "govt_short",
            "Mid (7-10Y)":   "govt_mid",
            "Long (15-30Y)": "govt_long",
        }
        for (label, col), color in zip(price_cols.items(), bucket_colors):
            if col not in prices.columns:
                continue
            series = prices[col].dropna()
            normalized = series / series.iloc[0] * 100
            ax.plot(normalized.index, normalized, label=label, color=color, linewidth=2)

    for (label, (start, end)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, color=color)

    full_start, full_end = _full_sample_dates(returns)
    ax.set_title(f"Cumulative Total Returns by Duration Bucket "
                 f"({full_start[:4]}–{full_end[:4]})",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Return (100 = start)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.DataFrame, source: str) -> plt.Figure:
    buckets = _bucket_labels(source)
    bucket_colors = ["steelblue", "darkorange", "green"]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (label, col), color in zip(axes, buckets.items(), bucket_colors):
        if col not in returns.columns:
            continue
        r = returns[col].dropna()
        cum = np.exp(r.cumsum())
        drawdown = (cum - cum.cummax()) / cum.cummax() * 100

        ax.fill_between(drawdown.index, drawdown, 0, color=color, alpha=0.4)
        ax.plot(drawdown.index, drawdown, color=color, linewidth=1)
        ax.set_title(f"Drawdown — {label}")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Drawdowns by Duration Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_annual_returns(returns: pd.DataFrame, source: str) -> plt.Figure:
    buckets = _bucket_labels(source)

    annual = pd.DataFrame()
    for label, col in buckets.items():
        if col not in returns.columns:
            continue
        annual[label] = returns[col].resample("YE").sum() * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(annual))
    width = 0.25
    colors = ["steelblue", "darkorange", "green"]

    for i, (label, color) in enumerate(zip(annual.columns, colors)):
        ax.bar(x + i * width, annual[label], width,
               label=label, color=color, alpha=0.8)

    ax.set_title("Annual Returns by Duration Bucket (%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Annual Return (%)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(d.year) for d in annual.index], rotation=45)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_summary_table(df_table: pd.DataFrame) -> plt.Figure:
    full_label = [p for p in df_table["Period"].unique() if "Full Sample" in p][0]
    full = df_table[df_table["Period"] == full_label].copy()
    sub  = df_table[df_table["Period"] != full_label].copy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, data, title in zip(axes, [full, sub], [full_label, "By Sub-period"]):
        ax.axis("off")
        display_cols = ["Period", "Bucket", "Ann. Return (%)",
                        "Ann. Vol (%)", "Sharpe", "Max Drawdown (%)"]
        table = ax.table(
            cellText=data[display_cols].values,
            colLabels=display_cols,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.1, 1.6)

        for j in range(len(display_cols)):
            table[(0, j)].set_facecolor("#2c3e50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        sharpe_col = display_cols.index("Sharpe")
        for i in range(1, len(data) + 1):
            val = table[(i, sharpe_col)].get_text().get_text()
            try:
                s = float(val)
                color = "#d5f5e3" if s > 1 else "#fdebd0" if s > 0 else "#fadbd8"
                table[(i, sharpe_col)].set_facecolor(color)
            except ValueError:
                pass

        ax.set_title(title, fontweight="bold", pad=10)

    fig.suptitle("Q3: Duration Risk — Summary Statistics",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 3. run_all ────────────────────────────────────────────────

def run_all() -> dict:
    returns, prices, source = load_q3_data()

    df_table = compute_summary_table(returns, source)
    full_label = [p for p in df_table["Period"].unique() if "Full Sample" in p][0]
    print(f"\n── Q3 Duration Risk Summary ({full_label}) ──")
    full = df_table[df_table["Period"] == full_label]
    print(full[["Bucket", "Ann. Return (%)", "Ann. Vol (%)",
                "Sharpe", "Max Drawdown (%)"]].to_string(index=False))

    return {
        "Q3_cumulative_returns.png": plot_cumulative_returns(returns, prices, source),
        "Q3_drawdown.png":           plot_drawdown(returns, source),
        "Q3_annual_returns.png":     plot_annual_returns(returns, source),
        "Q3_summary_table.png":      plot_summary_table(df_table),
    }