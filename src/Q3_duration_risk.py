# src/Q3_duration_risk.py
# ─────────────────────────────────────────────
# Q3: Have investors historically been compensated for taking duration risk?
# INPUT:  data/processed/etf_prices.parquet
#         data/processed/etf_returns.parquet
# OUTPUT: Q3_cumulative_returns.png
#         Q3_summary_table.png
#         Q3_drawdown.png
#         Q3_annual_returns.png
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from src.config import SUBPERIODS, COLORS
from src.data_loader import load_etf_prices, load_etf_returns


# ── 1. Summary statistics ─────────────────────────────────────

def compute_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """
    Compute annualized return, vol, Sharpe, and max drawdown for a return series.
    Input: monthly log returns.
    """
    ann_return = returns.mean() * periods_per_year
    ann_vol    = returns.std() * np.sqrt(periods_per_year)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

    # Max drawdown from cumulative log return index
    cum = np.exp(returns.cumsum())
    rolling_max = cum.cummax()
    drawdown = (cum - rolling_max) / rolling_max
    max_dd = drawdown.min()

    return {
        "Ann. Return (%)": round(ann_return * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe":          round(sharpe, 2),
        "Max Drawdown (%)":round(max_dd * 100, 2),
    }


def compute_summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute stats for short/mid/long govt buckets over full sample
    and each sub-period.
    """
    buckets = {
        "Short (1-3Y)":  "govt_short_logret",
        "Mid (7-10Y)":   "govt_mid_logret",
        "Long (15-30Y)": "govt_long_logret",
    }

    rows = []
    periods = {"Full Sample": ("2004-09", "2025-12"), **SUBPERIODS}

    for period_label, (start, end) in periods.items():
        for bucket_label, col in buckets.items():
            if col not in returns.columns:
                continue
            r = returns.loc[start:end, col].dropna()
            if len(r) < 6:
                continue
            stats = compute_stats(r)
            rows.append({
                "Period":  period_label,
                "Bucket":  bucket_label,
                **stats
            })

    return pd.DataFrame(rows)


# ── 2. Plots ──────────────────────────────────────────────────

def plot_cumulative_returns(prices: pd.DataFrame) -> plt.Figure:
    """Exhibit 1: Cumulative total return for short, mid, long govt bond ETFs."""
    buckets = {
        "Short (1-3Y)":  ("govt_short",  "steelblue"),
        "Mid (7-10Y)":   ("govt_mid",    "darkorange"),
        "Long (15-30Y)": ("govt_long",   "green"),
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for label, (col, color) in buckets.items():
        if col not in prices.columns:
            continue
        series = prices[col].dropna()
        # Normalize to 100 at start
        normalized = series / series.iloc[0] * 100
        ax.plot(normalized.index, normalized, label=label,
                color=color, linewidth=2)

    # Shade sub-periods
    for (label, (start, end)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.15, color=color)

    ax.set_title("Cumulative Total Returns by Duration Bucket (Indexed to 100)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Return (100 = start)")
    ax.set_xlabel("")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.DataFrame) -> plt.Figure:
    """Exhibit 2: Drawdown chart for each duration bucket."""
    buckets = {
        "Short (1-3Y)":  ("govt_short_logret", "steelblue"),
        "Mid (7-10Y)":   ("govt_mid_logret",   "darkorange"),
        "Long (15-30Y)": ("govt_long_logret",  "green"),
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for ax, (label, (col, color)) in zip(axes, buckets.items()):
        if col not in returns.columns:
            continue
        r = returns[col].dropna()
        cum = np.exp(r.cumsum())
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max * 100

        ax.fill_between(drawdown.index, drawdown, 0,
                        color=color, alpha=0.4)
        ax.plot(drawdown.index, drawdown, color=color, linewidth=1)
        ax.set_title(f"Drawdown — {label}")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Drawdowns by Duration Bucket", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_annual_returns(returns: pd.DataFrame) -> plt.Figure:
    """Exhibit 3: Annual returns per bucket as grouped bar chart."""
    buckets = {
        "Short (1-3Y)":  "govt_short_logret",
        "Mid (7-10Y)":   "govt_mid_logret",
        "Long (15-30Y)": "govt_long_logret",
    }

    # Resample to annual
    annual = pd.DataFrame()
    for label, col in buckets.items():
        if col not in returns.columns:
            continue
        annual[label] = returns[col].resample("YE").sum() * 100  # log return sum = annual

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(annual))
    width = 0.25
    colors = ["steelblue", "darkorange", "green"]

    for i, (label, color) in enumerate(zip(annual.columns, colors)):
        bars = ax.bar(x + i * width, annual[label], width,
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
    """Exhibit 4: Summary statistics table as figure."""
    # Pivot so periods are rows, buckets are column groups
    full = df_table[df_table["Period"] == "Full Sample"].copy()
    sub  = df_table[df_table["Period"] != "Full Sample"].copy()

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, data, title in zip(
        axes,
        [full, sub],
        ["Full Sample (2004–2025)", "By Sub-period"]
    ):
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

        # Header styling
        for j in range(len(display_cols)):
            table[(0, j)].set_facecolor("#2c3e50")
            table[(0, j)].set_text_props(color="white", fontweight="bold")

        # Color Sharpe column by value
        sharpe_col = display_cols.index("Sharpe")
        for i in range(1, len(data) + 1):
            val = table[(i, sharpe_col)].get_text().get_text()
            try:
                s = float(val)
                color = "#d5f5e3" if s > 0.5 else "#fdebd0" if s > 0 else "#fadbd8"
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
    """Run full Q3 analysis. Returns dict of {filename: Figure}."""
    prices  = load_etf_prices()
    returns = load_etf_returns()

    # Quick console summary
    df_table = compute_summary_table(returns)
    print("\n── Q3 Duration Risk Summary (Full Sample) ──")
    full = df_table[df_table["Period"] == "Full Sample"]
    print(full[["Bucket", "Ann. Return (%)", "Ann. Vol (%)",
                "Sharpe", "Max Drawdown (%)"]].to_string(index=False))

    return {
        "Q3_cumulative_returns.png": plot_cumulative_returns(prices),
        "Q3_drawdown.png":           plot_drawdown(returns),
        "Q3_annual_returns.png":     plot_annual_returns(returns),
        "Q3_summary_table.png":      plot_summary_table(df_table),
    }