# src/Q4_credit_markets.py
# ─────────────────────────────────────────────
# Q4: Realized returns and risk of corporate bonds versus government bonds
# Q5 (partial): Credit excess return analysis and systematic risk setup
# INPUTS:  data/processed/etf_prices.parquet
#          data/processed/etf_returns.parquet
# OUTPUTS: Q4_cumulative_returns.png
#          Q4_credit_excess.png
#          Q4_summary_table.png
#          Q4_drawdown.png
#          Q4_rolling_sharpe.png
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from src.config import SUBPERIODS, COLORS
from src.data_loader import load_etf_prices, load_etf_returns


# ── 1. Core calculations ──────────────────────────────────────

def compute_credit_excess(returns: pd.DataFrame) -> pd.Series:
    """
    Credit excess return = corporate log return minus govt mid log return.
    Isolates credit risk premium by stripping out (approximate) duration component.
    Both series have similar duration (~7-8Y) so the difference is predominantly credit.
    """
    excess = returns["corp_ig_logret"] - returns["govt_mid_logret"]
    excess.name = "credit_excess"
    return excess.dropna()


def compute_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """
    Annualized return, vol, Sharpe, and max drawdown from monthly log returns.
    Ann. Return = mean(r) * 12
    Ann. Vol    = std(r) * sqrt(12)
    Sharpe      = Ann.Return / Ann.Vol  (no rf deduction — comparing within bonds)
    Max DD      = min((cum_t - rolling_max_t) / rolling_max_t)
    """
    r = returns.dropna()
    if len(r) < 6:
        return {"Ann. Return (%)": np.nan, "Ann. Vol (%)": np.nan,
                "Sharpe": np.nan, "Max Drawdown (%)": np.nan}

    ann_return = r.mean() * periods_per_year
    ann_vol    = r.std()  * np.sqrt(periods_per_year)
    sharpe     = ann_return / ann_vol if ann_vol > 0 else np.nan

    cum         = np.exp(r.cumsum())
    rolling_max = cum.cummax()
    max_dd      = ((cum - rolling_max) / rolling_max).min()

    return {
        "Ann. Return (%)":  round(ann_return * 100, 2),
        "Ann. Vol (%)":     round(ann_vol    * 100, 2),
        "Sharpe":           round(sharpe,           2),
        "Max Drawdown (%)": round(max_dd     * 100, 2),
    }


def compute_summary_table(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Side-by-side stats for corp_ig vs govt_mid over full sample and sub-periods.
    """
    periods = {"Full Sample": ("2004-09", "2025-12"), **SUBPERIODS}
    rows = []

    for period_label, (start, end) in periods.items():
        for asset_label, col in [("Corporate IG", "corp_ig_logret"),
                                  ("Govt Mid (7-10Y)", "govt_mid_logret")]:
            if col not in returns.columns:
                continue
            r = returns.loc[start:end, col].dropna()
            stats = compute_stats(r)
            rows.append({"Period": period_label, "Asset": asset_label, **stats})

    return pd.DataFrame(rows)


# ── 2. Plots ──────────────────────────────────────────────────

def plot_cumulative_returns(prices: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 1: Cumulative total return — corporate IG vs govt mid.
    Both indexed to 100 at their first common date.
    """
    corp = prices["corp_ig"].dropna()
    govt = prices["govt_mid"].dropna()

    # Align to common start date
    start = max(corp.index.min(), govt.index.min())
    corp  = corp.loc[start:]
    govt  = govt.loc[start:]

    corp_idx = corp / corp.iloc[0] * 100
    govt_idx = govt / govt.iloc[0] * 100

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(corp_idx.index, corp_idx, label="Corporate IG (IEAC)",
            color="steelblue", linewidth=2)
    ax.plot(govt_idx.index, govt_idx, label="Govt Mid 7-10Y (IBGM)",
            color="darkorange", linewidth=2)

    # Sub-period shading
    for (label, (s, e)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color=color)

    ax.set_title("Cumulative Total Returns: Corporate IG vs Government Bonds",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Return (100 = start)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
    fig.tight_layout()
    return fig


def plot_credit_excess(returns: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 2: Credit excess return — monthly bars + cumulative line.
    Monthly bars show episodic nature of credit premium.
    Cumulative line shows long-run trend — upward slope = persistent premium.
    """
    excess = compute_credit_excess(returns) * 100  # to percent

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Top panel — monthly excess return bars
    colors_bar = ["steelblue" if v >= 0 else "crimson" for v in excess]
    ax1.bar(excess.index, excess, color=colors_bar, alpha=0.7, width=20)
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axhline(excess.mean(), color="steelblue", linewidth=1.5,
                linestyle="--", label=f"Mean = {excess.mean():.2f}%/month")
    ax1.set_title("Monthly Credit Excess Return (Corp IG minus Govt Mid)",
                  fontsize=12, fontweight="bold")
    ax1.set_ylabel("Monthly Excess Return (%)")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, axis="y")

    # Bottom panel — cumulative excess return
    cumulative = excess.cumsum()
    ax2.plot(cumulative.index, cumulative, color="steelblue", linewidth=2)
    ax2.fill_between(cumulative.index, cumulative, 0,
                     where=cumulative >= 0, alpha=0.2, color="steelblue")
    ax2.fill_between(cumulative.index, cumulative, 0,
                     where=cumulative < 0, alpha=0.3, color="crimson")
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_title("Cumulative Credit Excess Return",
                  fontsize=12, fontweight="bold")
    ax2.set_ylabel("Cumulative Excess Return (%)")
    ax2.grid(alpha=0.3)

    # Sub-period shading on both panels
    for ax in [ax1, ax2]:
        for (label, (s, e)), color in zip(SUBPERIODS.items(), COLORS):
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.1, color=color)

    fig.tight_layout()
    return fig


def plot_drawdown(returns: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 3: Drawdown comparison — corp vs govt mid.
    Reveals asymmetric risk profile of credit: slow grind up, sudden crashes.
    Key episodes: GFC 2008, Sovereign Crisis 2011, COVID 2020, Hiking 2022.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    assets = [
        ("Corporate IG",     "corp_ig_logret",  "steelblue"),
        ("Govt Mid (7-10Y)", "govt_mid_logret", "darkorange"),
    ]

    for ax, (label, col, color) in zip(axes, assets):
        if col not in returns.columns:
            continue
        r   = returns[col].dropna()
        cum = np.exp(r.cumsum())
        dd  = (cum - cum.cummax()) / cum.cummax() * 100

        ax.fill_between(dd.index, dd, 0, color=color, alpha=0.4)
        ax.plot(dd.index, dd, color=color, linewidth=1)
        ax.set_title(f"Drawdown — {label}  (worst: {dd.min():.1f}%)",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Drawdown (%)")
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(alpha=0.3)

        # Annotate worst drawdown point
        worst_date = dd.idxmin()
        ax.annotate(f"{dd.min():.1f}%\n{worst_date.strftime('%Y-%m')}",
                    xy=(worst_date, dd.min()),
                    xytext=(20, 20), textcoords="offset points",
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color))

    fig.suptitle("Drawdown Comparison: Corporate IG vs Government Bonds",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rolling_sharpe(returns: pd.DataFrame, window: int = 36) -> plt.Figure:
    """
    Exhibit 4: Rolling 36-month Sharpe ratio — corp vs govt mid.
    Shows whether the risk-adjusted case for corporate bonds is stable or regime-dependent.
    Rolling Sharpe = (mean(r) * 12) / (std(r) * sqrt(12))  over trailing 36 months.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    assets = [
        ("Corporate IG",     "corp_ig_logret",  "steelblue"),
        ("Govt Mid (7-10Y)", "govt_mid_logret", "darkorange"),
    ]

    for label, col, color in assets:
        if col not in returns.columns:
            continue
        r = returns[col].dropna()
        rolling_ret = r.rolling(window).mean() * 12
        rolling_vol = r.rolling(window).std()  * np.sqrt(12)
        rolling_sharpe = rolling_ret / rolling_vol

        ax.plot(rolling_sharpe.index, rolling_sharpe,
                label=label, color=color, linewidth=1.5)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(1, color="grey",  linewidth=0.5, linestyle=":")

    # Sub-period shading
    for (label, (s, e)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.15, color=color)

    ax.set_title(f"Rolling {window}-Month Sharpe Ratio: Corporate IG vs Government Bonds",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio (annualized)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_summary_table(df_table: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 5: Summary statistics table — corp vs govt, full sample + sub-periods.
    Sharpe cells colour-coded: green > 0.5, orange 0-0.5, red < 0.
    """
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.axis("off")

    display_cols = ["Period", "Asset", "Ann. Return (%)",
                    "Ann. Vol (%)", "Sharpe", "Max Drawdown (%)"]

    table = ax.table(
        cellText=df_table[display_cols].values,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.7)

    n_cols = len(display_cols)
    # Header styling
    for j in range(n_cols):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row styling — alternate shading, colour Sharpe column
    sharpe_col = display_cols.index("Sharpe")
    ret_col    = display_cols.index("Ann. Return (%)")
    dd_col     = display_cols.index("Max Drawdown (%)")

    for i in range(1, len(df_table) + 1):
        # Alternate row background
        row_color = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
        for j in range(n_cols):
            table[(i, j)].set_facecolor(row_color)

        # Colour Sharpe
        try:
            s = float(table[(i, sharpe_col)].get_text().get_text())
            table[(i, sharpe_col)].set_facecolor(
                "#D5F5E3" if s > 0.5 else "#FDEBD0" if s > 0 else "#FADBD8")
        except ValueError:
            pass

        # Colour return — positive green, negative red
        try:
            r = float(table[(i, ret_col)].get_text().get_text())
            table[(i, ret_col)].set_facecolor(
                "#D5F5E3" if r > 0 else "#FADBD8")
        except ValueError:
            pass

    fig.suptitle("Q4: Corporate IG vs Government Bonds — Summary Statistics",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


# ── 3. Console summary ────────────────────────────────────────

def print_console_summary(returns: pd.DataFrame):
    """Print key numbers to console for quick inspection."""
    excess = compute_credit_excess(returns) * 100
    print("\n── Q4 Credit Excess Return Summary ──")
    print(f"  Full sample mean:     {excess.mean():.3f}% / month  "
          f"({excess.mean()*12:.2f}% annualized)")
    print(f"  Full sample vol:      {excess.std():.3f}% / month")
    print(f"  % months positive:    {(excess > 0).mean()*100:.1f}%")
    print(f"  Best month:           {excess.max():.2f}%  ({excess.idxmax().strftime('%Y-%m')})")
    print(f"  Worst month:          {excess.min():.2f}%  ({excess.idxmin().strftime('%Y-%m')})")

    print("\n── Sub-period Credit Premium ──")
    for label, (start, end) in SUBPERIODS.items():
        sub = excess.loc[start:end]
        if len(sub) < 3:
            continue
        print(f"  {label:<30} mean = {sub.mean():+.3f}%/mo  "
              f"({sub.mean()*12:+.2f}% ann)")


# ── 4. run_all ────────────────────────────────────────────────

def run_all() -> dict:
    """Run full Q4 analysis. Returns dict of {filename: Figure}."""
    prices  = load_etf_prices()
    returns = load_etf_returns()

    print_console_summary(returns)

    df_table = compute_summary_table(returns)

    return {
        "Q4_cumulative_returns.png": plot_cumulative_returns(prices),
        "Q4_credit_excess.png":      plot_credit_excess(returns),
        "Q4_drawdown.png":           plot_drawdown(returns),
        "Q4_rolling_sharpe.png":     plot_rolling_sharpe(returns),
        "Q4_summary_table.png":      plot_summary_table(df_table),
    }