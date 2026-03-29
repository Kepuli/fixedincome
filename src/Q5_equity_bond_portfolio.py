from scipy.optimize import minimize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from src.config import SUBPERIODS, COLORS
from src.data_loader import (
    load_etf_returns,
    load_msci_europe,
    load_govt_returns_refinitiv,
    load_corp_returns_refinitiv,
)
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════
# 1. DATA PREPARATION
# ══════════════════════════════════════════════════════════════

def load_all_returns() -> pd.DataFrame:
    """
    Load and align all return series for PORTFOLIO CONSTRUCTION.
    Uses ETF data for maturity buckets (short/mid/long) since
    Refinitiv maturity sub-indices are not yet available.

    Columns after alignment:
      equity     — MSCI Europe log return
      govt_short — iShares Euro Govt 1-3Y log return
      govt_mid   — iShares Euro Govt 7-10Y log return
      govt_long  — iShares Euro Govt 15-30Y log return
      corp_ig    — iShares EUR Corp Bond IG log return

    Common sample starts at latest first-valid date (~2009-05).
    """
    etf  = load_etf_returns()
    msci = load_msci_europe()

    # Normalize both to month-end before joining
    etf.index  = etf.index.to_period("M").to_timestamp("M")
    msci.index = msci.index.to_period("M").to_timestamp("M")

    df = pd.DataFrame({
        "equity":     msci,
        "govt_short": etf["govt_short_logret"],
        "govt_mid":   etf["govt_mid_logret"],
        "govt_long":  etf["govt_long_logret"],
        "corp_ig":    etf["corp_ig_logret"],
    }).dropna()

    print(f"\n── Q5 Portfolio Data Coverage (ETF) ──")
    print(f"  Common sample: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Observations:  {len(df)} monthly")

    return df


def load_regression_returns() -> tuple[pd.DataFrame, str]:
    """
    Load return data for the SYSTEMATIC RISK REGRESSION.
    Prefers Refinitiv iBoxx (2000–2025) for longer sample.
    Falls back to ETF data if unavailable.

    Returns
    -------
    df     : DataFrame with columns ['equity', 'govt', 'corp_ig']
    source : 'refinitiv' or 'etf'
    """
    govt_ref = load_govt_returns_refinitiv()
    corp_ref = load_corp_returns_refinitiv()
    msci     = load_msci_europe()

    if govt_ref is not None and corp_ref is not None and len(govt_ref) > 0 and len(corp_ref) > 0:
        # ── Refinitiv path: align all three series ──
        # Normalize to month-end
        msci.index     = msci.index.to_period("M").to_timestamp("M")
        govt_ref.index = govt_ref.index.to_period("M").to_timestamp("M")
        corp_ref.index = corp_ref.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "equity":  msci,
            "govt":    govt_ref,
            "corp_ig": corp_ref,
        }).dropna()

        source = "refinitiv"
        print(f"\n── Q5 Regression Data Coverage (Refinitiv iBoxx) ──")
        print(f"  Sample: {df.index.min().date()} → {df.index.max().date()}")
        print(f"  Observations: {len(df)} monthly")
    else:
        # ── ETF fallback: use govt_mid as government factor ──
        etf  = load_etf_returns()
        etf.index  = etf.index.to_period("M").to_timestamp("M")
        msci.index = msci.index.to_period("M").to_timestamp("M")

        df = pd.DataFrame({
            "equity":  msci,
            "govt":    etf["govt_mid_logret"],
            "corp_ig": etf["corp_ig_logret"],
        }).dropna()

        source = "etf"
        print(f"\n── Q5 Regression Data Coverage (ETF fallback) ──")
        print(f"  Sample: {df.index.min().date()} → {df.index.max().date()}")
        print(f"  Observations: {len(df)} monthly")

    return df, source


# ══════════════════════════════════════════════════════════════
# 2. PORTFOLIO CONSTRUCTION
# ══════════════════════════════════════════════════════════════

# Fixed portfolio weights
# Using log return weighting — valid approximation for monthly returns
# (error < 0.01% per month, negligible for our purposes)

BOND_PORTFOLIOS = {
    "Pure Govt\n(100% govt_mid)": {
        "govt_mid": 1.0
    },
    "Blended Bond\n(50/50 govt/corp)": {
        "govt_mid": 0.50,
        "corp_ig":  0.50,
    },
    "Diversified Bond\n(33% short/mid/corp)": {
        "govt_short": 0.33,
        "govt_mid":   0.34,
        "corp_ig":    0.33,
    },
}

MIXED_PORTFOLIOS = {
    "60/40\n(equity/govt)": {
        "equity":   0.60,
        "govt_mid": 0.40,
    },
    "60/30/10\n(equity/govt/corp)": {
        "equity":   0.60,
        "govt_mid": 0.30,
        "corp_ig":  0.10,
    },
    "60/20/20\n(equity/govt/corp)": {
        "equity":   0.60,
        "govt_mid": 0.20,
        "corp_ig":  0.20,
    },
}


def build_portfolio_returns(returns: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Compute portfolio log return at each t as weighted sum of asset log returns.
    r_portfolio_t = sum(w_i * r_i_t)
    Assumes monthly rebalancing back to target weights.
    """
    port_ret = sum(w * returns[asset] for asset, w in weights.items()
                   if asset in returns.columns)
    return port_ret


def build_all_portfolios(returns: pd.DataFrame,
                          definitions: dict) -> pd.DataFrame:
    """Build all portfolios from a weight definition dict."""
    result = {}
    for name, weights in definitions.items():
        result[name] = build_portfolio_returns(returns, weights)
    return pd.DataFrame(result)


# ══════════════════════════════════════════════════════════════
# 3. STATISTICS
# ══════════════════════════════════════════════════════════════

def compute_stats(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """
    Full statistics suite for a monthly log return series.
    """
    r = returns.dropna()
    if len(r) < 6:
        return {k: np.nan for k in ["Ann. Return (%)", "Ann. Vol (%)",
                                     "Sharpe", "Max DD (%)", "Calmar", "Skewness"]}

    ann_ret = r.mean() * periods_per_year
    ann_vol = r.std()  * np.sqrt(periods_per_year)
    sharpe  = ann_ret / ann_vol if ann_vol > 0 else np.nan
    skew    = r.skew()

    cum     = np.exp(r.cumsum())
    max_dd  = ((cum - cum.cummax()) / cum.cummax()).min()
    calmar  = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    return {
        "Ann. Return (%)": round(ann_ret * 100, 2),
        "Ann. Vol (%)":    round(ann_vol * 100, 2),
        "Sharpe":          round(sharpe,         2),
        "Max DD (%)":      round(max_dd  * 100,  2),
        "Calmar":          round(calmar,          2),
        "Skewness":        round(skew,            2),
    }


def compute_all_stats(port_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute stats for all portfolios in a DataFrame."""
    rows = []
    for name, series in port_returns.items():
        label = name.replace("\n", " ")
        rows.append({"Portfolio": label, **compute_stats(series)})
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 4. SYSTEMATIC RISK REGRESSION
# ══════════════════════════════════════════════════════════════

def run_systematic_risk_regression(reg_returns: pd.DataFrame,
                                    source: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fama-style factor regression to decompose corporate bond returns:

    corp_ig_t = alpha + beta_equity * equity_t + beta_govt * govt_t + epsilon_t

    When source='refinitiv', govt factor is the all-maturity iBoxx EUR Sovereigns
    index (broader factor, 2000-2025). When source='etf', govt factor is the
    7-10Y maturity bucket (2009-2025).

    Uses OLS with HC3 robust standard errors.
    """
    y = reg_returns["corp_ig"]
    X = sm.add_constant(pd.DataFrame({
        "equity": reg_returns["equity"],
        "govt":   reg_returns["govt"],
    }))

    model = sm.OLS(y, X).fit(cov_type="HC3")

    govt_factor_desc = ("iBoxx EUR Sovereigns (all mat.)" if source == "refinitiv"
                        else "Govt Mid 7-10Y ETF")

    print(f"\n── Q5 Systematic Risk Regression ({source.upper()}) ──")
    print(f"  Govt factor:      {govt_factor_desc}")
    print(f"  Sample:           {reg_returns.index.min().date()} → "
          f"{reg_returns.index.max().date()} ({len(reg_returns)} obs)")
    print(f"  Alpha (monthly):  {model.params['const']:.4f}  "
          f"(p={model.pvalues['const']:.3f})  "
          f"ann. = {model.params['const']*12*100:.2f}%")
    print(f"  Beta equity:      {model.params['equity']:.4f}  "
          f"(p={model.pvalues['equity']:.3f})")
    print(f"  Beta govt:        {model.params['govt']:.4f}  "
          f"(p={model.pvalues['govt']:.3f})")
    print(f"  R-squared:        {model.rsquared:.3f}")

    return model


def compute_rolling_equity_beta(returns: pd.DataFrame,
                                 window: int = 36) -> pd.Series:
    """
    Rolling beta of corp_ig on equity over trailing window months.
    beta_t = cov(corp, equity) / var(equity)  over window t-w:t

    Works with both portfolio data (has 'corp_ig' col) and regression
    data (also has 'corp_ig' col).
    """
    rolling_beta = pd.Series(index=returns.index, dtype=float)

    for i in range(window, len(returns)):
        window_data = returns.iloc[i - window:i]
        cov_matrix  = window_data[["corp_ig", "equity"]].cov()
        beta        = cov_matrix.loc["corp_ig", "equity"] / cov_matrix.loc["equity", "equity"]
        rolling_beta.iloc[i] = beta

    return rolling_beta.dropna()


# ══════════════════════════════════════════════════════════════
# 5. EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════

def compute_frontier_points(returns: pd.DataFrame,
                             include_corp: bool = True,
                             n_portfolios: int = 7000) -> tuple:
    """
    Monte Carlo simulation of random portfolio weights to approximate
    the efficient frontier.
    """
    if include_corp:
        assets = ["equity", "govt_mid", "corp_ig"]
    else:
        assets = ["equity", "govt_mid"]

    r     = returns[assets].dropna()
    mu    = r.mean() * 12
    sigma = r.cov()  * 12

    port_vols = []
    port_rets = []

    np.random.seed(42)
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(len(assets)))
        p_ret  = np.dot(w, mu)
        p_vol  = np.sqrt(w @ sigma.values @ w)
        port_rets.append(p_ret * 100)
        port_vols.append(p_vol * 100)

    return np.array(port_vols), np.array(port_rets)


# ══════════════════════════════════════════════════════════════
# 6. PLOTS
# ══════════════════════════════════════════════════════════════

def _shade_subperiods(ax):
    """Apply sub-period shading to a time-series axis."""
    for (label, (s, e)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.12, color=color)


def _cumulative_index(log_returns: pd.Series) -> pd.Series:
    """Convert log return series to cumulative index starting at 100."""
    return np.exp(log_returns.cumsum()) * 100 / np.exp(log_returns.cumsum()).iloc[0] * 1


def plot_bond_portfolios_cumulative(port_returns: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 1 — Part 1: Cumulative returns of bond-only portfolios.
    """
    colors = ["darkorange", "steelblue", "green"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for (name, series), color in zip(port_returns.items(), colors):
        cum = np.exp(series.cumsum())
        cum = cum / cum.iloc[0] * 100
        ax.plot(cum.index, cum, label=name.replace("\n", " "),
                color=color, linewidth=2)

    _shade_subperiods(ax)
    ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Part 1: Cumulative Returns — Bond-Only Portfolios",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Return (100 = start)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_diversified_cumulative(port_returns: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 2 — Part 2: Cumulative returns of bond+equity portfolios.
    """
    colors = ["steelblue", "darkorange", "green"]
    fig, ax = plt.subplots(figsize=(14, 6))

    for (name, series), color in zip(port_returns.items(), colors):
        cum = np.exp(series.cumsum())
        cum = cum / cum.iloc[0] * 100
        ax.plot(cum.index, cum, label=name.replace("\n", " "),
                color=color, linewidth=2)

    _shade_subperiods(ax)
    ax.axhline(100, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Part 2: Cumulative Returns — Diversified Bond+Equity Portfolios",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Indexed Return (100 = start)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(returns: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 3: Correlation matrix of all asset returns.
    """
    labels = {
        "equity":     "Equity\n(MSCI Europe)",
        "govt_short": "Govt Short\n(1-3Y)",
        "govt_mid":   "Govt Mid\n(7-10Y)",
        "govt_long":  "Govt Long\n(15-30Y)",
        "corp_ig":    "Corp IG",
    }
    corr = returns.rename(columns=labels).corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                vmin=-1, vmax=1, center=0, ax=ax,
                linewidths=0.5, annot_kws={"size": 10})
    ax.set_title("Asset Return Correlations (Full Sample)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def extract_frontier(vols: np.ndarray, rets: np.ndarray) -> tuple:
    """Extract efficient frontier (upper envelope)."""
    df = pd.DataFrame({"vol": vols, "ret": rets})
    df = df.sort_values("vol")

    frontier = []
    max_ret = -np.inf

    for _, row in df.iterrows():
        if row["ret"] > max_ret:
            frontier.append(row)
            max_ret = row["ret"]

    frontier = pd.DataFrame(frontier)
    return frontier["vol"].values, frontier["ret"].values


def compute_exact_frontier(returns, include_corp=True, n_points=100):

    if include_corp:
        assets = ["equity", "govt_mid", "corp_ig"]
    else:
        assets = ["equity", "govt_mid"]

    r = returns[assets].dropna()
    mu = r.mean() * 12
    Sigma = r.cov() * 12

    n = len(mu)

    def portfolio_vol(w):
        return np.sqrt(w @ Sigma @ w)

    def portfolio_ret(w):
        return w @ mu

    bounds = [(0, 1)] * n
    constraints_base = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    target_returns = np.linspace(mu.min(), mu.max(), n_points)

    vols, rets = [], []

    prev_w = np.ones(n) / n

    for target in target_returns:

        constraints = constraints_base + [
            {'type': 'eq', 'fun': lambda w, target=target: portfolio_ret(w) - target}
        ]

        res = minimize(
            portfolio_vol,
            prev_w,
            bounds=bounds,
            constraints=constraints
        )

        if res.success:
            w_opt = res.x
            vols.append(portfolio_vol(w_opt) * 100)
            rets.append(portfolio_ret(w_opt) * 100)
            prev_w = w_opt
        else:
            prev_w = np.ones(n) / n

    return np.array(vols), np.array(rets)


def plot_efficient_frontier(returns: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Monte Carlo clouds ──
    vols_no, rets_no = compute_frontier_points(returns, include_corp=False)
    vols_yes, rets_yes = compute_frontier_points(returns, include_corp=True)

    ax.scatter(vols_no, rets_no, color="darkorange", alpha=0.05, s=3)
    ax.scatter(vols_yes, rets_yes, color="steelblue", alpha=0.08, s=4)

    # ── Exact frontiers ──
    fvols_no, frets_no = compute_exact_frontier(returns, include_corp=False)
    fvols_yes, frets_yes = compute_exact_frontier(returns, include_corp=True)

    # Smooth
    frets_yes = savgol_filter(frets_yes, 11, 3)

    ax.plot(fvols_no, frets_no,
            color="darkorange", linewidth=1.5,
            label="Frontier: Equity + Govt")

    ax.plot(fvols_yes, frets_yes,
            color="steelblue", linewidth=1.5,
            label="Frontier: Equity + Govt + Corp")

    # ── Sharpe coloring for 3-asset cloud ──
    sharpe_yes = rets_yes / vols_yes
    sc = ax.scatter(vols_yes, rets_yes, c=sharpe_yes, cmap="RdYlGn",
                    alpha=0.25, s=6)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

    # ── Specific portfolios ──
    specific = {
        "60/40 Equity/Govt": {"equity": 0.6, "govt_mid": 0.4},
        "60/30/10 +Corp":    {"equity": 0.6, "govt_mid": 0.3, "corp_ig": 0.1},
        "60/20/20 +Corp":    {"equity": 0.6, "govt_mid": 0.2, "corp_ig": 0.2},
    }

    marker_colors = ["navy", "darkgreen", "darkred"]

    for (name, weights), mc in zip(specific.items(), marker_colors):
        r = build_portfolio_returns(returns, weights)
        p_ret = r.mean() * 12 * 100
        p_vol = r.std() * np.sqrt(12) * 100

        ax.scatter(p_vol, p_ret, color=mc, s=120, zorder=5, marker="*", label=name)
        ax.annotate(name, (p_vol, p_ret),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=8, color=mc)

    ax.set_title("Efficient Frontier: With vs Without Corporate Bonds",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_summary_table(bond_stats: pd.DataFrame,
                        mixed_stats: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 5: Combined summary statistics table for all portfolios.
    """
    all_stats = pd.concat([bond_stats, mixed_stats], ignore_index=True)
    display_cols = ["Portfolio", "Ann. Return (%)", "Ann. Vol (%)",
                    "Sharpe", "Max DD (%)", "Calmar", "Skewness"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    table = ax.table(
        cellText=all_stats[display_cols].values,
        colLabels=display_cols,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.8)

    # Header
    for j in range(len(display_cols)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    sharpe_col = display_cols.index("Sharpe")
    ret_col    = display_cols.index("Ann. Return (%)")

    for i in range(1, len(all_stats) + 1):
        bg = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
        for j in range(len(display_cols)):
            table[(i, j)].set_facecolor(bg)

        try:
            s = float(table[(i, sharpe_col)].get_text().get_text())
            table[(i, sharpe_col)].set_facecolor(
                "#D5F5E3" if s > 0.5 else "#FDEBD0" if s > 0 else "#FADBD8")
        except ValueError:
            pass

        try:
            r = float(table[(i, ret_col)].get_text().get_text())
            table[(i, ret_col)].set_facecolor(
                "#D5F5E3" if r > 0 else "#FADBD8")
        except ValueError:
            pass

    fig.suptitle("Q5: Portfolio Summary Statistics — All Portfolios",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_systematic_risk_table(model, source: str) -> plt.Figure:
    """
    Exhibit 6: Systematic risk regression results as a formatted table.
    corp_ig = alpha + beta_equity * equity + beta_govt * govt + epsilon

    Govt factor label adapts to data source (iBoxx all-mat vs ETF 7-10Y).
    """
    govt_factor_name = ("Govt (iBoxx all mat.)" if source == "refinitiv"
                        else "Govt Mid (7-10Y)")

    rows = [
        ["Alpha (monthly %)", f"{model.params['const']*100:.3f}",
         f"{model.pvalues['const']:.3f}",
         "★" if model.pvalues['const'] < 0.05 else ""],
        ["Alpha (annualized %)", f"{model.params['const']*12*100:.2f}",
         "—", "Residual credit premium after factor adjustment"],
        ["Beta — Equity", f"{model.params['equity']:.4f}",
         f"{model.pvalues['equity']:.3f}",
         "★" if model.pvalues['equity'] < 0.05 else ""],
        [f"Beta — {govt_factor_name}", f"{model.params['govt']:.4f}",
         f"{model.pvalues['govt']:.3f}",
         "★" if model.pvalues['govt'] < 0.05 else ""],
        ["R-squared", f"{model.rsquared:.3f}", "—",
         f"Factor model explains {model.rsquared*100:.1f}% of corp variance"],
        ["1 - R²  (idiosyncratic)", f"{1-model.rsquared:.3f}", "—",
         f"{(1-model.rsquared)*100:.1f}% unexplained by equity + duration"],
        ["Observations", f"{int(model.nobs)}", "—", "Monthly observations"],
    ]

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.axis("off")
    cols = ["Parameter", "Estimate", "p-value", "Interpretation"]
    table = ax.table(cellText=rows, colLabels=cols,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.9)

    for j in range(4):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(rows, 1):
        bg = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
        for j in range(4):
            table[(i, j)].set_facecolor(bg)

    # Highlight alpha row
    try:
        a_pval = model.pvalues['const']
        color = "#D5F5E3" if a_pval < 0.05 else "#FDEBD0"
        for j in range(4):
            table[(1, j)].set_facecolor(color)
    except Exception:
        pass

    source_note = ("iBoxx EUR Sovereigns (all mat.)" if source == "refinitiv"
                   else "Govt Mid 7-10Y ETF")
    fig.suptitle(
        f"Systematic Risk Decomposition: corp_ig = α + β_equity·equity + β_govt·govt + ε\n"
        f"★ = significant at 5% level  |  HC3 robust SE  |  Govt factor: {source_note}",
        fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_rolling_equity_beta(returns: pd.DataFrame,
                              window: int = 36) -> plt.Figure:
    """
    Exhibit 7: Rolling 36-month beta of corporate bonds to equity.
    Uses portfolio data (has 'corp_ig' and 'equity' columns).
    """
    rolling_beta = compute_rolling_equity_beta(returns, window)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling_beta.index, rolling_beta, color="steelblue", linewidth=1.5)
    ax.fill_between(rolling_beta.index, rolling_beta, 0,
                    where=rolling_beta > 0, alpha=0.2, color="steelblue")
    ax.fill_between(rolling_beta.index, rolling_beta, 0,
                    where=rolling_beta < 0, alpha=0.2, color="crimson")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(rolling_beta.mean(), color="steelblue", linewidth=1,
               linestyle="--", label=f"Mean beta = {rolling_beta.mean():.3f}")

    _shade_subperiods(ax)

    ax.set_title(f"Rolling {window}-Month Equity Beta of Corporate IG Bonds",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Beta to Equity (MSCI Europe)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown_all(bond_ports: pd.DataFrame,
                       mixed_ports: pd.DataFrame) -> plt.Figure:
    """
    Exhibit 8: Drawdown for all portfolios side by side.
    """
    all_ports = pd.concat([bond_ports, mixed_ports], axis=1)
    colors_list = ["darkorange", "steelblue", "green",
                   "navy", "darkgreen", "darkred"]

    fig, axes = plt.subplots(len(all_ports.columns), 1,
                              figsize=(14, 2.5 * len(all_ports.columns)),
                              sharex=True)

    for ax, (name, series), color in zip(axes, all_ports.items(), colors_list):
        r   = series.dropna()
        cum = np.exp(r.cumsum())
        dd  = (cum - cum.cummax()) / cum.cummax() * 100

        ax.fill_between(dd.index, dd, 0, color=color, alpha=0.4)
        ax.plot(dd.index, dd, color=color, linewidth=1)
        worst = dd.min()
        worst_date = dd.idxmin()
        ax.set_title(f"{name.replace(chr(10), ' ')}  —  worst: {worst:.1f}% "
                     f"({worst_date.strftime('%Y-%m')})",
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("DD (%)", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(alpha=0.3)

    fig.suptitle("Drawdown Comparison — All Portfolios",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_frontiers_and_cals(returns: pd.DataFrame, rf: float = 0.0) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 7))

    # ── Frontiers ──
    fvols_no, frets_no = compute_exact_frontier(returns, include_corp=False)
    fvols_yes, frets_yes = compute_exact_frontier(returns, include_corp=True)

    ax.plot(fvols_no, frets_no,
            color="darkorange", linewidth=2,
            label="Frontier: Equity + Govt")

    ax.plot(fvols_yes, frets_yes,
            color="steelblue", linewidth=2,
            label="Frontier: Equity + Govt + Corp")

    # ── Tangency portfolios ──
    def compute_tangent(returns, include_corp):
        if include_corp:
            assets = ["equity", "govt_mid", "corp_ig"]
        else:
            assets = ["equity", "govt_mid"]

        r = returns[assets].dropna()
        mu = r.mean() * 12
        Sigma = r.cov() * 12

        excess = mu - rf
        w = np.linalg.inv(Sigma) @ excess
        w /= np.sum(w)

        ret = w @ mu
        vol = np.sqrt(w @ Sigma @ w)

        return ret * 100, vol * 100

    ret_no, vol_no = compute_tangent(returns, include_corp=False)
    ret_yes, vol_yes = compute_tangent(returns, include_corp=True)

    # ── CAL lines ──
    x = np.linspace(0, max(fvols_yes) * 1.2, 200)

    ax.plot(x, (ret_no / vol_no) * x,
            linestyle="--", color="darkorange", alpha=0.8,
            label="CAL (no corp)")

    ax.plot(x, (ret_yes / vol_yes) * x,
            linestyle=(3, (6, 4)),
            color="steelblue", alpha=0.8,
            label="CAL (with corp)")

    # ── Mark tangency points ──
    ax.scatter(vol_no, ret_no, color="darkorange", s=80, zorder=5)
    ax.scatter(vol_yes, ret_yes, color="steelblue", s=80, zorder=5)

    # ── Styling ──
    ax.set_title("Efficient Frontier & Capital Allocation Lines",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Return (%)")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 7. RUN ALL
# ══════════════════════════════════════════════════════════════

def run_all() -> dict:
    """
    Run full Q5 analysis.
    Returns dict of {filename: Figure} for main.py to save.

    Data strategy:
      - Portfolio construction uses ETF data (needs maturity buckets)
      - Systematic risk regression uses Refinitiv iBoxx (longer sample)
        with ETF fallback if unavailable
    """
    # ── Load data ───────────────────────────────────────────
    returns = load_all_returns()                          # ETF-based, for portfolios
    reg_returns, reg_source = load_regression_returns()   # Refinitiv-preferred, for regression

    # ── Build portfolios ────────────────────────────────────
    bond_port_returns  = build_all_portfolios(returns, BOND_PORTFOLIOS)
    mixed_port_returns = build_all_portfolios(returns, MIXED_PORTFOLIOS)

    # ── Statistics ──────────────────────────────────────────
    bond_stats  = compute_all_stats(bond_port_returns)
    mixed_stats = compute_all_stats(mixed_port_returns)

    print("\n── Q5 Bond Portfolio Stats ──")
    print(bond_stats[["Portfolio", "Ann. Return (%)", "Sharpe",
                       "Max DD (%)"]].to_string(index=False))

    print("\n── Q5 Mixed Portfolio Stats ──")
    print(mixed_stats[["Portfolio", "Ann. Return (%)", "Sharpe",
                        "Max DD (%)"]].to_string(index=False))

    # ── Systematic risk (uses longer Refinitiv sample if available) ──
    model = run_systematic_risk_regression(reg_returns, reg_source)

    # ── Figures ─────────────────────────────────────────────
    return {
        "Q5_bond_portfolio_cumulative.png": plot_bond_portfolios_cumulative(bond_port_returns),
        "Q5_diversified_cumulative.png":    plot_diversified_cumulative(mixed_port_returns),
        "Q5_correlation_heatmap.png":       plot_correlation_heatmap(returns),
        "Q5_efficient_frontier.png":        plot_efficient_frontier(returns),
        "Q5_summary_table.png":             plot_summary_table(bond_stats, mixed_stats),
        "Q5_systematic_risk_regression.png":plot_systematic_risk_table(model, reg_source),
        "Q5_rolling_equity_beta.png":       plot_rolling_equity_beta(returns),
        "Q5_drawdown_all.png":              plot_drawdown_all(bond_port_returns,
                                                              mixed_port_returns),
        "Q5_frontier_cals.png":             plot_frontiers_and_cals(returns),
    }