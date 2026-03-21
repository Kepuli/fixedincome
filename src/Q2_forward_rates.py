# src/Q2_forward_rates.py
# ─────────────────────────────────────────────
# Q2: How well have Euro area forward rates predicted future interest rates?
# INPUT:  data/processed/ecb_spot_rates.parquet
# OUTPUT: Q2_scatter.png, Q2_timeseries.png, Q2_regression_table.png
# ─────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import statsmodels.api as sm
from src.config import SUBPERIODS, COLORS
from src.data_loader import load_spot_rates


# ── 1. Compute implied forward rates ─────────────────────────

def compute_forward_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute discrete 1-year forward rates at annual horizons from spot curve.
    Formula: f(n, n+1) = ((1 + s_{n+1})^{n+1} / (1 + s_n)^n) - 1
    Returns DataFrame with columns: fwd_1y1y, fwd_2y1y, fwd_3y1y, fwd_4y1y
    """
    s = df[["spot_1y", "spot_2y", "spot_3y", "spot_4y", "spot_5y"]] / 100  # to decimal

    fwd = pd.DataFrame(index=df.index)
    fwd["fwd_1y1y"] = ((1 + s["spot_2y"]) ** 2 / (1 + s["spot_1y"]) ** 1) - 1
    fwd["fwd_2y1y"] = ((1 + s["spot_3y"]) ** 3 / (1 + s["spot_2y"]) ** 2) - 1
    fwd["fwd_3y1y"] = ((1 + s["spot_4y"]) ** 4 / (1 + s["spot_3y"]) ** 3) - 1
    fwd["fwd_4y1y"] = ((1 + s["spot_5y"]) ** 5 / (1 + s["spot_4y"]) ** 4) - 1

    return fwd * 100  # back to percent


# ── 2. Align with realized future spot rates ──────────────────

def build_regression_data(df: pd.DataFrame, fwd: pd.DataFrame) -> dict:
    """
    For each horizon h, align forward rate at t with realized spot_1y at t+h.
    shift(-h) moves the realized rate backward so it lines up with the forward at t.
    Returns dict of DataFrames ready for regression, one per horizon.
    """
    horizons = {
        "1Y": ("fwd_1y1y", 12),
        "2Y": ("fwd_2y1y", 24),
        "3Y": ("fwd_3y1y", 36),
        "4Y": ("fwd_4y1y", 48),
    }

    reg_data = {}
    for label, (fwd_col, shift) in horizons.items():
        combined = pd.DataFrame({
            "forward_rate": fwd[fwd_col],
            "realized_rate": df["spot_1y"].shift(-shift),
        }).dropna()
        reg_data[label] = combined

    return reg_data


# ── 3. Run OLS regressions ────────────────────────────────────

def run_regressions(reg_data: dict) -> dict:
    """
    OLS: realized_rate = α + β × forward_rate + ε
    β = 1 → expectations hypothesis holds
    β < 1 → term premium exists
    β < 0 → forward rates systematically wrong
    """
    results = {}
    for label, df in reg_data.items():
        X = sm.add_constant(df["forward_rate"])
        y = df["realized_rate"]
        model = sm.OLS(y, X).fit()
        results[label] = model
    return results


# ── 4. Plots ──────────────────────────────────────────────────

def plot_scatter(reg_data: dict, results: dict) -> plt.Figure:
    """Exhibit 1: Scatter plots — forward rate vs realized rate per horizon."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (label, df) in enumerate(reg_data.items()):
        ax = axes[i]
        model = results[label]
        alpha = model.params["const"]
        beta = model.params["forward_rate"]
        r2 = model.rsquared

        # Scatter
        ax.scatter(df["forward_rate"], df["realized_rate"],
                   alpha=0.4, s=20, color="steelblue")

        # Regression line
        x_range = np.linspace(df["forward_rate"].min(), df["forward_rate"].max(), 100)
        ax.plot(x_range, alpha + beta * x_range, color="red",
                linewidth=2, label=f"β = {beta:.2f}, R² = {r2:.2f}")

        # 45-degree line (perfect prediction)
        lims = [min(df["forward_rate"].min(), df["realized_rate"].min()),
                max(df["forward_rate"].max(), df["realized_rate"].max())]
        ax.plot(lims, lims, color="black", linewidth=1,
                linestyle="--", label="β = 1 (perfect)")

        ax.set_title(f"Horizon: {label} forward")
        ax.set_xlabel("Forward Rate (%)")
        ax.set_ylabel("Realized Spot Rate (%)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Forward Rates vs Realized Rates — Fama-Bliss Regression",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_timeseries(df: pd.DataFrame, fwd: pd.DataFrame) -> plt.Figure:
    """Exhibit 2: Time series — forward rate vs realized rate per horizon."""
    horizons = [
        ("fwd_1y1y", 12, "1Y horizon"),
        ("fwd_2y1y", 24, "2Y horizon"),
        ("fwd_3y1y", 36, "3Y horizon"),
        ("fwd_4y1y", 48, "4Y horizon"),
    ]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    for ax, (fwd_col, shift, label) in zip(axes, horizons):
        realized = df["spot_1y"].shift(-shift)
        ax.plot(fwd.index, fwd[fwd_col], label="Forward rate (implied)",
                color="steelblue", linewidth=1.5)
        ax.plot(df.index, realized, label="Realized spot rate",
                color="red", linewidth=1.5, linestyle="--")
        ax.set_title(label)
        ax.set_ylabel("Rate (%)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    fig.suptitle("Forward Rates vs Realized Future Spot Rates",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_regression_table(results: dict) -> plt.Figure:
    """Exhibit 3: Regression summary table as a figure for the report."""
    rows = []
    for label, model in results.items():
        rows.append({
            "Horizon":   label,
            "α (const)": f"{model.params['const']:.3f}",
            "β":         f"{model.params['forward_rate']:.3f}",
            "β p-value": f"{model.pvalues['forward_rate']:.3f}",
            "R²":        f"{model.rsquared:.3f}",
            "N obs":     str(int(model.nobs)),
        })

    df_table = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")
    table = ax.table(
        cellText=df_table.values,
        colLabels=df_table.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Header styling
    for j in range(len(df_table.columns)):
        table[(0, j)].set_facecolor("#2c3e50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Highlight β column — color by value
    for i, row in enumerate(rows):
        beta = float(row["β"])
        color = "#d5f5e3" if beta > 0.8 else "#fdebd0" if beta > 0 else "#fadbd8"
        table[(i + 1, 2)].set_facecolor(color)

    fig.suptitle("Fama-Bliss Regression Results: realized = α + β × forward + ε",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout()
    return fig


# ── 5. run_all ────────────────────────────────────────────────

def run_all() -> dict:
    """Run full Q2 analysis. Returns dict of {filename: Figure}."""
    df = load_spot_rates()
    fwd = compute_forward_rates(df)
    reg_data = build_regression_data(df, fwd)
    results = run_regressions(reg_data)

    # Print results to console for quick inspection
    print("\n── Q2 Regression Results ──")
    for label, model in results.items():
        beta = model.params["forward_rate"]
        pval = model.pvalues["forward_rate"]
        r2 = model.rsquared
        print(f"  {label}: β = {beta:.3f} (p={pval:.3f}), R² = {r2:.3f}")

    return {
        "Q2_scatter.png":          plot_scatter(reg_data, results),
        "Q2_timeseries.png":       plot_timeseries(df, fwd),
        "Q2_regression_table.png": plot_regression_table(results),
    }