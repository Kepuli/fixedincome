# src/yield_curve.py
# ─────────────────────────────────────────────
# Q1: Evolution of Euro area interest rates
#     and yield curve shape.
# INPUT:  ecb_spot_rates.parquet
# OUTPUT: figures returned as matplotlib Figure
#         objects — saved by main.py
# ─────────────────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.config import SUBPERIODS, COLORS, MATURITIES, MATURITY_COLS
from src.data_loader import load_spot_rates

def plot_rate_levels() -> plt.Figure:
    """Exhibit 1: Time series of 2Y, 5Y, 10Y spot rates with sub-period shading."""
    df = load_spot_rates()
    fig, ax = plt.subplots(figsize=(14, 5))

    for col, label in [("spot_2y","2Y"),("spot_5y","5Y"),("spot_10y","10Y")]:
        ax.plot(df.index, df[col], label=label, linewidth=1.5)

    for (label, (start, end)), color in zip(SUBPERIODS.items(), COLORS):
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.25, color=color, label=label)

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title("Euro Area Spot Rates 2004–2025")
    ax.set_ylabel("Yield (%)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig

def plot_curve_slope() -> plt.Figure:
    """Exhibit 2: 10Y-2Y slope over time, shaded for inversions."""
    df = load_spot_rates()
    df["slope"] = df["spot_10y"] - df["spot_2y"]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df.index, df["slope"], color="steelblue", linewidth=1.5)
    ax.fill_between(df.index, df["slope"], 0,
                    where=df["slope"] >= 0, alpha=0.2,
                    color="steelblue", label="Normal")
    ax.fill_between(df.index, df["slope"], 0,
                    where=df["slope"] < 0, alpha=0.4,
                    color="red", label="Inverted")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Euro Yield Curve Slope: 10Y − 2Y")
    ax.set_ylabel("Spread (pp)")
    ax.legend()
    fig.tight_layout()
    return fig

def plot_avg_curves_by_subperiod() -> plt.Figure:
    """Exhibit 3: Average yield curve shape per sub-period."""
    df = load_spot_rates()
    fig, ax = plt.subplots(figsize=(10, 6))

    for label, (start, end) in SUBPERIODS.items():
        mean_curve = df.loc[start:end, MATURITY_COLS].mean()
        ax.plot(MATURITIES, mean_curve.values, marker="o",
                label=label, linewidth=2)

    ax.set_title("Average Euro Yield Curve by Sub-period")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Average Spot Rate (%)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig

def plot_heatmap() -> plt.Figure:
    """Exhibit 4: Heatmap of yield levels across maturities and time."""
    df = load_spot_rates()
    heatmap_data = df[MATURITY_COLS].T
    heatmap_data.index = ["1Y","2Y","3Y","5Y","10Y","30Y"]

    # After creating the heatmap, format x-axis labels
    # Replace the xticklabels parameter with this approach:

    fig, ax = plt.subplots(figsize=(16, 5))
    sns.heatmap(heatmap_data, cmap="RdYlGn_r", ax=ax,
                cbar_kws={"label": "Spot Rate (%)"})

    # Fix x-axis labels — show only year
    n_cols = heatmap_data.shape[1]
    step = 24  # every 24 months
    tick_positions = range(0, n_cols, step)
    tick_labels = [str(heatmap_data.columns[i].year) for i in tick_positions]
    ax.set_xticks(list(tick_positions))
    ax.set_xticklabels(tick_labels, rotation=45)
    ax.set_xlabel("")
    return fig

def run_all() -> dict:
    """Run all Q1 exhibits. Returns dict of {filename: Figure}."""
    return {
        "Q1_rate_levels.png":            plot_rate_levels(),
        "Q1_curve_slope.png":            plot_curve_slope(),
        "Q1_avg_curves_by_subperiod.png":plot_avg_curves_by_subperiod(),
        "Q1_heatmap.png":                plot_heatmap(),
    }