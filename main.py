# main.py
# ─────────────────────────────────────────────
# Master script. Runs all analysis modules and saves outputs.
# Usage: python main.py
# ─────────────────────────────────────────────
from src.config import OUTPUTS
import src.Q1_yield_curve          as q1
import src.Q2_forward_rates        as q2
import src.Q3_duration_risk        as q3
import src.Q4_credit_markets       as q4
import src.Q5_equity_bond_portfolio as q5
import matplotlib.pyplot as plt

OUTPUTS.mkdir(exist_ok=True)

def save_figures(figures: dict):
    for filename, fig in figures.items():
        path = OUTPUTS / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
        plt.close(fig)

def save_console_summary(outputs_path):
    """
    Save all key numerical results to a clean CSV and formatted text file
    for easy copy-paste into the report.
    """
    import csv

    # ── Q2 Regression Results ─────────────────────────────
    q2_rows = [
        ["Horizon", "Beta", "p-value", "R²", "Interpretation"],
        ["1Y", 0.792, 0.000, 0.615, "Strong predictive power, some term premium"],
        ["2Y", 0.429, 0.000, 0.194, "Moderate power, significant term premium"],
        ["3Y", 0.077, 0.188, 0.008, "No predictive power, large term premium"],
        ["4Y", -0.175, 0.000, 0.060, "Negative — forward rates predict wrong direction"],
    ]

    # ── Q3 Duration Risk ──────────────────────────────────
    q3_rows = [
        ["Bucket", "Ann. Return (%)", "Ann. Vol (%)", "Sharpe", "Max Drawdown (%)"],
        ["Short (1-3Y)",  1.82, 11.23, 0.16, -42.16],
        ["Mid (7-10Y)",   2.22,  5.82, 0.38, -22.87],
        ["Long (15-30Y)", 0.83,  1.84, 0.45,  -5.84],
    ]

    # ── Q4 Credit Premium ─────────────────────────────────
    q4_rows = [
        ["Period", "Mean monthly (%)", "Annualized (%)"],
        ["Full sample",              -0.026, -0.31],
        ["Post-Lehman (2008-2009)",  +0.304, +3.65],
        ["Sovereign Crisis (2010-12)", -0.197, -2.36],
        ["QE Era (2013-2021)",       -0.107, -1.28],
        ["Hiking Cycle (2022-2025)", +0.237, +2.84],
    ]

    # ── Q5 Bond Portfolios ────────────────────────────────
    q5_bond_rows = [
        ["Portfolio", "Ann. Return (%)", "Sharpe", "Max DD (%)"],
        ["Pure Govt (100% govt_mid)",          2.28, 0.40, -22.87],
        ["Blended Bond (50/50 govt/corp)",      2.12, 0.45, -18.80],
        ["Diversified Bond (33% short/mid/corp)", 2.14, 0.32, -26.70],
    ]

    # ── Q5 Mixed Portfolios ───────────────────────────────
    q5_mixed_rows = [
        ["Portfolio", "Ann. Return (%)", "Sharpe", "Max DD (%)"],
        ["60/40 (equity/govt)",          5.49, 0.52, -25.08],
        ["60/30/10 (equity/govt/corp)",  5.46, 0.51, -24.70],
        ["60/20/20 (equity/govt/corp)",  5.42, 0.50, -24.32],
    ]

    # ── Q5 Systematic Risk ────────────────────────────────
    q5_reg_rows = [
        ["Parameter", "Estimate", "p-value", "Interpretation"],
        ["Alpha (monthly %)",    -0.0001, 0.928, "Not significant — no free credit alpha"],
        ["Alpha (annualized %)", -0.08,   "—",   "After stripping equity + duration risk"],
        ["Beta equity",          0.1210,  0.000, "Significant equity-like systematic risk"],
        ["Beta govt",            0.4921,  0.000, "Significant duration exposure"],
        ["R-squared",            0.561,   "—",   "56% of corp variance explained by 2 factors"],
        ["Observations",         199,     "—",   "Monthly, 2009-06 to 2025-12"],
    ]

    # ── Save individual CSVs ──────────────────────────────
    tables = {
        "Q2_regression_results.csv":    q2_rows,
        "Q3_duration_summary.csv":      q3_rows,
        "Q4_credit_premium.csv":        q4_rows,
        "Q5_bond_portfolios.csv":       q5_bond_rows,
        "Q5_mixed_portfolios.csv":      q5_mixed_rows,
        "Q5_systematic_risk.csv":       q5_reg_rows,
    }

    for filename, rows in tables.items():
        path = outputs_path / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        print(f"  saved → {path}")

    # ── Save one combined text summary ────────────────────
    txt_path = outputs_path / "all_results_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        sections = [
            ("Q2: Forward Rate Regression Results", q2_rows),
            ("Q3: Duration Risk — Full Sample",      q3_rows),
            ("Q4: Credit Premium by Sub-period",     q4_rows),
            ("Q5: Bond-Only Portfolios",             q5_bond_rows),
            ("Q5: Mixed Bond+Equity Portfolios",     q5_mixed_rows),
            ("Q5: Systematic Risk Regression",       q5_reg_rows),
        ]
        for title, rows in sections:
            f.write(f"\n{'─'*60}\n{title}\n{'─'*60}\n")
            col_widths = [max(len(str(row[i])) for row in rows)
                          for i in range(len(rows[0]))]
            for row in rows:
                line = "  ".join(str(v).ljust(col_widths[i])
                                 for i, v in enumerate(row))
                f.write(line + "\n")

    print(f"  saved → {txt_path}")


def run_module(label: str, module):
    """Run a module's run_all() safely — skips if not implemented yet."""
    print(f"── {label} ──")
    if not hasattr(module, "run_all"):
        print(f"  ⚠ skipped — run_all() not implemented yet\n")
        return
    try:
        save_figures(module.run_all())
    except NotImplementedError:
        print(f"  ⚠ skipped — not implemented yet\n")
    except Exception as e:
        print(f"  ✗ failed — {e}\n")

#run_module("Q1: Yield Curve Dynamics",          q1)
#run_module("Q2: Forward Rates",                 q2)
#run_module("Q3: Duration Risk",                 q3)
#run_module("Q4: Credit Markets",                q4)
run_module("Q5: Portfolio & Systematic Risk",   q5)

print("\n── Saving numerical summary tables ──")
save_console_summary(OUTPUTS)

print("\nDone. All outputs saved to /outputs/")
