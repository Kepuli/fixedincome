# main.py
# ─────────────────────────────────────────────
# Master script. Runs all analysis modules and
# saves outputs. Run this to reproduce all charts.
#
# Usage: python main.py
# ─────────────────────────────────────────────
from src.config import OUTPUTS
import src.Q1_yield_curve   as q1
import src.Q2_forward_rates as q2
import src.Q3_duration_risk as q3
import src.Q4_credit_markets as q4_q5
import matplotlib.pyplot as plt

def save_figures(figures: dict):
    for filename, fig in figures.items():
        path = OUTPUTS / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
        plt.close(fig)

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

run_module("Q1: Yield Curve Dynamics", q1)
run_module("Q2: Forward Rates",        q2)
run_module("Q3: Duration Risk",        q3)
run_module("Q4/Q5: Credit Markets",    q4_q5)

print("\nDone. All outputs saved to /outputs/")