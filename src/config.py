# src/config.py
# ─────────────────────────────────────────────
# Central config — import this in every module.
# If sub-periods or paths change, edit here only.
# ─────────────────────────────────────────────
from pathlib import Path

# Project root (works regardless of where script is run from)
ROOT = Path(__file__).parent.parent
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS        = ROOT / "outputs"

SUBPERIODS = {
    "Pre-GFC (2004–2007)":         ("2004-09", "2007-07"),
    "Pre-Lehman (2007–2008)":      ("2007-08", "2008-08"),
    "Post-Lehman (2008–2009)":     ("2008-09", "2009-12"),
    "Sovereign Crisis (2010–12)":  ("2010-01", "2012-12"),
    "QE Era (2013–2021)":          ("2013-01", "2021-12"),
    "Hiking Cycle (2022–2025)":    ("2022-01", "2025-12"),
}

COLORS = ["#d4e6f1", "#d5f5e3", "#fdebd0", "#e8daef", "#fadbd8"]

MATURITIES     = [1, 2, 3, 5, 10, 30]
MATURITY_COLS  = ["spot_1y","spot_2y","spot_3y",
                  "spot_5y","spot_10y","spot_30y"]