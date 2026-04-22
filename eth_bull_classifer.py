"""
eth_bull_classifier.py
======================
Shared BULL depth classification logic used by MacroSupervisor,
the backtest harness, and live tactical bots.

No external dependencies beyond numpy. Import freely from any module.
"""

import numpy as np
from typing import Dict, Tuple

DEEP_THRESHOLD        = -0.30   # cycle trough <= -30% -> DEEP
SHALLOW_RECOV_CUTOFF  = -0.13   # cycle trough <= -13% -> SHALLOW_RECOV_DEEP

PAUSE_REGIMES = {"CRASH", "CORRECTION", "RECOVERY"}
PEAK_REGIMES  = {"BULL", "RANGE"}
MIN_PEAK_BARS = 48

STOP_LOSS_BY_CLASS: Dict[str, float] = {
    "DEEP":                0.20,
    "SHALLOW_RECOV_LIGHT": 0.10,
    "SHALLOW_RECOV_DEEP":  0.10,
    "SHALLOW_CONT":        0.10,
}


def classify_bull_depth(cycle_trough_pct: float) -> str:
    ...  # exact same body as today


def _cycle_trough_pct(
    regime_arr, close_arr, entry_idx, min_peak_bars=MIN_PEAK_BARS, debug=False
) -> Tuple[float, int, int]:
    ...  # exact same body as today
