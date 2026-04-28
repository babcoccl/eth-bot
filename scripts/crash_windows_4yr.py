
# ============================================================
# CRASH_WINDOWS  —  4.3yr dataset  (2022-01-01 -> 2026-04-11)
# Severity: CATASTROPHIC > MAJOR > MODERATE > SHALLOW
# Each entry includes next recovery window start for context
# ============================================================
CRASH_WINDOWS = [
    # ──── 2022 bear market ────────────────────────────────────
    {"label": "#2  Jan-Feb22",  "start": "2022-01-05", "end": "2022-02-04",
     "severity": "MAJOR",        "days": 29.3,  "chg_pct": -24.4,
     "recovery_start": "2022-02-04"},
    {"label": "#11 Apr-Jul22",  "start": "2022-04-06", "end": "2022-07-06",
     "severity": "CATASTROPHIC", "days": 90.6,  "chg_pct": -64.7,
     "recovery_start": "2022-07-24",  # window #14 CORRECTION +37%
     "stress_test": True},
    {"label": "#19 Aug-Oct22",  "start": "2022-08-19", "end": "2022-10-16",
     "severity": "CATASTROPHIC", "days": 58.7,  "chg_pct": -26.1,
     "recovery_start": "2022-10-22"},  # window #21 +24.1%
    {"label": "#24 Nov-Dec22",  "start": "2022-11-08", "end": "2022-12-08",
     "severity": "MODERATE",     "days": 30.5,  "chg_pct": -14.6,
     "recovery_start": "2023-01-02"},  # window #30 +7.9% -> #31 BULL +24.9%
    # ──── 2024 corrections ────────────────────────────────────
    {"label": "#66 Mar-May24",  "start": "2024-03-15", "end": "2024-05-03",
     "severity": "MAJOR",        "days": 48.9,  "chg_pct": -18.1,
     "recovery_start": "2024-05-20"},  # window #70 +26.3%
    {"label": "#76 Jul-Aug24",  "start": "2024-07-25", "end": "2024-08-08",
     "severity": "MAJOR",        "days": 14.6,  "chg_pct": -21.2,
     "recovery_start": "2024-11-18"},  # long recovery, then #86 +11%
    # ──── 2025 bear ───────────────────────────────────────────
    {"label": "#93 Jan-Mar25",  "start": "2025-01-26", "end": "2025-03-23",
     "severity": "CATASTROPHIC", "days": 55.2,  "chg_pct": -38.1,
     "recovery_start": "2025-05-08"},  # #99 BULL +36.1%
    {"label": "#95 Mar-Apr25",  "start": "2025-03-26", "end": "2025-04-24",
     "severity": "MODERATE",     "days": 29.5,  "chg_pct": -12.7,
     "recovery_start": "2025-04-25"},
    # ──── already validated (keep for regression) ─────────────
    {"label": "#104 Jun-Jul25", "start": "2025-06-05", "end": "2025-07-06",
     "severity": "SHALLOW",      "days": 30.7,  "chg_pct": +5.1,
     "recovery_start": "2025-07-06"},
    {"label": "#120 Oct25",     "start": "2025-10-10", "end": "2025-10-24",
     "severity": "SHALLOW",      "days": 14.2,  "chg_pct": -4.0,
     "recovery_start": "2025-10-24"},
    {"label": "#122 Oct-Dec25", "start": "2025-10-28", "end": "2025-12-09",
     "severity": "MAJOR",        "days": 42.4,  "chg_pct": -23.4,
     "recovery_start": "2025-12-30"},
    {"label": "#129 Jan-Mar26", "start": "2026-01-20", "end": "2026-03-04",
     "severity": "CATASTROPHIC", "days": 42.9,  "chg_pct": -30.4,
     "recovery_start": "2026-03-12"},
    {"label": "#133 Mar-Apr26", "start": "2026-03-22", "end": "2026-04-07",
     "severity": "SHALLOW",      "days": 16.2,  "chg_pct": -0.0,
     "recovery_start": "2026-04-07"},
]
