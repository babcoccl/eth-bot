#!/usr/bin/env python3
"""
correction_windows_4yr.py  —  ETH-USD CORRECTION regime windows 2022-2026
==========================================================================
A CORRECTION window is defined as:
  - Price drops 5–25% from local high
  - Develops over 5–35 days
  - Followed by recovery (not a full crash)
  - NOT classified as CRASH (>25% in <30 days with sharp capitulation)

Severity:
  SHALLOW    : -5%  to -10%
  MODERATE   : -10% to -18%
  DEEP       : -18% to -25%

Windows are manually cataloged from 4yr ETH/USD price history.
Each dict: label, start, end, days, drop_pct, severity
  start = first day of sustained decline from local high
  end   = approximate local bottom / regime transition date
"""

CORRECTION_WINDOWS = [
    # ── 2022 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#C01 Mar22",
        "start":    "2022-03-28",
        "end":      "2022-04-05",
        "days":     8.0,
        "drop_pct": -12.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C04 Jul22",
        "start":    "2022-07-30",
        "end":      "2022-08-05",
        "days":     6.0,
        "drop_pct": -8.0,
        "severity": "SHALLOW",
    },
    {
        "label":    "#C05 Sep22",
        "start":    "2022-09-08",
        "end":      "2022-09-21",
        "days":     13.0,
        "drop_pct": -22.0,
        "severity": "DEEP",
    },
    {
        "label":    "#C06 Oct-Nov22",
        "start":    "2022-10-28",
        "end":      "2022-11-07",
        "days":     10.0,
        "drop_pct": -10.0,
        "severity": "MODERATE",
    },
    # ── 2023 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#C07 Feb23",
        "start":    "2023-02-16",
        "end":      "2023-02-25",
        "days":     9.0,
        "drop_pct": -11.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C08 Apr23",
        "start":    "2023-04-19",
        "end":      "2023-04-27",
        "days":     8.0,
        "drop_pct": -9.0,
        "severity": "SHALLOW",
    },
    {
        "label":    "#C09 Aug23",
        "start":    "2023-08-17",
        "end":      "2023-08-29",
        "days":     12.0,
        "drop_pct": -15.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C10 Oct23",
        "start":    "2023-10-03",
        "end":      "2023-10-12",
        "days":     9.0,
        "drop_pct": -8.0,
        "severity": "SHALLOW",
    },
    # ── 2024 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#C11 Apr24",
        "start":    "2024-04-09",
        "end":      "2024-04-20",
        "days":     11.0,
        "drop_pct": -22.0,
        "severity": "DEEP",
    },
    {
        "label":    "#C12 May24",
        "start":    "2024-05-20",
        "end":      "2024-05-31",
        "days":     11.0,
        "drop_pct": -12.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C13 Sep24",
        "start":    "2024-09-03",
        "end":      "2024-09-12",
        "days":     9.0,
        "drop_pct": -9.0,
        "severity": "SHALLOW",
    },
    {
        "label":    "#C14 Nov24",
        "start":    "2024-11-25",
        "end":      "2024-12-05",
        "days":     10.0,
        "drop_pct": -10.0,
        "severity": "MODERATE",
    },
    # ── 2025 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#C15 Feb25",
        "start":    "2025-02-03",
        "end":      "2025-02-11",
        "days":     8.0,
        "drop_pct": -14.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C16 Jun25",
        "start":    "2025-06-23",
        "end":      "2025-07-01",
        "days":     8.0,
        "drop_pct": -9.0,
        "severity": "SHALLOW",
    },
    {
        "label":    "#C17 Aug25",
        "start":    "2025-08-05",
        "end":      "2025-08-14",
        "days":     9.0,
        "drop_pct": -11.0,
        "severity": "MODERATE",
    },
    {
        "label":    "#C18 Sep25",
        "start":    "2025-09-18",
        "end":      "2025-09-26",
        "days":     8.0,
        "drop_pct": -8.0,
        "severity": "SHALLOW",
    },
    # ── 2026 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#C19 Feb26",
        "start":    "2026-02-10",
        "end":      "2026-02-20",
        "days":     10.0,
        "drop_pct": -13.0,
        "severity": "MODERATE",
    },
]
