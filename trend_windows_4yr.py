#!/usr/bin/env python3
"""
trend_windows_4yr.py  —  ETH-USD BULL / RECOVERY trend windows 2022-2026
=========================================================================
A TREND window is defined as:
  - h1 regime confirmed UPTREND for >= 7 consecutive days
  - Price gains >= 8% over the window
  - Preceded by a correction or crash bottom (not mid-rally entries)

Strength:
  MODERATE : +8%  to +20% gain over window
  STRONG   : +20% to +40%
  PARABOLIC: >+40%

Windows are manually cataloged from 4yr ETH/USD price history.
Each dict: label, start, end, days, gain_pct, strength
  start = confirmed UPTREND h1 regime entry date (after bottom)
  end   = first regime break or exhaustion signal date
"""

TREND_WINDOWS = [
    # ── 2022 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#T01 Mar-Apr22",
        "start":    "2022-03-15",
        "end":      "2022-04-02",
        "days":     18.0,
        "gain_pct": +16.0,
        "strength": "MODERATE",
    },
    {
        "label":    "#T02 Aug22",
        "start":    "2022-08-05",
        "end":      "2022-08-14",
        "days":     9.0,
        "gain_pct": +38.0,
        "strength": "STRONG",
    },
    # ── 2023 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#T03 Jan23",
        "start":    "2023-01-01",
        "end":      "2023-02-03",
        "days":     33.0,
        "gain_pct": +36.0,
        "strength": "STRONG",
    },
    {
        "label":    "#T04 Mar-Apr23",
        "start":    "2023-03-10",
        "end":      "2023-04-18",
        "days":     39.0,
        "gain_pct": +22.0,
        "strength": "STRONG",
    },
    {
        "label":    "#T05 Oct-Nov23",
        "start":    "2023-10-12",
        "end":      "2023-11-20",
        "days":     39.0,
        "gain_pct": +30.0,
        "strength": "STRONG",
    },
    # ── 2024 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#T06 Jan-Mar24",
        "start":    "2024-01-23",
        "end":      "2024-03-12",
        "days":     49.0,
        "gain_pct": +55.0,
        "strength": "PARABOLIC",
    },
    {
        "label":    "#T07 May-Jun24",
        "start":    "2024-05-01",
        "end":      "2024-06-10",
        "days":     40.0,
        "gain_pct": +14.0,
        "strength": "MODERATE",
    },
    {
        "label":    "#T08 Sep-Oct24",
        "start":    "2024-09-12",
        "end":      "2024-10-20",
        "days":     38.0,
        "gain_pct": +20.0,
        "strength": "MODERATE",
    },
    {
        "label":    "#T09 Oct-Nov24",
        "start":    "2024-10-22",
        "end":      "2024-11-22",
        "days":     31.0,
        "gain_pct": +45.0,
        "strength": "PARABOLIC",
    },
    # ── 2025 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#T10 Jan25",
        "start":    "2025-01-13",
        "end":      "2025-02-01",
        "days":     19.0,
        "gain_pct": +12.0,
        "strength": "MODERATE",
    },
    {
        "label":    "#T11 May25",
        "start":    "2025-05-08",
        "end":      "2025-05-16",
        "days":     8.0,
        "gain_pct": +36.0,
        "strength": "STRONG",
    },
    {
        "label":    "#T12 Jul-Aug25",
        "start":    "2025-07-06",
        "end":      "2025-07-31",
        "days":     25.0,
        "gain_pct": +28.0,
        "strength": "STRONG",
    },
    {
        "label":    "#T13 Oct-Nov25",
        "start":    "2025-10-10",
        "end":      "2025-11-20",
        "days":     41.0,
        "gain_pct": +22.0,
        "strength": "STRONG",
    },
    # ── 2026 ─────────────────────────────────────────────────────────────────
    {
        "label":    "#T14 Jan26",
        "start":    "2025-12-30",
        "end":      "2026-01-06",
        "days":     7.0,
        "gain_pct": +8.0,
        "strength": "MODERATE",
    },
]
