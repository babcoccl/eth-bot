# ETH Regime Transition Analysis — v30

Generated from `eth_audit_transitions.csv` (2021-01-01 → 2026-04-15).

---

## Transition Probability Matrix

| FROM \ TO   | CRASH  | CORRECTION | RECOVERY | BULL   | RANGE  |
|-------------|--------|------------|----------|--------|--------|
| **CRASH**   | —      | 41.5%      | 58.5%    | —      | —      |
| **CORRECTION** | 68.2% | —         | 31.8%    | —      | —      |
| **RECOVERY** | 64.5% | —          | —        | 13.2%  | 22.4%  |
| **BULL**    | 4.3%   | —          | —        | —      | 95.7%  |
| **RANGE**   | 21.3%  | —          | —        | 78.7%  | —      |

---

## Timing (median days in source regime before transition)

| FROM        | TO         | Count | Median days | Mean days |
|-------------|------------|-------|-------------|----------|
| CRASH       | RECOVERY   | 62    | 14.2d       | 13.9d    |
| CRASH       | CORRECTION | 44    | 2.4d        | 3.6d     |
| CORRECTION  | CRASH      | 30    | 0.8d        | 2.5d     |
| CORRECTION  | RECOVERY   | 14    | 4.3d        | 5.6d     |
| RECOVERY    | CRASH      | 49    | 4.0d        | 4.2d     |
| RECOVERY    | BULL       | 10    | **7.0d**    | 7.0d     |
| RECOVERY    | RANGE      | 17    | **7.0d**    | 7.0d     |
| RANGE       | BULL       | 85    | 0.9d        | 1.7d     |
| RANGE       | CRASH      | 23    | 1.7d        | 2.1d     |
| BULL        | RANGE      | 90    | 1.0d        | 1.8d     |
| BULL        | CRASH      | 4     | 2.6d        | 2.6d     |

---

## Key Findings

### 1. RECOVERY is a High-Failure Staging State

Only **13.2%** of RECOVERY periods commit to BULL. The majority (64.5%)
re-enter CRASH. This is the largest source of false "recovery" signals.

Important: all RECOVERY→BULL and RECOVERY→RANGE transitions occur at
**exactly 7.0 days** (median and mean identical). This is not a
coincidence — it reflects a hard `recovery_bars = 168` (168h = 7d)
constraint in the supervisor. The classifier holds the RECOVERY label
for exactly 7 days before promoting to BULL/RANGE or re-pausing.

### 2. The Real BULL Path is RANGE→BULL

The dominant path to a committed BULL is:
`CRASH → RECOVERY (7d) → RANGE → BULL`

RANGE→BULL has a **78.7% probability** and fires within a median of
**0.9 days** of RANGE onset. The live bot should monitor for RANGE onset
as the actionable signal, not wait for full BULL confirmation.

### 3. 10-Day Confidence Window

Within any 10-day window from regime onset:
- RANGE → BULL: **76.8%** probability
- RECOVERY → CRASH: **64.5%** probability (dominant failure mode)
- CORRECTION → CRASH: **65.9%** probability
- BULL → RANGE: **95.7%** probability (BULLs are short-lived)

### 4. BULL Segments Are Short

Median BULL duration is only **1.0 day** before transitioning to RANGE.
This means the 21-trade backtest with median hold of 14.2 days is
spanning multiple BULL→RANGE→BULL cycles within a single logical
"uptrend". The backtest correctly holds through RANGE periods (since
the exit trigger is CRASH, not RANGE).

### 5. Current Trade (Apr 14, 2026)

BULL committed at $2,362 on 2026-04-14. Cycle trough was ~−58% in
April 2026. This classifies as **DEEP** — the highest-confidence
bull class historically.

---

## Bot Parameter Recommendations

### Consider Earlier Entry at RANGE Onset

Given RANGE→BULL fires in median 0.9d with 78.7% probability, and
BULL segments themselves last only ~1d before cycling through RANGE
again, entering at RANGE onset (rather than waiting for BULL commit)
would capture more of the move at the cost of a 21.3% RANGE→CRASH
false positive rate.

Suggested filter to reduce RANGE→CRASH false positives:
- Require EMA fast > EMA slow (trend confirmation) at RANGE onset
- Require cycle trough < −15% (not a flat-market RANGE)

### RECOVERY Confirmation Filter

Do not act on RECOVERY alone. Wait for RANGE onset. The 7-day
`recovery_bars` hard constraint means every RECOVERY exits at exactly
day 7 — but 64.5% re-crash before getting there. Only 35.5% reach
the RANGE/BULL promotion.

### Stop-Loss Calibration

With median BULL duration of 1.0d and max BULL→CRASH at 2.6d median,
a 15% hard stop from peak is appropriate for deep cycles. Consider
tightening to 10% for SHALLOW entries where the risk/reward is weaker.
