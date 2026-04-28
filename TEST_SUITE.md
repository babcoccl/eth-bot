# ETH Bot Regression & Test Suite

This document serves as the primary index for all testing harnesses in the repository. It is designed to be consumed by both human developers and AI agents (e.g., Jules).

## 1. Regression Testing (Gold Standard)

The primary gate for any code change is the **Integration Regression Suite**.

| Script | Purpose |
| :--- | :--- |
| `check_regressions.py` | Runs the full integration test and compares results against `regression_baseline.json`. |
| `regression_baseline.json` | Stores the "accepted" PnL and trade counts for the current bot network. |

**Jules Workflow**: 
1. Run `python check_regressions.py`.
2. If `PASS`, the code change is performance-neutral or better.
3. If `FAIL`, analyze the cycle that degraded to determine if the change is actually an improvement or an accidental regression.

## 2. Integrated System Harness

| Script | Purpose |
| :--- | :--- |
| `eth_test_harness_integration_v1.py` | Simulates the MacroSupervisor orchestrating TrendBot, RangeBot, and RecoveryBot across historical cycles. |

**Output**: Generates `integration_v1_summary.csv` and prints a hypothesis evaluation table.

## 3. Isolated Bot Harnesses

Use these to tune individual bot parameters in isolation.

| Harness | Target Bot |
| :--- | :--- |
| `eth_test_harness_trendbot_v1.py` | TrendBot (Momentum) |
| `eth_test_harness_rangebot.py` | RangeBot (Grid) |
| `eth_test_harness_recoverybot.py` | RecoveryBot (DCB Short) |
| `eth_test_harness_correctionbot_v1.py` | CorrectionBot (Crash Deep Dip) |

## 4. Maintenance & Tuning

| Script | Purpose |
| :--- | :--- |
| `scratch/sweep_recoverybot.py` | Runs a grid search to optimize Fibonacci and volume parameters for the RecoveryBot. |
| `eth_regime_audit.py` | Audits the MacroSupervisor regime classifications against the price database. |

---
*Last Updated: 2026-04-28*
