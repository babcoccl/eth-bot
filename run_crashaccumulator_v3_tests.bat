@echo off
echo CrashAccumulator v3 Tests - Depth-Adaptive + FREEZE mode
echo ===========================================================
python eth_test_harness_crashaccumulator_v3.py --capital 400 --preset accumulator_v3 --max-hold-days 365
echo.
echo Also run conservative for regression:
python eth_test_harness_crashaccumulator_v3.py --capital 400 --preset accumulator_v2_conservative --max-hold-days 365
pause
