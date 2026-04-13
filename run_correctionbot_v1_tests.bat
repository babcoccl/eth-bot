@echo off
echo CorrectionBot v1 Tests
echo ==================================================

echo.
echo Running standard preset...
python eth_test_harness_correctionbot_v1.py --preset correction_v1 --workers 4

echo.
echo Running aggressive preset (regression)...
python eth_test_harness_correctionbot_v1.py --preset correction_v1_aggressive --workers 4

echo.
echo Running conservative preset (regression)...
python eth_test_harness_correctionbot_v1.py --preset correction_v1_conservative --workers 4

pause
