import json
import pandas as pd
import subprocess
import sys
import os

BASELINE_FILE = "regression_baseline.json"
SUMMARY_FILE  = "integration_v1_summary.csv"
TOLERANCE     = 0.01 # 1%

def run_harness():
    print("Running eth_test_harness_integration_v1.py...")
    # Using subprocess to run the harness
    result = subprocess.run([sys.executable, "eth_test_harness_integration_v1.py"], capture_output=True, text=True)
    if result.returncode != 0:
        print("Harness failed to run!")
        print(result.stderr)
        return False
    return True

def check_regressions():
    if not os.path.exists(BASELINE_FILE):
        print(f"Error: {BASELINE_FILE} not found.")
        return False
    
    if not os.path.exists(SUMMARY_FILE):
        print(f"Error: {SUMMARY_FILE} not found. Did the harness run successfully?")
        return False

    with open(BASELINE_FILE, "r") as f:
        baseline = json.load(f)
    
    df = pd.read_csv(SUMMARY_FILE)
    
    print("\nRegression Check Results:")
    print("=========================")
    
    all_pass = True
    
    for cycle_label, expected in baseline["cycles"].items():
        row = df[df["label"] == cycle_label]
        if row.empty:
            print(f"[FAIL] {cycle_label}: Cycle missing from results!")
            all_pass = False
            continue
            
        actual_pnl = float(row.iloc[0]["combined_pnl"])
        expected_pnl = expected["combined_pnl"]
        
        diff = abs(actual_pnl - expected_pnl)
        # Handle zero division if expected_pnl is 0
        if abs(expected_pnl) > 0.01:
            pct_diff = diff / abs(expected_pnl)
        else:
            pct_diff = diff
            
        if pct_diff > TOLERANCE:
            print(f"[FAIL] {cycle_label}: PnL=${actual_pnl:+.2f} (Expected=${expected_pnl:+.2f}, Diff={pct_diff:.2%})")
            all_pass = False
        else:
            print(f"[PASS] {cycle_label}: PnL=${actual_pnl:+.2f} (Diff={pct_diff:.2%})")

    # Check Total PnL
    total_actual = df["combined_pnl"].sum()
    total_expected = baseline["baseline_total_pnl"]
    total_diff = abs(total_actual - total_expected) / total_expected
    
    if total_diff > TOLERANCE:
        print(f"\n[FAIL] TOTAL PnL: ${total_actual:+.2f} (Expected=${total_expected:+.2f}, Diff={total_diff:.2%})")
        all_pass = False
    else:
        print(f"\n[PASS] TOTAL PnL: ${total_actual:+.2f} (Diff={total_diff:.2%})")

    return all_pass

if __name__ == "__main__":
    # If harness fails, exit 1
    if not run_harness():
        sys.exit(1)
        
    if check_regressions():
        print("\nALL REGRESSIONS PASSED.")
        sys.exit(0)
    else:
        print("\nREGRESSION DETECTED!")
        sys.exit(1)
