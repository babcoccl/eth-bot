#!/usr/bin/env python3
import time
import os
import sys
from eth_macrosupervisor_v31 import MacroSupervisor

def main():
    print("ETH Bot E2E Simulation (Orchestrator Scaling)")
    print("==============================================")
    print("This script simulates a live trading loop.")
    print("Use 'eth_llm_advisor_mock.py' in another terminal to change conviction.")
    print("-" * 60)

    sup = MacroSupervisor()
    base_capital = 1000.0

    # Simulate bot loading
    from eth_trendbot_v1 import TrendBot
    bot = TrendBot(symbol="ETH-USD")
    if bot.load_from_disk():
        print(f"[RECOVERY] Resumed position: {bot._position.qty} ETH")
    else:
        print("[INIT] No existing state found. Starting fresh.")

    try:
        while True:
            # The supervisor automatically re-loads advisor_state.json in get_capital_scale()
            scale = sup.get_capital_scale()
            current_capital = base_capital * scale
            
            # Print status line
            sys.stdout.write(f"\rConviction: {sup.conviction_score:.2f} | Scaling: {scale:.2f} | Active Capital: ${current_capital:>7.2f} | Notes: {sup.advisor_notes[:30]}...")
            sys.stdout.flush()
            
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")

if __name__ == "__main__":
    main()
