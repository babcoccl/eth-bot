#!/usr/bin/env python3
import json
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Mock LLM Advisor for ETH Bot")
    parser.add_argument("--conviction", type=float, default=1.0, help="Conviction score (0.0 to 1.0)")
    parser.add_argument("--notes", type=str, default="Mocked sentiment analysis.", help="Advisor notes")
    args = parser.parse_args()

    state = {
        "conviction": args.conviction,
        "notes": args.notes
    }

    _dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(_dir, "advisor_state.json")

    with open(path, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Advisor state updated: conviction={args.conviction}")
    print(f"Notes: {args.notes}")

if __name__ == "__main__":
    main()
