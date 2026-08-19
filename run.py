"""
run.py — the single entry point for the DTW Market Regime Classifier.

Easiest way to run the project:

    # FAST pipeline (recommended, ~15 minutes)
    python run.py

    # ORIGINAL pipeline (the slow, first implementation — many hours)
    python run.py --original

    # Dashboards
    streamlit run dashboard_fast.py        # fast (~15 min)
    streamlit run dashboard_original.py    # original (hours)

    # Tests
    pytest -v
"""

import argparse
import os
import sys

# Make the project root importable so `from src...` / `from src_fast...` work
# no matter where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="Run the DTW Market Regime Classifier backtest."
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="run the ORIGINAL (very slow) pipeline from src/ instead of the fast one",
    )
    args = parser.parse_args()

    if args.original:
        print("=" * 60)
        print("ORIGINAL pipeline (src/) — this can take MANY HOURS.")
        print("=" * 60)
        from src.backtester import main as run
    else:
        print("=" * 60)
        print("FAST pipeline (src_fast/) — about 15 minutes.")
        print("=" * 60)
        from src_fast.backtester import main as run

    run()


if __name__ == "__main__":
    main()
