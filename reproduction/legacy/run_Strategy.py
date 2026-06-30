"""Backward-compatible confidence weighting strategy search entry point."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from protcross.experiments.strategy_search import main


if __name__ == "__main__":
    raise SystemExit(main())
