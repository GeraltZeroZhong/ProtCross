"""Backward-compatible multi-seed benchmark entry point."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from protcross.experiments.multiseed_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
