"""Backward-compatible preprocessing entry point.

Prefer ``protcross preprocess`` after installing the package.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from protcross.cli.preprocess import main


if __name__ == "__main__":
    raise SystemExit(main())
