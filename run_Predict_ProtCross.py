"""Backward-compatible single-structure prediction entry point.

Prefer ``protcross-predict`` after installing the package. The legacy argument
names from 0.1.0 are still accepted here.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.cli.predict import main


if __name__ == "__main__":
    raise SystemExit(main())

