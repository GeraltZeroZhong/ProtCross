"""Backward-compatible label mapping entry point.

Prefer ``protcross-map-labels`` after installing the package.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from evopoint_da.cli.map_labels import main


if __name__ == "__main__":
    raise SystemExit(main())

