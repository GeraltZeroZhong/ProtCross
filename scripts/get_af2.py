"""Backward-compatible AlphaFold download entry point.

Prefer ``protcross-download-af2`` after installing the package.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from evopoint_da.cli.download_af2 import main


if __name__ == "__main__":
    raise SystemExit(main())

