"""Backward-compatible asset setup entry point.

Prefer ``protcross setup-assets`` or ``protcross-setup-assets`` after installing
the package.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from evopoint_da.cli.setup_assets import main


if __name__ == "__main__":
    raise SystemExit(main())

