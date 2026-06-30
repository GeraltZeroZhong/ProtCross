"""Legacy label mapping entry point.

This file is kept for old command references. Use
``reproduction/legacy/map_labels.py`` or ``protcross map-labels`` for the
maintained implementation.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../src"))

from protcross.cli.map_labels import main


if __name__ == "__main__":
    raise SystemExit(main())
