"""Deprecated compatibility alias for the renamed :mod:`protcross` package."""

from __future__ import annotations

import importlib
import warnings


warnings.warn(
    "evopoint_da has been renamed to protcross and will be removed in a future release. "
    "Import protcross instead.",
    DeprecationWarning,
    stacklevel=2,
)

_protcross = importlib.import_module("protcross")

__version__ = _protcross.__version__


def __getattr__(name: str):
    return getattr(_protcross, name)


__all__ = ["__version__"]
