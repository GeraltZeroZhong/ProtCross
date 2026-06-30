"""Helpers for the deprecated evopoint_da compatibility package."""

from __future__ import annotations

import importlib
import sys
import warnings


def alias_module(old_name: str, new_name: str):
    warnings.warn(
        f"{old_name} has been renamed to {new_name}; import {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )
    module = importlib.import_module(new_name)
    sys.modules[old_name] = module
    return module


def alias_children(package_name: str, new_package_name: str, children: tuple[str, ...]) -> None:
    for child in children:
        sys.modules[f"{package_name}.{child}"] = importlib.import_module(f"{new_package_name}.{child}")
