"""Data parsing and loading utilities.

Heavy training/preprocessing dependencies are imported lazily so lightweight
CLI help paths stay usable in minimal environments.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "AF2DownloadConfig": ("protcross.data.af2", "AF2DownloadConfig"),
    "AF2Downloader": ("protcross.data.af2", "AF2Downloader"),
    "download_af2_structures": ("protcross.data.af2", "download_af2_structures"),
    "LabelMappingConfig": ("protcross.data.label_mapping", "LabelMappingConfig"),
    "map_labels": ("protcross.data.label_mapping", "map_labels"),
    "PCAReducer": ("protcross.data.pca", "PCAReducer"),
    "DEFAULT_IGNORED_HETATM_RESNAMES": ("protcross.data.structure", "DEFAULT_IGNORED_HETATM_RESNAMES"),
    "MAX_ESM_RESIDUES": ("protcross.data.structure", "MAX_ESM_RESIDUES"),
    "STANDARD_AA": ("protcross.data.structure", "STANDARD_AA"),
    "WATER_RESNAMES": ("protcross.data.structure", "WATER_RESNAMES"),
    "StructureParser": ("protcross.data.structure", "StructureParser"),
    "truncate_parsed_structure": ("protcross.data.structure", "truncate_parsed_structure"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
