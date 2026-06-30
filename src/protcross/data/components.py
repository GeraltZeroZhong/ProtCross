"""Backward-compatible imports for 0.1.0 scripts.

New code should import from ``protcross.data.esm``, ``protcross.data.pca``
and ``protcross.data.structure`` directly.
"""

from .esm import ESMFeatureExtractor
from .pca import PCAReducer
from .structure import MAX_ESM_RESIDUES, STANDARD_AA, StructureParser, truncate_parsed_structure

__all__ = [
    "ESMFeatureExtractor",
    "MAX_ESM_RESIDUES",
    "PCAReducer",
    "STANDARD_AA",
    "StructureParser",
    "truncate_parsed_structure",
]

