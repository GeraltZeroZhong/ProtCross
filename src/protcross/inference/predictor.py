"""Lightweight prediction API for applying ProtCross to one structure."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version as distribution_version
from dataclasses import dataclass, field
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import time
from typing import Optional
import uuid

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from protcross.assets import (
    PredictorAssets,
    build_prediction_asset_metadata,
    require_esm_license_acceptance,
    resolve_prediction_assets,
)
from protcross.data import (
    MAX_ESM_RESIDUES,
    PCAReducer,
    StructureParser,
    parsed_structure_long_chunks,
    parsed_structure_sequence_chunks,
    truncate_parsed_structure_by_chain,
)
from protcross.data.esm import ESMFeatureExtractor
from protcross.models import EvoPointDALitModule

from .pdb import write_bfactor_pdb


@dataclass
class _PreparedPrediction:
    input_pdb: Path
    parsed: dict
    input_metadata: dict[str, object]
    inspection: dict[str, object] | None
    warnings: list[str]


@dataclass(frozen=True)
class ResiduePrediction:
    residue_id: str
    probability: float
    is_binding: bool

    @property
    def score(self) -> float:
        """Canonical uncalibrated model score; ``probability`` is a compatibility name."""
        return self.probability


@dataclass
class PredictionResult:
    input_pdb: Path
    residue_ids: list[str]
    probabilities: np.ndarray
    threshold: float = 0.5
    truncated: bool = False
    original_length: int | None = None
    ca_coords: np.ndarray | None = None
    cluster_cutoff: float = 8.0
    residue_metadata: list[dict[str, object]] | None = None
    asset_version: str | None = None
    asset_metadata: dict[str, object] | None = None
    device: str | None = None
    max_len: int | None = None
    output_files: dict[str, str] | None = None
    output_format_warning: str | None = None
    unscored_bfactor_policy: str = "zero"
    elapsed_seconds: float | None = None
    warnings: list[str] | None = None
    structure_summary: dict[str, object] | None = None
    input_metadata: dict[str, object] | None = None
    runtime_metadata: dict[str, str] | None = None
    _records_cache: list[dict[str, str | int | float | None]] | None = field(default=None, init=False, repr=False)
    _cluster_indices_cache: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    _cluster_id_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _pocket_dict_cache: dict | None = field(default=None, init=False, repr=False)
    _summary_dict_cache: dict | None = field(default=None, init=False, repr=False)

    def __setattr__(self, name: str, value) -> None:
        if name == "threshold":
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("Prediction threshold must be finite and in [0, 1].")
        elif name == "cluster_cutoff":
            value = float(value)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Pocket cluster cutoff must be finite and greater than 0.")
        object.__setattr__(self, name, value)
        if not name.startswith("_") and "_records_cache" in self.__dict__:
            object.__setattr__(self, "_records_cache", None)
            object.__setattr__(self, "_cluster_indices_cache", None)
            object.__setattr__(self, "_cluster_id_cache", None)
            object.__setattr__(self, "_pocket_dict_cache", None)
            object.__setattr__(self, "_summary_dict_cache", None)

    def __post_init__(self) -> None:
        self.input_pdb = Path(self.input_pdb)
        self.residue_ids = [str(residue_id) for residue_id in self.residue_ids]
        self.asset_metadata = deepcopy(self.asset_metadata) if self.asset_metadata is not None else None
        self.output_files = dict(self.output_files) if self.output_files is not None else None
        self.structure_summary = (
            deepcopy(self.structure_summary) if self.structure_summary is not None else None
        )
        self.input_metadata = deepcopy(self.input_metadata) if self.input_metadata is not None else None
        self.runtime_metadata = dict(self.runtime_metadata) if self.runtime_metadata is not None else None
        self.probabilities = np.array(self.probabilities, dtype=float, copy=True)
        if self.probabilities.ndim != 1:
            raise ValueError("Prediction probabilities must be a one-dimensional array.")
        if not np.all(np.isfinite(self.probabilities)):
            raise ValueError("Prediction probabilities must be finite.")
        if len(self.probabilities) and (
            float(self.probabilities.min()) < 0.0 or float(self.probabilities.max()) > 1.0
        ):
            raise ValueError("Prediction probabilities must be in the [0, 1] range.")
        if len(self.residue_ids) != len(self.probabilities):
            raise ValueError(
                "Prediction residue_ids and probabilities must have the same length; "
                f"got {len(self.residue_ids)} and {len(self.probabilities)}."
            )
        if self.unscored_bfactor_policy not in {"keep", "zero"}:
            raise ValueError("unscored_bfactor_policy must be 'keep' or 'zero'.")
        if self.ca_coords is not None:
            coords = np.array(self.ca_coords, dtype=float, copy=True)
            if coords.shape != (len(self.residue_ids), 3):
                raise ValueError(
                    "Original CA coordinates must have shape "
                    f"({len(self.residue_ids)}, 3); got {coords.shape}."
                )
            if not np.all(np.isfinite(coords)):
                raise ValueError("Original CA coordinates must be finite.")
            coords.setflags(write=False)
            self.ca_coords = coords
        if self.residue_metadata is None:
            self.residue_metadata = [
                self._default_metadata(residue_id, index) for index, residue_id in enumerate(self.residue_ids)
            ]
        if len(self.residue_metadata) != len(self.residue_ids):
            raise ValueError(
                "Prediction residue_metadata and residue_ids must have the same length; "
                f"got {len(self.residue_metadata)} and {len(self.residue_ids)}."
            )
        if self.warnings is None:
            self.warnings = []
        self.warnings = [str(warning) for warning in self.warnings if str(warning)]
        self.residue_metadata = [
            self._normalize_metadata(metadata, residue_id, index)
            for index, (metadata, residue_id) in enumerate(zip(self.residue_metadata, self.residue_ids))
        ]
        self.probabilities.setflags(write=False)

    @property
    def binding_residues(self) -> list[ResiduePrediction]:
        return [
            ResiduePrediction(residue_id, float(probability), bool(probability > self.threshold))
            for residue_id, probability in zip(self.residue_ids, self.probabilities)
            if probability > self.threshold
        ]

    @property
    def scores(self) -> np.ndarray:
        """Canonical view of uncalibrated scores; ``probabilities`` remains API-compatible."""
        return self.probabilities

    def write_pdb(
        self,
        output_pdb: str | Path,
        *,
        missing_value: float | None = None,
        unscored_bfactor_policy: str | None = None,
    ) -> None:
        policy = unscored_bfactor_policy or self.unscored_bfactor_policy
        if policy not in {"keep", "zero"}:
            raise ValueError("unscored_bfactor_policy must be 'keep' or 'zero'.")
        if missing_value is None and policy == "zero":
            missing_value = 0.0
        final_path = Path(output_pdb).expanduser()
        if self.input_metadata:
            _assert_input_file_unchanged(self.input_pdb, self.input_metadata)
        self._validate_single_output_path(final_path)
        input_is_cif = self.input_pdb.suffix.lower() in {".cif", ".mmcif"}
        output_is_cif = final_path.suffix.lower() in {".cif", ".mmcif"}
        if input_is_cif != output_is_cif:
            raise ValueError(
                "Annotated structure output must use the same PDB or mmCIF format as the input; "
                "use a dedicated structure converter separately."
            )
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._temporary_output_path(final_path)
        try:
            write_bfactor_pdb(
                self.input_pdb,
                temporary_path,
                self.residue_ids,
                self.probabilities,
                missing_value=missing_value,
                residue_metadata=self.residue_metadata,
            )
            temporary_path.replace(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def write_scores_tsv(self, output_tsv: str | Path, *, schema: str = "extended") -> None:
        final_path = Path(output_tsv).expanduser()
        self._validate_single_output_path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._temporary_output_path(final_path)
        try:
            self._write_scores_tsv_file(temporary_path, schema=schema)
            temporary_path.replace(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_scores_tsv_file(self, output_tsv: Path, *, schema: str) -> None:
        records = self.to_records()
        with output_tsv.open("w", encoding="utf-8") as file:
            if schema == "legacy":
                file.write("residue_id\tchain_id\tresidue_number\tprobability\tis_binding\n")
                for record in records:
                    file.write(
                        f"{record['residue_id']}\t{record['chain_id']}\t{record['residue_number']}\t"
                        f"{record['probability']:.6f}\t{record['is_binding']}\n"
                    )
                return
            if schema != "extended":
                raise ValueError("scores TSV schema must be 'extended' or 'legacy'.")
            file.write(
                "residue_id\tresidue_key\tresidue_id_namespace\tmodel_id\tchain_id\tauth_asym_id\tlabel_asym_id\t"
                "residue_number\tauth_seq_id\tlabel_seq_id\tinsertion_code\tresname\tone_letter_code\t"
                "input_bfactor\tmodel_score\tprobability\tis_binding\tx\ty\tz\tcluster_id\tis_scored\t"
                "rank_global\trank_within_chain\n"
            )
            for record in records:
                cluster_id = record["cluster_id"] if record["cluster_id"] is not None else ""
                file.write(
                    f"{record['residue_id']}\t{record['residue_key']}\t"
                    f"{self._format_optional_value(record['residue_id_namespace'])}\t"
                    f"{self._format_optional_value(record['model_id'])}\t"
                    f"{self._format_optional_value(record['chain_id'])}\t"
                    f"{self._format_optional_value(record['auth_asym_id'])}\t"
                    f"{self._format_optional_value(record['label_asym_id'])}\t"
                    f"{self._format_optional_value(record['residue_number'])}\t"
                    f"{self._format_optional_value(record['auth_seq_id'])}\t"
                    f"{self._format_optional_value(record['label_seq_id'])}\t"
                    f"{self._format_optional_value(record['insertion_code'])}\t"
                    f"{self._format_optional_value(record['resname'])}\t"
                    f"{self._format_optional_value(record['one_letter_code'])}\t"
                    f"{self._format_optional_float(record['input_bfactor'])}\t"
                    f"{record['score']:.6f}\t{record['probability']:.6f}\t{record['is_binding']}\t"
                    f"{self._format_optional_float(record['x'])}\t"
                    f"{self._format_optional_float(record['y'])}\t"
                    f"{self._format_optional_float(record['z'])}\t{cluster_id}\t"
                    f"{record['is_scored']}\t{record['rank_global']}\t{record['rank_within_chain']}\n"
                )

    def write_pocket_json(self, output_json: str | Path) -> None:
        self._write_json_atomic(output_json, self.to_pocket_dict())

    def write_summary_json(self, output_json: str | Path) -> None:
        self._write_json_atomic(output_json, self.to_summary_dict())

    @staticmethod
    def _temporary_output_path(final_path: Path) -> Path:
        return final_path.with_name(
            f".{final_path.stem}.{uuid.uuid4().hex}.part{final_path.suffix}"
        )

    def _write_json_atomic(self, output_json: str | Path, payload: dict) -> None:
        final_path = Path(output_json).expanduser()
        self._validate_single_output_path(final_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self._temporary_output_path(final_path)
        try:
            temporary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
            temporary_path.replace(final_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _validate_single_output_path(self, path: Path) -> None:
        if path.is_dir():
            raise IsADirectoryError(f"Output path is a directory, not a file: {path}.")
        if path.resolve(strict=False) == self.input_pdb.expanduser().resolve(strict=False):
            raise ValueError(f"Output path must not overwrite the input structure: {path}.")

    def to_records(self) -> list[dict[str, str | int | float | None]]:
        if self._records_cache is not None:
            return deepcopy(self._records_cache)
        records = []
        coords = self._coordinate_array(required=False)
        cluster_ids = self._cluster_id_by_index() if coords is not None else np.zeros(len(self.residue_ids), dtype=int)
        global_ranks = self._global_ranks()
        chain_ranks = self._chain_ranks()
        for index, (residue_id, probability) in enumerate(zip(self.residue_ids, self.probabilities)):
            metadata = self.residue_metadata[index]
            coord = coords[index] if coords is not None else (None, None, None)
            records.append(
                {
                    "residue_id": residue_id,
                    "residue_key": metadata["residue_key"],
                    "residue_id_namespace": metadata.get("residue_id_namespace"),
                    "model_id": metadata["model_id"],
                    "chain_id": metadata["chain_id"],
                    "auth_asym_id": metadata.get("auth_asym_id"),
                    "label_asym_id": metadata.get("label_asym_id"),
                    "residue_number": metadata["residue_number"],
                    "auth_seq_id": metadata.get("auth_seq_id"),
                    "label_seq_id": metadata.get("label_seq_id"),
                    "insertion_code": metadata.get("insertion_code", ""),
                    "resname": metadata.get("resname", ""),
                    "one_letter_code": metadata.get("one_letter_code", ""),
                    "input_bfactor": metadata.get("input_bfactor"),
                    "score": float(probability),
                    "probability": float(probability),
                    "is_binding": int(probability > self.threshold),
                    "x": None if coord[0] is None else float(coord[0]),
                    "y": None if coord[1] is None else float(coord[1]),
                    "z": None if coord[2] is None else float(coord[2]),
                    "cluster_id": int(cluster_ids[index]) if cluster_ids[index] > 0 else None,
                    "is_scored": 1,
                    "rank_global": int(global_ranks[index]),
                    "rank_within_chain": int(chain_ranks[index]),
                }
            )
        self._records_cache = records
        return deepcopy(records)

    def to_pocket_dict(self) -> dict:
        if self._pocket_dict_cache is not None:
            return deepcopy(self._pocket_dict_cache)
        coords = self._coordinate_array(required=True)
        selected_indices = self._selected_indices()
        clusters = self._cluster_indices()
        records = self.to_records()
        payload = {
            "schema_version": "protcross-pocket-v2",
            "protcross_version": self._protcross_version(),
            "asset_version": self.asset_version,
            "assets": self.asset_metadata or {},
            "input_structure": str(self.input_pdb),
            "input_file": self.input_metadata or {},
            "runtime": self.runtime_metadata or {},
            "threshold": float(self.threshold),
            "threshold_operator": ">",
            "score_type": "softmax_class_score",
            "score_calibrated": False,
            "score_interpretation": (
                "Relative ProtCross binding-site model score. It is not a calibrated probability "
                "or a binding-affinity estimate."
            ),
            "compatibility_aliases": {"probability": "score"},
            "geometry_backend": "deterministic-pytorch",
            "determinism_scope": (
                "canonical residue ordering and deterministic neighbor selection; "
                "floating-point reductions remain device and kernel dependent"
            ),
            "scoring_procedure_version": "protcross-0.2.2-deterministic-pytorch",
            "residue_ordering": "canonical_model_chain_polymer_order",
            "rigid_rotation_invariant": False,
            "coordinate_frame": "input_structure",
            "coordinate_units": "angstrom",
            "atom_basis": "CA",
            "center_type": "score_weighted_ca_centroid",
            "center_interpretation": (
                "Centroid of predicted residue CA coordinates; not a ligand centroid, cavity center, "
                "surface-accessible point, or validated docking-box center."
            ),
            "cluster_method": "single_linkage_connected_components_on_selected_ca_atoms",
            "cluster_cutoff": float(self.cluster_cutoff),
            "cluster_rank_basis": "residue_count_then_score_mean_then_score_max",
            "truncated": bool(self.truncated),
            "original_residue_count": self.original_residue_count,
            "residues_scored": len(self.residue_ids),
            "selected_residue_count": int(len(selected_indices)),
            "chains_analyzed": self.chains_analyzed,
            "residue_id_namespaces": self.residue_id_namespaces,
            "warnings": self.warnings,
            "structure_summary": self.structure_summary or {},
            "aggregate_pocket": (
                self._pocket_from_indices(selected_indices, coords, records=records)
                if len(selected_indices)
                else None
            ),
            "clustered_pockets": [
                self._pocket_from_indices(indices, coords, cluster_id=cluster_id, records=records)
                for cluster_id, indices in enumerate(clusters, start=1)
            ],
        }
        self._pocket_dict_cache = payload
        return deepcopy(payload)

    def to_summary_dict(self) -> dict:
        if self._summary_dict_cache is not None:
            return deepcopy(self._summary_dict_cache)
        pockets = (
            self.to_pocket_dict()
            if self.ca_coords is not None
            else {"aggregate_pocket": None, "clustered_pockets": []}
        )
        aggregate_pocket = pockets["aggregate_pocket"]
        top_cluster = pockets["clustered_pockets"][0] if pockets["clustered_pockets"] else None
        top_residues = [
            {
                "residue_id": record["residue_id"],
                "residue_key": record["residue_key"],
                "chain_id": record["chain_id"],
                "residue_number": record["residue_number"],
                "insertion_code": record["insertion_code"],
                "score": record["score"],
                "probability": record["probability"],
                "rank_global": record["rank_global"],
            }
            for record in sorted(self.to_records(), key=lambda record: int(record["rank_global"]))[:10]
        ]
        payload = {
            "schema_version": "protcross-summary-v2",
            "protcross_version": self._protcross_version(),
            "asset_version": self.asset_version,
            "assets": self.asset_metadata or {},
            "input_structure": str(self.input_pdb),
            "input_file": self.input_metadata or {},
            "runtime": self.runtime_metadata or {},
            "device": self.device,
            "threshold": float(self.threshold),
            "threshold_operator": ">",
            "score_type": "softmax_class_score",
            "score_calibrated": False,
            "score_interpretation": (
                "Relative ProtCross binding-site model score; not a calibrated probability or binding affinity."
            ),
            "compatibility_aliases": {
                "probability": "score",
                "probability_min": "score_min",
                "probability_max": "score_max",
                "probability_mean": "score_mean",
            },
            "geometry_backend": "deterministic-pytorch",
            "determinism_scope": (
                "canonical residue ordering and deterministic neighbor selection; "
                "floating-point reductions remain device and kernel dependent"
            ),
            "scoring_procedure_version": "protcross-0.2.2-deterministic-pytorch",
            "residue_ordering": "canonical_model_chain_polymer_order",
            "rigid_rotation_invariant": False,
            "threshold_origin": "user decision threshold; 0.5 is a neutral default, not an independently validated optimum",
            "cluster_cutoff": float(self.cluster_cutoff),
            "chains_analyzed": self.chains_analyzed,
            "residue_id_namespaces": self.residue_id_namespaces,
            "residues_scored": len(self.residue_ids),
            "original_residue_count": self.original_residue_count,
            "selected_residue_count": int(len(self._selected_indices())),
            "selected_residue_fraction": (
                float(len(self._selected_indices()) / len(self.residue_ids)) if self.residue_ids else 0.0
            ),
            "score_min": float(self.probabilities.min()) if len(self.probabilities) else None,
            "score_max": float(self.probabilities.max()) if len(self.probabilities) else None,
            "score_mean": float(self.probabilities.mean()) if len(self.probabilities) else None,
            "probability_min": float(self.probabilities.min()) if len(self.probabilities) else None,
            "probability_max": float(self.probabilities.max()) if len(self.probabilities) else None,
            "probability_mean": float(self.probabilities.mean()) if len(self.probabilities) else None,
            "truncated": bool(self.truncated),
            "max_len": self.max_len,
            "unscored_bfactor_policy": self.unscored_bfactor_policy,
            "annotated_structure_score_precision": (
                "PDB B-factor fields round to two decimal places; mmCIF uses up to six significant digits. "
                "Use scores.tsv or JSON for quantitative values."
            ),
            "annotated_structure_preservation": (
                "PDB output preserves input records and patches only B-factor columns on ATOM/HETATM lines; "
                "mmCIF output preserves data categories while reserializing text."
            ),
            "elapsed_seconds": self.elapsed_seconds,
            "aggregate_pocket": (
                {
                    "center": aggregate_pocket["center"],
                    "residue_count": aggregate_pocket["residue_count"],
                    "score_mean": aggregate_pocket["score_mean"],
                    "score_max": aggregate_pocket["score_max"],
                }
                if aggregate_pocket
                else None
            ),
            "top_pocket": (
                {
                    "cluster_id": top_cluster["cluster_id"],
                    "center": top_cluster["center"],
                    "residue_count": top_cluster["residue_count"],
                    "score_mean": top_cluster["score_mean"],
                    "score_max": top_cluster["score_max"],
                }
                if top_cluster
                else None
            ),
            "cluster_count": len(pockets["clustered_pockets"]),
            "top_residues": top_residues,
            "output_files": self.output_files or {},
            "output_format_warning": self.output_format_warning,
            "warnings": self.warnings,
            "structure_summary": self.structure_summary or {},
            "intended_use": "Hypothesis generation and residue ranking for small-molecule-adjacent site analysis.",
            "limitations": [
                "The 0.1.2 release checkpoint has not been established as a calibrated probability model.",
                "Release labels use CA-within-6-Angstrom proximity to eligible hetero/nonstandard residue atoms after a name blacklist; they are not ligand-specific affinity labels.",
                "The supplied coordinate assembly is used as-is; biological assembly operators are not applied.",
                "Multiple selected chains share one geometry graph, and residue clusters can span chain interfaces.",
                "Predicted-residue cluster centroids are not validated docking coordinates.",
            ],
        }
        self._summary_dict_cache = payload
        return deepcopy(payload)

    def format_summary(self, *, max_items: int = 50) -> str:
        summary = self.to_summary_dict()
        hits = self.binding_residues
        lines = [
            f"Input: {self.input_pdb}",
            f"Asset version: {self.asset_version or 'unspecified'}",
            f"Device: {self.device or 'unspecified'}",
            "Chains analyzed: "
            + (
                ", ".join(
                    chain if str(chain).strip() else "<blank>"
                    for chain in summary["chains_analyzed"]
                )
                or "none"
            ),
            f"Residues scored: {summary['residues_scored']} / {summary['original_residue_count']}",
            f"Threshold: {self.threshold:.2f} (binary calls, TSV is_binding, residue selection, clustering, summary)",
            f"Cluster cutoff: {self.cluster_cutoff:.2f} A",
            f"Unscored B-factor policy: {summary['unscored_bfactor_policy']}",
            "Score interpretation: uncalibrated softmax class score (not binding probability or affinity)",
            f"Residues above threshold: {len(hits)}",
        ]
        if summary["probability_min"] is not None:
            lines.append(
                "Model score range: "
                f"{summary['probability_min']:.4f} - {summary['probability_max']:.4f} "
                f"(mean {summary['probability_mean']:.4f})"
            )
        if summary["elapsed_seconds"] is not None:
            lines.append(f"Elapsed: {float(summary['elapsed_seconds']):.2f} s")
        if summary["output_format_warning"]:
            lines.append(f"Output format warning: {summary['output_format_warning']}")
        for warning in summary["warnings"]:
            lines.append(f"WARNING: {warning}")
        if self.truncated:
            lines.append(f"WARNING: input was truncated from {self.original_length} to {len(self.residue_ids)} residues.")
        top_hits = [
            record for record in summary["top_residues"] if float(record["probability"]) > self.threshold
        ][:max_items]
        if top_hits:
            preview = ", ".join(
                f"{record['residue_id']}({float(record['probability']):.2f})" for record in top_hits
            )
            if len(hits) > len(top_hits):
                preview += f", ... ({len(hits)} total)"
            lines.append(f"Top residues: {preview}")
        if summary["top_pocket"]:
            center = summary["top_pocket"]["center"]
            lines.append(f"Top cluster score-weighted CA centroid: {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")
            lines.append(f"Top cluster residues: {summary['top_pocket']['residue_count']}")
            lines.append(f"Predicted-residue clusters: {summary['cluster_count']}")
        if summary["output_files"]:
            for label, path in summary["output_files"].items():
                lines.append(f"Wrote {label}: {path}")
        return "\n".join(lines)

    @property
    def original_residue_count(self) -> int:
        return int(self.original_length or len(self.residue_ids))

    @property
    def chains_analyzed(self) -> list[str]:
        return sorted(
            {
                str(metadata.get("chain_id") or "")
                for metadata in self.residue_metadata
            }
        )

    @property
    def residue_id_namespaces(self) -> list[str]:
        return sorted(
            {
                str(metadata.get("residue_id_namespace", "unknown"))
                for metadata in self.residue_metadata
                if metadata.get("residue_id_namespace")
            }
        )

    def _selected_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.probabilities, dtype=float) > self.threshold)

    def _cluster_id_by_index(self) -> np.ndarray:
        if self._cluster_id_cache is not None:
            return self._cluster_id_cache
        cluster_ids = np.zeros(len(self.residue_ids), dtype=int)
        for cluster_id, indices in enumerate(self._cluster_indices(), start=1):
            cluster_ids[indices] = cluster_id
        self._cluster_id_cache = cluster_ids
        return cluster_ids

    def _cluster_indices(self) -> list[np.ndarray]:
        if self._cluster_indices_cache is not None:
            return self._cluster_indices_cache
        coords = self._coordinate_array(required=True)
        selected = self._selected_indices()
        if len(selected) == 0:
            self._cluster_indices_cache = []
            return self._cluster_indices_cache

        selected_coords = coords[selected]
        visited = np.zeros(len(selected), dtype=bool)
        components: list[np.ndarray] = []

        for start in range(len(selected)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                distances = np.linalg.norm(selected_coords - selected_coords[current], axis=1)
                neighbors = np.flatnonzero((distances <= self.cluster_cutoff) & (~visited))
                for neighbor in neighbors:
                    visited[neighbor] = True
                    stack.append(int(neighbor))
            components.append(selected[np.asarray(component, dtype=int)])

        components.sort(
            key=lambda indices: (
                -len(indices),
                -float(np.mean(self.probabilities[indices])),
                -float(np.max(self.probabilities[indices])),
                int(indices.min()),
            )
        )
        self._cluster_indices_cache = components
        return components

    def _pocket_from_indices(
        self,
        indices: np.ndarray,
        coords: np.ndarray,
        *,
        records: list[dict[str, str | int | float | None]],
        cluster_id: int | None = None,
    ) -> dict:
        pocket_coords = coords[indices]
        pocket_probs = np.asarray(self.probabilities, dtype=float)[indices]
        weight_sum = float(pocket_probs.sum())
        if weight_sum > 0:
            center = np.average(pocket_coords, axis=0, weights=pocket_probs)
        else:
            center = pocket_coords.mean(axis=0)
        center_unweighted = pocket_coords.mean(axis=0)
        bbox_min = pocket_coords.min(axis=0)
        bbox_max = pocket_coords.max(axis=0)
        radius = float(np.max(np.linalg.norm(pocket_coords - center, axis=1)))
        diameter = _max_pairwise_distance(pocket_coords)

        pocket_records = []
        for index in indices:
            coord = coords[int(index)]
            record = dict(records[int(index)])
            pocket_records.append(
                {
                    "residue_id": record["residue_id"],
                    "residue_key": record["residue_key"],
                    "residue_id_namespace": record["residue_id_namespace"],
                    "model_id": record["model_id"],
                    "chain_id": record["chain_id"],
                    "auth_asym_id": record["auth_asym_id"],
                    "label_asym_id": record["label_asym_id"],
                    "residue_number": record["residue_number"],
                    "auth_seq_id": record["auth_seq_id"],
                    "label_seq_id": record["label_seq_id"],
                    "insertion_code": record["insertion_code"],
                    "resname": record["resname"],
                    "one_letter_code": record["one_letter_code"],
                    "input_bfactor": record["input_bfactor"],
                    "score": record["score"],
                    "probability": record["probability"],
                    "coord": [float(coord[0]), float(coord[1]), float(coord[2])],
                }
            )

        pocket = {
            "center": [float(center[0]), float(center[1]), float(center[2])],
            "center_unweighted": [
                float(center_unweighted[0]),
                float(center_unweighted[1]),
                float(center_unweighted[2]),
            ],
            "residue_count": int(len(indices)),
            "score_mean": float(np.mean(pocket_probs)),
            "score_max": float(np.max(pocket_probs)),
            "bbox_min": [float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])],
            "bbox_max": [float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])],
            "radius": radius,
            "max_pairwise_ca_distance": diameter,
            "diffuse_single_linkage_cluster": bool(diameter > max(2.0 * self.cluster_cutoff, 16.0)),
            "residues": pocket_records,
        }
        if cluster_id is not None:
            pocket = {"cluster_id": cluster_id, **pocket}
        return pocket

    def _coordinate_array(self, *, required: bool) -> np.ndarray | None:
        if self.ca_coords is None:
            if required:
                raise ValueError("Original CA coordinates are required for pocket reporting.")
            return None
        coords = np.asarray(self.ca_coords, dtype=float)
        if coords.shape != (len(self.residue_ids), 3):
            raise ValueError(
                "Original CA coordinates must have shape "
                f"({len(self.residue_ids)}, 3); got {coords.shape}."
            )
        return coords

    def _global_ranks(self) -> np.ndarray:
        ranks = np.empty(len(self.probabilities), dtype=int)
        for rank, index in enumerate(np.argsort(-self.probabilities), start=1):
            ranks[int(index)] = rank
        return ranks

    def _chain_ranks(self) -> np.ndarray:
        ranks = np.empty(len(self.probabilities), dtype=int)
        chain_ids = sorted(
            {
                str(metadata.get("chain_id") or "")
                for metadata in self.residue_metadata
            }
        )
        for chain_id in chain_ids:
            indices = [
                index
                for index, metadata in enumerate(self.residue_metadata)
                if str(metadata.get("chain_id") or "") == chain_id
            ]
            for rank, relative_index in enumerate(np.argsort(-self.probabilities[indices]), start=1):
                ranks[indices[int(relative_index)]] = rank
        return ranks

    @staticmethod
    def _format_optional_float(value: float | None) -> str:
        return "" if value is None else f"{value:.6f}"

    @staticmethod
    def _format_optional_value(value: object | None) -> str:
        return "" if value is None else str(value)

    @staticmethod
    def _split_residue_id(residue_id: str) -> tuple[str, str]:
        if "_" not in residue_id:
            return "", residue_id
        chain_id, residue_number = residue_id.split("_", 1)
        return chain_id, residue_number

    @classmethod
    def _default_metadata(cls, residue_id: str, index: int) -> dict[str, object]:
        chain_id, residue_number = cls._split_residue_id(residue_id)
        return {
            "residue_id": residue_id,
            "residue_key": f"legacy:{index}:{residue_id}",
            "residue_id_namespace": "legacy",
            "model_id": "0",
            "chain_id": chain_id,
            "auth_asym_id": chain_id,
            "label_asym_id": None,
            "residue_number": residue_number,
            "auth_seq_id": residue_number,
            "label_seq_id": None,
            "insertion_code": "",
            "resname": "",
            "one_letter_code": "",
            "input_bfactor": None,
        }

    @classmethod
    def _normalize_metadata(cls, metadata: dict[str, object], residue_id: str, index: int) -> dict[str, object]:
        normalized = cls._default_metadata(residue_id, index)
        normalized.update(metadata)
        normalized["residue_id"] = residue_id
        return normalized

    @staticmethod
    def _protcross_version() -> str:
        try:
            from protcross import __version__

            return __version__
        except Exception:
            return "unknown"


class ProtCrossPredictor:
    """Inference-only facade around feature extraction, PCA and model forward."""

    def __init__(
        self,
        ckpt_path: str | Path | None = None,
        esm_weights: str | Path | None = None,
        pca_path: str | Path | None = None,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
        embedding_cache_dir: str | Path | None = None,
        esm_extractor: ESMFeatureExtractor | None = None,
        pca_reducer: PCAReducer | None = None,
        structure_parser: StructureParser | None = None,
        model: EvoPointDALitModule | None = None,
        asset_version: str | None = None,
        asset_metadata: dict[str, object] | None = None,
        accept_esm_license: bool = False,
        feature_pipeline_fingerprint: str | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        if max_len <= 0 or max_len > MAX_ESM_RESIDUES:
            raise ValueError(f"max_len must be between 1 and {MAX_ESM_RESIDUES}.")
        self.max_len = max_len
        if pca_dim <= 0:
            raise ValueError("pca_dim must be greater than 0.")
        self.pca_dim = pca_dim
        self.embedding_cache_dir = Path(embedding_cache_dir).expanduser() if embedding_cache_dir else None
        injected_feature_pipeline = esm_extractor is not None or pca_reducer is not None
        if feature_pipeline_fingerprint is not None:
            feature_pipeline_fingerprint = str(feature_pipeline_fingerprint).strip()
            if not feature_pipeline_fingerprint:
                raise ValueError("feature_pipeline_fingerprint must be a non-empty string when provided.")
        if self.embedding_cache_dir is not None and injected_feature_pipeline and feature_pipeline_fingerprint is None:
            raise ValueError(
                "embedding_cache_dir with an injected ESM extractor or PCA reducer requires "
                "feature_pipeline_fingerprint to prevent cache collisions between custom feature pipelines."
            )
        self.feature_pipeline_fingerprint = feature_pipeline_fingerprint
        self.asset_version = asset_version
        esm_weights_path = self._optional_existing_path(esm_weights, "esm_weights") if esm_weights is not None else None
        pca_path_obj = self._optional_existing_path(pca_path, "pca_path") if pca_path is not None else None
        ckpt_path_obj = self._optional_existing_path(ckpt_path, "ckpt_path") if ckpt_path is not None else None
        self.asset_metadata = deepcopy(asset_metadata) if asset_metadata is not None else None
        if self.asset_metadata is None and all(
            path is not None for path in (ckpt_path_obj, esm_weights_path, pca_path_obj)
        ):
            self.asset_metadata = build_prediction_asset_metadata(
                ckpt_path_obj,
                esm_weights_path,
                pca_path_obj,
                asset_version=asset_version,
                selected_bundle_version=asset_version,
                sources={name: "direct" for name in ("checkpoint", "esm_weights", "pca")},
            )
        self._embedding_cache_asset_identity = self._embedding_cache_asset_identity_for(
            esm_weights_path,
            pca_path_obj,
            self.asset_metadata,
        )
        self.structure_parser = structure_parser or StructureParser()
        if esm_extractor is None:
            require_esm_license_acceptance(accept_esm_license)
        self.esm_extractor = esm_extractor or ESMFeatureExtractor(
            self._require_path(esm_weights_path, "esm_weights"),
            self.device,
        )
        self.pca_reducer = pca_reducer or self._load_pca(self._require_path(pca_path_obj, "pca_path"), pca_dim)
        reducer_dim = getattr(self.pca_reducer, "n_components", None)
        self._expected_feature_dim = int(reducer_dim) if reducer_dim is not None else None
        self.model = model or self._load_model(self._require_path(ckpt_path_obj, "ckpt_path"))

    @classmethod
    def from_files(
        cls,
        ckpt_path: str | Path,
        esm_weights: str | Path,
        pca_path: str | Path,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
        asset_version: str | None = None,
        asset_metadata: dict[str, object] | None = None,
        embedding_cache_dir: str | Path | None = None,
        accept_esm_license: bool = False,
    ) -> "ProtCrossPredictor":
        return cls(
            ckpt_path=ckpt_path,
            esm_weights=esm_weights,
            pca_path=pca_path,
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
            asset_version=asset_version,
            asset_metadata=asset_metadata,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=accept_esm_license,
        )

    @classmethod
    def from_assets(
        cls,
        assets: PredictorAssets,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
        embedding_cache_dir: str | Path | None = None,
        accept_esm_license: bool = False,
    ) -> "ProtCrossPredictor":
        resolved_device = cls._resolve_device(device)
        asset_metadata = build_prediction_asset_metadata(
            assets.checkpoint,
            assets.esm_weights,
            assets.pca,
            asset_version=assets.asset_version,
            selected_bundle_version=assets.asset_version,
            sources={name: "predictor_assets" for name in ("checkpoint", "esm_weights", "pca")},
        )
        return cls.from_files(
            ckpt_path=assets.checkpoint,
            esm_weights=assets.esm_weights,
            pca_path=assets.pca,
            device=resolved_device,
            pca_dim=pca_dim,
            max_len=max_len,
            asset_version=assets.asset_version,
            asset_metadata=asset_metadata,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=accept_esm_license,
        )

    @classmethod
    def from_env(
        cls,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
        embedding_cache_dir: str | Path | None = None,
        accept_esm_license: bool = False,
        trust_unverified_assets: bool = False,
    ) -> "ProtCrossPredictor":
        PredictorAssets.from_env()
        resolved_device = cls._resolve_device(device)
        resolved = resolve_prediction_assets(
            auto_setup_assets=False,
            offline=True,
            accept_esm_license=accept_esm_license,
            require_esm_license_for_use=True,
            trust_unverified_assets=trust_unverified_assets,
        )
        return cls.from_files(
            ckpt_path=resolved.checkpoint,
            esm_weights=resolved.esm_weights,
            pca_path=resolved.pca,
            device=resolved_device,
            pca_dim=pca_dim,
            max_len=max_len,
            asset_version=resolved.asset_version,
            asset_metadata=resolved.asset_metadata,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=True,
        )

    @classmethod
    def from_default_assets(
        cls,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
        asset_version: str = "default",
        embedding_cache_dir: str | Path | None = None,
        accept_esm_license: bool = False,
    ) -> "ProtCrossPredictor":
        resolved_device = cls._resolve_device(device)
        default_assets = PredictorAssets.from_default_dir(asset_version=asset_version)
        resolved = resolve_prediction_assets(
            ckpt_path=default_assets.checkpoint,
            esm_weights=default_assets.esm_weights,
            pca_path=default_assets.pca,
            auto_setup_assets=False,
            asset_version=default_assets.asset_version,
            offline=True,
            accept_esm_license=accept_esm_license,
            require_esm_license_for_use=True,
        )
        return cls.from_files(
            ckpt_path=resolved.checkpoint,
            esm_weights=resolved.esm_weights,
            pca_path=resolved.pca,
            device=resolved_device,
            pca_dim=pca_dim,
            max_len=max_len,
            asset_version=resolved.asset_version,
            asset_metadata=resolved.asset_metadata,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=True,
        )

    def predict(
        self,
        pdb_file: str | Path,
        *,
        chain_id: Optional[str] = None,
        threshold: float = 0.5,
        pocket_cluster_cutoff: float = 8.0,
        output_pdb: str | Path | None = None,
        scores_tsv: str | Path | None = None,
        pocket_json: str | Path | None = None,
        summary_json: str | Path | None = None,
        allow_truncation: bool = False,
        unscored_bfactor_policy: str = "zero",
        structure_inspection: Mapping[str, object] | None = None,
    ) -> PredictionResult:
        _validate_prediction_options(
            threshold=threshold,
            pocket_cluster_cutoff=pocket_cluster_cutoff,
            max_len=self.max_len,
            unscored_bfactor_policy=unscored_bfactor_policy,
        )
        pdb_file = Path(pdb_file).expanduser()
        if not pdb_file.exists():
            raise FileNotFoundError(f"Input structure not found: {pdb_file}")
        output_pdb, scores_tsv, pocket_json, summary_json = (
            _expanded_optional_path(path)
            for path in (output_pdb, scores_tsv, pocket_json, summary_json)
        )
        _validate_output_paths(
            pdb_file,
            output_pdb,
            scores_tsv,
            pocket_json,
            summary_json,
        )
        prepared = self._prepare_prediction(
            pdb_file,
            chain_id=chain_id,
            allow_truncation=allow_truncation,
            structure_inspection=structure_inspection,
        )
        start_time = time.perf_counter()
        features = self._featurize_parsed(prepared.parsed)
        data = Data(
            x=features,
            pos=torch.from_numpy(prepared.parsed["coords"]),
            batch=torch.zeros(len(prepared.parsed["coords"]), dtype=torch.long),
        ).to(self.device)

        probabilities = self._infer(data)
        elapsed_seconds = time.perf_counter() - start_time
        return self._finish_prediction(
            prepared,
            probabilities,
            threshold=threshold,
            pocket_cluster_cutoff=pocket_cluster_cutoff,
            output_pdb=output_pdb,
            scores_tsv=scores_tsv,
            pocket_json=pocket_json,
            summary_json=summary_json,
            unscored_bfactor_policy=unscored_bfactor_policy,
            elapsed_seconds=elapsed_seconds,
            execution_mode="single_fp32",
            microbatch_size=1,
        )

    def predict_many(
        self,
        structures: Sequence[str | Path],
        *,
        chain_id: Optional[str] = None,
        threshold: float = 0.5,
        pocket_cluster_cutoff: float = 8.0,
        output_pdb: str | Path | None = None,
        scores_tsv: str | Path | None = None,
        pocket_json: str | Path | None = None,
        summary_json: str | Path | None = None,
        output_paths: Sequence[Mapping[str, str | Path] | None] | None = None,
        allow_truncation: bool = False,
        unscored_bfactor_policy: str = "zero",
        structure_inspections: Sequence[Mapping[str, object] | None] | None = None,
        batch_size: int = 4,
        max_batch_residues: int = 4096,
        max_batch_quadratic_cost: int = 4 * MAX_ESM_RESIDUES * MAX_ESM_RESIDUES,
        feature_batch_size: int | None = None,
        max_feature_padded_tokens: int = 2048,
        return_exceptions: bool = False,
    ) -> list[PredictionResult | Exception]:
        """Predict structures with bounded ESM and PointNet microbatches.

        Each structure remains one independent PointNet graph while each chain
        remains one independent ESM context.  Batches are bounded by count,
        total residues, and ``sum(n_i**2)`` to account for geometric search.
        Per-item output mappings may use either prediction argument names or
        result keys (for example ``output_pdb``/``structure``).
        """
        structures = list(structures)
        if not structures:
            return []
        _validate_prediction_options(
            threshold=threshold,
            pocket_cluster_cutoff=pocket_cluster_cutoff,
            max_len=self.max_len,
            unscored_bfactor_policy=unscored_bfactor_policy,
        )
        for name, value in (
            ("batch_size", batch_size),
            ("max_batch_residues", max_batch_residues),
            ("max_batch_quadratic_cost", max_batch_quadratic_cost),
            ("max_feature_padded_tokens", max_feature_padded_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if feature_batch_size is None:
            feature_batch_size = batch_size if self.device != "cpu" else min(batch_size, 2)
        if (
            isinstance(feature_batch_size, bool)
            or not isinstance(feature_batch_size, Integral)
            or feature_batch_size <= 0
        ):
            raise ValueError("feature_batch_size must be a positive integer.")
        if structure_inspections is not None and len(structure_inspections) != len(structures):
            raise ValueError("structure_inspections must have one entry per input structure.")
        if output_paths is not None and len(output_paths) != len(structures):
            raise ValueError("output_paths must have one entry per input structure.")
        shared_outputs = (output_pdb, scores_tsv, pocket_json, summary_json)
        if output_paths is not None and any(path is not None for path in shared_outputs):
            raise ValueError("Pass either output_paths or the single-structure output arguments, not both.")
        if len(structures) > 1 and any(path is not None for path in shared_outputs):
            raise ValueError(
                "Shared output paths are unsafe for multiple structures; pass one output_paths mapping per input."
            )

        results: list[PredictionResult | Exception | None] = [None] * len(structures)
        resolved_paths: list[tuple[Path | None, ...] | None] = [None] * len(structures)
        claimed_outputs: dict[Path, int] = {}
        for index, structure in enumerate(structures):
            try:
                paths = (
                    self._mapped_output_paths(output_paths[index])
                    if output_paths is not None
                    else tuple(_expanded_optional_path(path) for path in shared_outputs)
                )
                input_path = Path(structure).expanduser()
                _validate_output_paths(input_path, *paths)
                item_claims: list[Path] = []
                for path in paths:
                    if path is None:
                        continue
                    resolved = path.resolve(strict=False)
                    if resolved in claimed_outputs:
                        raise ValueError(
                            f"Batch output path {path} is shared by inputs at indices "
                            f"{claimed_outputs[resolved]} and {index}."
                        )
                    item_claims.append(resolved)
                claimed_outputs.update((path, index) for path in item_claims)
                resolved_paths[index] = paths
            except Exception as exc:
                if not return_exceptions:
                    raise
                results[index] = exc

        pending: list[tuple[int, _PreparedPrediction, tuple[Path | None, ...]]] = []
        pending_residues = 0
        pending_cost = 0

        def flush_pending() -> None:
            nonlocal pending, pending_residues, pending_cost
            if not pending:
                return
            completed = self._execute_prediction_group(
                pending,
                feature_batch_size=feature_batch_size,
                max_feature_padded_tokens=max_feature_padded_tokens,
                return_exceptions=return_exceptions,
                threshold=threshold,
                pocket_cluster_cutoff=pocket_cluster_cutoff,
                unscored_bfactor_policy=unscored_bfactor_policy,
            )
            for completed_index, value in completed.items():
                results[completed_index] = value
            pending = []
            pending_residues = 0
            pending_cost = 0

        for index, structure in enumerate(structures):
            paths = resolved_paths[index]
            if paths is None:
                continue
            try:
                prepared = self._prepare_prediction(
                    Path(structure).expanduser(),
                    chain_id=chain_id,
                    allow_truncation=allow_truncation,
                    structure_inspection=(
                        structure_inspections[index] if structure_inspections is not None else None
                    ),
                )
                residue_count = len(prepared.parsed["coords"])
                quadratic_cost = residue_count * residue_count
                if residue_count > max_batch_residues or quadratic_cost > max_batch_quadratic_cost:
                    raise ValueError(
                        f"Input graph at index {index} has {residue_count} residues and quadratic cost "
                        f"{quadratic_cost}, exceeding batch limits max_batch_residues={max_batch_residues} "
                        f"and max_batch_quadratic_cost={max_batch_quadratic_cost}. Select a chain or "
                        "raise the explicit limits for this structure."
                    )
                if pending and (
                    len(pending) >= batch_size
                    or pending_residues + residue_count > max_batch_residues
                    or pending_cost + quadratic_cost > max_batch_quadratic_cost
                ):
                    flush_pending()
                pending.append((index, prepared, paths))
                pending_residues += residue_count
                pending_cost += quadratic_cost
            except Exception as exc:
                if not return_exceptions:
                    raise
                results[index] = exc
        flush_pending()

        if any(value is None for value in results):
            raise RuntimeError("Internal error: one or more batch inputs did not receive a result.")
        return [value for value in results if value is not None]

    def _prepare_prediction(
        self,
        pdb_file: Path,
        *,
        chain_id: str | None,
        allow_truncation: bool,
        structure_inspection: Mapping[str, object] | None,
    ) -> _PreparedPrediction:
        if not pdb_file.exists():
            raise FileNotFoundError(f"Input structure not found: {pdb_file}")
        input_metadata = _input_file_metadata(pdb_file)
        prediction_parser = getattr(self.structure_parser, "parse_file", None)
        if callable(prediction_parser):
            parsed = prediction_parser(pdb_file, chain_id=chain_id)
        else:
            parsed = self.structure_parser.parse_file_with_labels(pdb_file, chain_id=chain_id)
        if not parsed:
            raise ValueError(f"No standard amino-acid residues with CA atoms found in {pdb_file}")

        long_chunks = parsed_structure_long_chunks(parsed, self.max_len)
        if long_chunks and not allow_truncation:
            longest = max(end - start for start, end in long_chunks)
            raise ValueError(
                f"Input has an ESM chain context of {longest} scored residues, which exceeds --max-len={self.max_len}. "
                "Pass --allow-truncation to score only the leading residues of each long chain."
            )
        parsed = truncate_parsed_structure_by_chain(parsed, self.max_len)

        inspection: dict[str, object] | None = None
        if structure_inspection is not None and _inspection_matches_input(
            structure_inspection,
            pdb_file,
            chain_id=chain_id,
            max_len=self.max_len,
        ):
            inspection = dict(structure_inspection)
        else:
            try:
                from protcross.data.inspection import inspect_structure

                inspection = inspect_structure(pdb_file, chain_id=chain_id, max_len=self.max_len)
            except Exception:
                # Custom parsers remain supported; the prediction parser is authoritative here.
                inspection = None
        structure_warnings = list(parsed.get("structure_warnings", []))
        if inspection:
            structure_warnings.extend(str(item) for item in inspection.get("warnings", []))
        return _PreparedPrediction(
            input_pdb=pdb_file,
            parsed=parsed,
            input_metadata=input_metadata,
            inspection=inspection,
            warnings=list(dict.fromkeys(structure_warnings)),
        )

    def _finish_prediction(
        self,
        prepared: _PreparedPrediction,
        probabilities: np.ndarray,
        *,
        threshold: float,
        pocket_cluster_cutoff: float,
        output_pdb: Path | None,
        scores_tsv: Path | None,
        pocket_json: Path | None,
        summary_json: Path | None,
        unscored_bfactor_policy: str,
        elapsed_seconds: float,
        execution_mode: str,
        microbatch_size: int,
    ) -> PredictionResult:
        _assert_input_file_unchanged(prepared.input_pdb, prepared.input_metadata)
        parsed = prepared.parsed
        runtime_metadata = _runtime_metadata()
        runtime_metadata.update(
            {
                "execution_mode": execution_mode,
                "precision": "float32",
                "effective_microbatch_size": str(microbatch_size),
                "elapsed_seconds_interpretation": (
                    "amortized_microbatch_compute_time" if microbatch_size > 1 else "single_item_compute_time"
                ),
            }
        )
        result = PredictionResult(
            input_pdb=prepared.input_pdb,
            residue_ids=list(parsed["residue_ids"]),
            probabilities=probabilities,
            threshold=threshold,
            truncated=bool(parsed.get("truncated", False)),
            original_length=parsed.get("original_length"),
            ca_coords=np.asarray(parsed.get("raw_coords", parsed["coords"]), dtype=np.float32),
            cluster_cutoff=pocket_cluster_cutoff,
            residue_metadata=(
                list(parsed["residue_metadata"]) if parsed.get("residue_metadata") is not None else None
            ),
            asset_version=self.asset_version,
            asset_metadata=self.asset_metadata,
            device=self.device,
            max_len=self.max_len,
            output_files=self._output_files(output_pdb, scores_tsv, pocket_json, summary_json),
            output_format_warning=self._output_format_warning(prepared.input_pdb, output_pdb),
            unscored_bfactor_policy=unscored_bfactor_policy,
            elapsed_seconds=elapsed_seconds,
            warnings=prepared.warnings,
            structure_summary=prepared.inspection,
            input_metadata=prepared.input_metadata,
            runtime_metadata=runtime_metadata,
        )
        self._write_result_package(
            result,
            output_pdb=output_pdb,
            scores_tsv=scores_tsv,
            pocket_json=pocket_json,
            summary_json=summary_json,
        )
        return result

    @staticmethod
    def _write_result_package(
        result: PredictionResult,
        *,
        output_pdb: Path | None,
        scores_tsv: Path | None,
        pocket_json: Path | None,
        summary_json: Path | None,
    ) -> None:
        """Stage every requested output before publishing the result package.

        Each stage file lives beside its final path, so publication uses an
        atomic same-filesystem replace.  Multi-file publication is guarded by
        per-file backups and best-effort rollback to prevent a failed writer or
        replace from leaving a mixture of old, new, and partial outputs.
        """
        requested = [
            (output_pdb, result.write_pdb),
            (scores_tsv, result.write_scores_tsv),
            (pocket_json, result.write_pocket_json),
            (summary_json, result.write_summary_json),
        ]
        requested = [(path, writer) for path, writer in requested if path is not None]
        if not requested:
            return

        transaction_id = uuid.uuid4().hex
        staged: dict[Path, Path] = {}
        backups: dict[Path, Path] = {}
        committed: set[Path] = set()
        try:
            for final_path, writer in requested:
                final_path.parent.mkdir(parents=True, exist_ok=True)
                stage_path = final_path.with_name(
                    f".{final_path.stem}.{transaction_id}.stage{final_path.suffix}"
                )
                stage_path.unlink(missing_ok=True)
                staged[final_path] = stage_path
                writer(stage_path)

            for final_path, _writer in requested:
                if final_path.exists():
                    backup_path = final_path.with_name(
                        f".{final_path.name}.{transaction_id}.backup"
                    )
                    backup_path.unlink(missing_ok=True)
                    final_path.replace(backup_path)
                    backups[final_path] = backup_path
                staged[final_path].replace(final_path)
                committed.add(final_path)
        except Exception as exc:
            rollback_errors = []
            for final_path, _writer in reversed(requested):
                try:
                    backup_path = backups.get(final_path)
                    if backup_path is not None and backup_path.exists():
                        final_path.unlink(missing_ok=True)
                        backup_path.replace(final_path)
                    elif final_path in committed:
                        final_path.unlink(missing_ok=True)
                except OSError as rollback_exc:
                    rollback_errors.append(f"{final_path}: {rollback_exc}")
            if rollback_errors:
                raise RuntimeError(
                    "Result-package publication failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from exc
            raise
        else:
            for backup_path in backups.values():
                backup_path.unlink(missing_ok=True)
        finally:
            for stage_path in staged.values():
                stage_path.unlink(missing_ok=True)

    @staticmethod
    def _mapped_output_paths(mapping: Mapping[str, str | Path] | None) -> tuple[Path | None, ...]:
        if mapping is None:
            return (None, None, None, None)
        aliases = (
            ("output_pdb", "structure"),
            ("scores_tsv", "scores_tsv"),
            ("pocket_json", "pockets_json"),
            ("summary_json", "summary_json"),
        )
        allowed = {name for pair in aliases for name in pair}
        unknown = set(mapping) - allowed
        if unknown:
            raise ValueError(f"Unknown batch output path key(s): {', '.join(sorted(unknown))}.")
        values = []
        for primary, alias in aliases:
            primary_value = mapping.get(primary)
            alias_value = mapping.get(alias)
            if primary != alias and primary_value is not None and alias_value is not None:
                raise ValueError(f"Specify only one of {primary!r} and {alias!r} for an item.")
            values.append(_expanded_optional_path(primary_value if primary_value is not None else alias_value))
        return tuple(values)

    @staticmethod
    def _prediction_groups(
        entries: list[tuple[int, _PreparedPrediction, tuple[Path | None, ...]]],
        *,
        batch_size: int,
        max_batch_residues: int,
        max_batch_quadratic_cost: int,
    ) -> list[list[tuple[int, _PreparedPrediction, tuple[Path | None, ...]]]]:
        groups = []
        current = []
        residues = 0
        cost = 0
        for entry in entries:
            count = len(entry[1].parsed["coords"])
            entry_cost = count * count
            if count > max_batch_residues or entry_cost > max_batch_quadratic_cost:
                raise ValueError(
                    f"A single input graph has {count} residues and quadratic cost {entry_cost}, "
                    "which exceeds the configured batch limits."
                )
            next_over_budget = current and (
                len(current) >= batch_size
                or residues + count > max_batch_residues
                or cost + entry_cost > max_batch_quadratic_cost
            )
            if next_over_budget:
                groups.append(current)
                current = []
                residues = 0
                cost = 0
            current.append(entry)
            residues += count
            cost += entry_cost
        if current:
            groups.append(current)
        return groups

    def _execute_prediction_group(
        self,
        group: list[tuple[int, _PreparedPrediction, tuple[Path | None, ...]]],
        *,
        feature_batch_size: int,
        max_feature_padded_tokens: int,
        return_exceptions: bool,
        threshold: float,
        pocket_cluster_cutoff: float,
        unscored_bfactor_policy: str,
    ) -> dict[int, PredictionResult | Exception]:
        computed = self._compute_prediction_group(
            group,
            feature_batch_size=feature_batch_size,
            max_feature_padded_tokens=max_feature_padded_tokens,
            isolate_errors=return_exceptions,
        )
        completed: dict[int, PredictionResult | Exception] = {}
        for index, prepared, paths in group:
            value = computed[index]
            if isinstance(value, Exception):
                if not return_exceptions:
                    raise value
                completed[index] = value
                continue
            probabilities, elapsed_seconds, effective_batch_size = value
            try:
                completed[index] = self._finish_prediction(
                    prepared,
                    probabilities,
                    threshold=threshold,
                    pocket_cluster_cutoff=pocket_cluster_cutoff,
                    output_pdb=paths[0],
                    scores_tsv=paths[1],
                    pocket_json=paths[2],
                    summary_json=paths[3],
                    unscored_bfactor_policy=unscored_bfactor_policy,
                    elapsed_seconds=elapsed_seconds,
                    execution_mode=(
                        "microbatch_fp32" if effective_batch_size > 1 else "single_fp32_fallback"
                    ),
                    microbatch_size=effective_batch_size,
                )
            except Exception as exc:
                if not return_exceptions:
                    raise
                completed[index] = exc
        return completed

    def _compute_prediction_group(
        self,
        group: list[tuple[int, _PreparedPrediction, tuple[Path | None, ...]]],
        *,
        feature_batch_size: int,
        max_feature_padded_tokens: int,
        isolate_errors: bool,
    ) -> dict[int, tuple[np.ndarray, float, int] | Exception]:
        start_time = time.perf_counter()
        try:
            features = self._featurize_parsed_many(
                [entry[1].parsed for entry in group],
                feature_batch_size=feature_batch_size,
                max_feature_padded_tokens=max_feature_padded_tokens,
            )
            predictions = self._infer_feature_batch(
                features,
                [entry[1].parsed for entry in group],
            )
            elapsed = (time.perf_counter() - start_time) / len(group)
            return {
                entry[0]: (probability, elapsed, effective_size)
                for entry, (probability, effective_size) in zip(group, predictions)
            }
        except Exception as exc:
            if len(group) == 1 or not isolate_errors:
                if len(group) == 1:
                    return {group[0][0]: exc}
                raise
            results: dict[int, tuple[np.ndarray, float, int] | Exception] = {}
            for entry in group:
                results.update(
                    self._compute_prediction_group(
                        [entry],
                        feature_batch_size=1,
                        max_feature_padded_tokens=max_feature_padded_tokens,
                        isolate_errors=True,
                    )
                )
            return results

    def _infer_feature_batch(
        self,
        features: list[torch.Tensor],
        parsed_items: list[dict],
    ) -> list[tuple[np.ndarray, int]]:
        data_list = [
            Data(x=item_features, pos=torch.from_numpy(parsed["coords"]))
            for item_features, parsed in zip(features, parsed_items)
        ]
        if len(data_list) == 1:
            data = data_list[0]
            data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
        else:
            data = Batch.from_data_list(data_list)
        try:
            probabilities = self._infer(data.to(self.device))
        except RuntimeError as exc:
            if len(data_list) <= 1 or "out of memory" not in str(exc).lower():
                raise
            del data
            if self.device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if self.device.startswith("mps") and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
            middle = len(data_list) // 2
            return self._infer_feature_batch(features[:middle], parsed_items[:middle]) + self._infer_feature_batch(
                features[middle:], parsed_items[middle:]
            )
        lengths = [len(parsed["coords"]) for parsed in parsed_items]
        if len(probabilities) != sum(lengths):
            if len(data_list) == 1:
                raise RuntimeError(
                    "Model returned a residue-score count that does not match the input graph."
                )
            # Preserve compatibility with injected/custom models that only
            # implement single-graph inference rather than per-node batching.
            return [
                (
                    self._infer(
                        Data(
                            x=item_features,
                            pos=torch.from_numpy(parsed["coords"]),
                            batch=torch.zeros(len(parsed["coords"]), dtype=torch.long),
                        ).to(self.device)
                    ),
                    1,
                )
                for item_features, parsed in zip(features, parsed_items)
            ]
        boundaries = np.cumsum(lengths[:-1], dtype=int)
        return [(values, len(data_list)) for values in np.split(probabilities, boundaries)]

    @staticmethod
    def _output_files(
        output_pdb: str | Path | None,
        scores_tsv: str | Path | None,
        pocket_json: str | Path | None,
        summary_json: str | Path | None,
    ) -> dict[str, str]:
        output_files = {}
        if output_pdb:
            output_files["structure"] = str(output_pdb)
        if scores_tsv:
            output_files["scores_tsv"] = str(scores_tsv)
        if pocket_json:
            output_files["pockets_json"] = str(pocket_json)
        if summary_json:
            output_files["summary_json"] = str(summary_json)
        return output_files

    @staticmethod
    def _output_format_warning(input_structure: Path, output_structure: str | Path | None) -> str | None:
        if not output_structure:
            return None
        input_suffix = input_structure.suffix.lower()
        output_suffix = Path(output_structure).suffix.lower()
        input_is_cif = input_suffix in {".cif", ".mmcif"}
        output_is_cif = output_suffix in {".cif", ".mmcif"}
        if input_is_cif != output_is_cif:
            return "Output structure format differs from the input structure extension."
        return None

    def _featurize(self, sequence: str) -> torch.Tensor:
        cache_path = self._embedding_cache_path(sequence)
        if cache_path and cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True).float()
                if self._valid_cached_features(cached, len(sequence)):
                    return cached
                raise ValueError("cached feature tensor has an incompatible shape or non-finite values")
            except Exception:
                # A killed writer or an older incompatible cache must not make
                # the underlying structure permanently unscorable.
                cache_path.unlink(missing_ok=True)

        raw_embeddings = self.esm_extractor.extract_residue_embeddings(sequence)
        reduced_embeddings = self.pca_reducer.transform(raw_embeddings)
        if not self._valid_cached_features(reduced_embeddings, len(sequence)):
            raise ValueError(
                "ESM/PCA feature tensor must be finite with shape "
                f"({len(sequence)}, expected PCA dimension); got {tuple(reduced_embeddings.shape)}."
            )
        if cache_path:
            self._write_cached_features(cache_path, reduced_embeddings)
        return reduced_embeddings.float()

    def _featurize_parsed(self, parsed: dict) -> torch.Tensor:
        sequence = parsed["sequence"]
        chunks = parsed_structure_sequence_chunks(parsed)
        if len(chunks) <= 1:
            return self._featurize(sequence)

        features = [self._featurize(sequence[start:end]) for start, end in chunks if start < end]
        return torch.cat(features, dim=0)

    def _featurize_parsed_many(
        self,
        parsed_items: list[dict],
        *,
        feature_batch_size: int,
        max_feature_padded_tokens: int,
    ) -> list[torch.Tensor]:
        sequence_chunks: list[str] = []
        owners: list[list[int]] = []
        for parsed in parsed_items:
            sequence = parsed["sequence"]
            chunks = parsed_structure_sequence_chunks(parsed)
            owners.append([])
            for start, end in chunks:
                if start >= end:
                    continue
                owners[-1].append(len(sequence_chunks))
                sequence_chunks.append(sequence[start:end])

        chunk_features = self._featurize_sequences_many(
            sequence_chunks,
            feature_batch_size=feature_batch_size,
            max_feature_padded_tokens=max_feature_padded_tokens,
        )
        return [torch.cat([chunk_features[index] for index in indices], dim=0) for indices in owners]

    def _featurize_sequences_many(
        self,
        sequences: list[str],
        *,
        feature_batch_size: int,
        max_feature_padded_tokens: int,
    ) -> list[torch.Tensor]:
        results: list[torch.Tensor | None] = [None] * len(sequences)
        missing_by_sequence: dict[str, list[int]] = {}
        for index, sequence in enumerate(sequences):
            cache_path = self._embedding_cache_path(sequence)
            if cache_path and cache_path.exists():
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=True).float()
                    if not self._valid_cached_features(cached, len(sequence)):
                        raise ValueError("invalid cached tensor")
                    results[index] = cached
                    continue
                except Exception:
                    cache_path.unlink(missing_ok=True)
            missing_by_sequence.setdefault(sequence, []).append(index)

        unique_sequences = list(missing_by_sequence)
        if unique_sequences:
            batch_extractor = getattr(self.esm_extractor, "extract_residue_embeddings_many", None)
            if callable(batch_extractor) and len(unique_sequences) > 1:
                raw_embeddings = batch_extractor(
                    unique_sequences,
                    max_batch_size=feature_batch_size,
                    max_padded_tokens=max_feature_padded_tokens,
                )
            else:
                raw_embeddings = [
                    self.esm_extractor.extract_residue_embeddings(sequence)
                    for sequence in unique_sequences
                ]
            if len(raw_embeddings) != len(unique_sequences):
                raise RuntimeError("ESM batch extractor returned the wrong number of feature tensors.")
            for sequence, raw in zip(unique_sequences, raw_embeddings):
                reduced = self.pca_reducer.transform(raw).float()
                if not self._valid_cached_features(reduced, len(sequence)):
                    raise ValueError(
                        "ESM/PCA feature tensor must be finite with shape "
                        f"({len(sequence)}, expected PCA dimension); got {tuple(reduced.shape)}."
                    )
                cache_path = self._embedding_cache_path(sequence)
                if cache_path:
                    self._write_cached_features(cache_path, reduced)
                for index in missing_by_sequence[sequence]:
                    results[index] = reduced
        if any(result is None for result in results):
            raise RuntimeError("Internal error while assembling batched ESM/PCA features.")
        return [result for result in results if result is not None]

    def _valid_cached_features(self, features: torch.Tensor, sequence_length: int) -> bool:
        return bool(
            isinstance(features, torch.Tensor)
            and features.ndim == 2
            and features.shape[0] == sequence_length
            and features.shape[1] > 0
            and (self._expected_feature_dim is None or features.shape[1] == self._expected_feature_dim)
            and torch.isfinite(features).all().item()
        )

    @staticmethod
    def _write_cached_features(cache_path: Path, features: torch.Tensor) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_name(
            f".{cache_path.stem}.{uuid.uuid4().hex}.part{cache_path.suffix}"
        )
        try:
            torch.save(features.cpu(), tmp_path)
            tmp_path.replace(cache_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _embedding_cache_path(self, sequence: str) -> Path | None:
        if self.embedding_cache_dir is None:
            return None
        payload = "\n".join(
            [
                sequence,
                "feature_cache_schema=protcross-esm-pca-v2",
                f"pca_dim={self.pca_dim}",
                f"max_len={self.max_len}",
                f"asset_version={self.asset_version or 'unknown'}",
                f"feature_pipeline={self.feature_pipeline_fingerprint or 'release-assets'}",
                self._embedding_cache_asset_identity,
            ]
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.embedding_cache_dir / f"{digest}.pt"

    @torch.no_grad()
    def _infer(self, data: Data) -> np.ndarray:
        self.model.eval()
        use_esm = bool(getattr(self.model.hparams, "use_esm", True))
        x = data.x if use_esm else None
        feats, _ = self.model.backbone(x, data.pos, data.batch)
        logits = self.model.seg_head(feats)
        return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

    def _load_model(self, ckpt_path: Path) -> EvoPointDALitModule:
        model = EvoPointDALitModule.load_from_checkpoint(str(ckpt_path), map_location=self.device)
        model.eval()
        model.to(self.device)
        return model

    @staticmethod
    def _load_pca(pca_path: Path, pca_dim: int) -> PCAReducer:
        reducer = PCAReducer(n_components=pca_dim)
        reducer.load(pca_path)
        actual_dim = getattr(reducer.pca, "n_components_", getattr(reducer.pca, "n_components", None))
        if actual_dim is not None and int(actual_dim) != pca_dim:
            raise ValueError(
                f"PCA reducer has {int(actual_dim)} components, but pca_dim={pca_dim} was requested. "
                "Use the dimension recorded by the matching asset bundle."
            )
        return reducer

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device in (None, "auto"):
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        normalized = str(device).strip().lower()
        try:
            parsed = torch.device(normalized)
        except (RuntimeError, ValueError) as exc:
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N") from exc
        if parsed.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but torch.cuda.is_available() is false; use --device cpu.")
            if parsed.index is not None and parsed.index >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device {parsed.index} was requested, but only {torch.cuda.device_count()} device(s) are available."
                )
        elif parsed.type == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise ValueError("MPS was requested but is unavailable; use --device cpu.")
        elif parsed.type != "cpu":
            raise ValueError("device must be auto, cpu, mps, cuda, or cuda:N")
        return normalized

    @staticmethod
    def _optional_existing_path(value: str | Path, name: str) -> Path:
        path = Path(value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        return path

    @classmethod
    def _require_path(cls, value: str | Path | None, name: str) -> Path:
        if value is None:
            raise ValueError(f"{name} is required.")
        return cls._optional_existing_path(value, name)

    @staticmethod
    def _embedding_cache_asset_identity_for(
        esm_weights: Path | None,
        pca_path: Path | None,
        asset_metadata: Mapping[str, object] | None = None,
    ) -> str:
        def identity(name: str, path: Path | None) -> str:
            entry = asset_metadata.get(name) if asset_metadata else None
            if isinstance(entry, Mapping):
                digest = entry.get("actual_sha256") or entry.get("sha256")
                if digest:
                    return f"sha256={digest}"
            return ProtCrossPredictor._file_cache_identity(path)

        return "\n".join(
            [
                f"esm={identity('esm_weights', esm_weights)}",
                f"pca={identity('pca', pca_path)}",
            ]
        )

    @staticmethod
    def _file_cache_identity(path: Path | None) -> str:
        if path is None:
            return "injected"
        stat = path.stat()
        return f"{path.resolve()}|size={stat.st_size}|mtime_ns={stat.st_mtime_ns}"



def predict_pdb(
    pdb_file: str | Path,
    *,
    ckpt_path: str | Path | None = None,
    esm_weights: str | Path | None = None,
    pca_path: str | Path | None = None,
    assets_dir: str | Path | None = None,
    output_pdb: str | Path | None = None,
    chain_id: Optional[str] = None,
    threshold: float = 0.5,
    device: str | None = None,
    scores_tsv: str | Path | None = None,
    pocket_json: str | Path | None = None,
    summary_json: str | Path | None = None,
    pocket_cluster_cutoff: float = 8.0,
    auto_setup_assets: bool = True,
    asset_version: str | None = None,
    refresh_assets: bool = False,
    offline: bool = False,
    accept_esm_license: bool = False,
    trust_unverified_assets: bool = False,
    max_len: int = MAX_ESM_RESIDUES,
    allow_truncation: bool = False,
    unscored_bfactor_policy: str = "zero",
    embedding_cache_dir: str | Path | None = None,
) -> PredictionResult:
    pdb_file = Path(pdb_file).expanduser()
    if not pdb_file.exists():
        raise FileNotFoundError(f"Input structure not found: {pdb_file}")
    _validate_prediction_options(
        threshold=threshold,
        pocket_cluster_cutoff=pocket_cluster_cutoff,
        max_len=max_len,
        unscored_bfactor_policy=unscored_bfactor_policy,
    )
    _validate_output_paths(
        pdb_file,
        output_pdb,
        scores_tsv,
        pocket_json,
        summary_json,
    )
    resolved_device = ProtCrossPredictor._resolve_device(device)
    structure_inspection = _preflight_prediction_structure(
        pdb_file,
        chain_id=chain_id,
        max_len=max_len,
        allow_truncation=allow_truncation,
    )
    resolved = resolve_prediction_assets(
        ckpt_path,
        esm_weights,
        pca_path,
        assets_dir=assets_dir,
        auto_setup_assets=auto_setup_assets,
        asset_version=asset_version,
        refresh_assets=refresh_assets,
        offline=offline,
        accept_esm_license=accept_esm_license,
        require_esm_license_for_use=True,
        trust_unverified_assets=trust_unverified_assets,
    )
    predictor = ProtCrossPredictor.from_files(
        ckpt_path=resolved.checkpoint,
        esm_weights=resolved.esm_weights,
        pca_path=resolved.pca,
        device=resolved_device,
        max_len=max_len,
        asset_version=resolved.asset_version,
        asset_metadata=resolved.asset_metadata,
        embedding_cache_dir=embedding_cache_dir,
        accept_esm_license=True,
    )
    return predictor.predict(
        pdb_file,
        chain_id=chain_id,
        threshold=threshold,
        pocket_cluster_cutoff=pocket_cluster_cutoff,
        output_pdb=output_pdb,
        scores_tsv=scores_tsv,
        pocket_json=pocket_json,
        summary_json=summary_json,
        allow_truncation=allow_truncation,
        unscored_bfactor_policy=unscored_bfactor_policy,
        structure_inspection=structure_inspection,
    )


def _resolve_predict_pdb_assets(
    ckpt_path: str | Path | None,
    esm_weights: str | Path | None,
    pca_path: str | Path | None,
    *,
    assets_dir: str | Path | None = None,
    auto_setup_assets: bool = True,
    asset_version: str | None = None,
    refresh_assets: bool = False,
    offline: bool = False,
    accept_esm_license: bool = False,
    trust_unverified_assets: bool = False,
) -> tuple[str | Path, str | Path, str | Path]:
    resolved = resolve_prediction_assets(
        ckpt_path,
        esm_weights,
        pca_path,
        assets_dir=assets_dir,
        auto_setup_assets=auto_setup_assets,
        asset_version=asset_version,
        refresh_assets=refresh_assets,
        offline=offline,
        accept_esm_license=accept_esm_license,
        trust_unverified_assets=trust_unverified_assets,
    )
    return resolved.checkpoint, resolved.esm_weights, resolved.pca


def _validate_prediction_options(
    *,
    threshold: float,
    pocket_cluster_cutoff: float,
    max_len: int,
    unscored_bfactor_policy: str,
) -> None:
    threshold_value = float(threshold)
    cutoff_value = float(pocket_cluster_cutoff)
    if not math.isfinite(threshold_value) or not 0.0 <= threshold_value <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if not math.isfinite(cutoff_value) or cutoff_value <= 0:
        raise ValueError("pocket_cluster_cutoff must be greater than 0.")
    if max_len <= 0 or max_len > MAX_ESM_RESIDUES:
        raise ValueError(f"max_len must be between 1 and {MAX_ESM_RESIDUES}.")
    if unscored_bfactor_policy not in {"keep", "zero"}:
        raise ValueError("unscored_bfactor_policy must be 'keep' or 'zero'.")


def _validate_output_paths(
    input_structure: str | Path,
    *output_paths: str | Path | None,
) -> None:
    input_path = Path(input_structure).expanduser()
    resolved_input = input_path.resolve(strict=False)
    if output_paths and output_paths[0] is not None:
        structure_output = Path(output_paths[0]).expanduser()
        input_is_cif = input_path.suffix.lower() in {".cif", ".mmcif"}
        output_is_cif = structure_output.suffix.lower() in {".cif", ".mmcif"}
        if input_is_cif != output_is_cif:
            raise ValueError(
                "Annotated structure output must use the same PDB or mmCIF format as the input."
            )
    seen: dict[Path, Path] = {}
    for value in output_paths:
        if value is None:
            continue
        path = Path(value).expanduser()
        if path.is_dir():
            raise IsADirectoryError(f"Output path is a directory, not a file: {path}.")
        resolved = path.resolve(strict=False)
        if resolved == resolved_input:
            raise ValueError(f"Output path must not overwrite the input structure: {path}.")
        if resolved in seen:
            raise ValueError(
                "Output paths must be distinct; "
                f"{seen[resolved]} and {path} resolve to the same file."
            )
        seen[resolved] = path


def _expanded_optional_path(value: str | Path | None) -> Path | None:
    return Path(value).expanduser() if value is not None else None


def _input_file_metadata(path: Path) -> dict[str, object]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _assert_input_file_unchanged(path: Path, expected: dict[str, object]) -> None:
    current = _input_file_metadata(path)
    if (
        current["sha256"] != expected.get("sha256")
        or current["size_bytes"] != expected.get("size_bytes")
    ):
        raise RuntimeError(
            f"Input structure changed while prediction results were being prepared: {path}. "
            "No annotated structure was written; rerun with a stable input file."
        )


@lru_cache(maxsize=1)
def _runtime_metadata_base() -> tuple[tuple[str, str], ...]:
    packages = {
        "biopython": "biopython",
        "esm": "esm",
        "torch_geometric": "torch-geometric",
    }
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }
    for key, distribution in packages.items():
        try:
            metadata[key] = distribution_version(distribution)
        except PackageNotFoundError:
            metadata[key] = "unknown"
    return tuple(sorted(metadata.items()))


def _runtime_metadata() -> dict[str, str]:
    return dict(_runtime_metadata_base())


def _inspection_matches_input(
    inspection: Mapping[str, object],
    path: Path,
    *,
    chain_id: str | None,
    max_len: int,
) -> bool:
    try:
        inspected_path = Path(str(inspection.get("input_structure", ""))).expanduser()
        stat = path.stat()
        selected_chains = list(inspection.get("selected_chains") or [])
        available_chains = list(inspection.get("available_chains") or [])
        expected_chains = [chain_id] if chain_id is not None else available_chains
        return (
            inspection.get("schema_version") == "protcross-structure-inspection-v1"
            and inspected_path.resolve(strict=False) == path.resolve(strict=False)
            and int(inspection.get("file_size_bytes", -1)) == stat.st_size
            and int(inspection.get("file_mtime_ns", -1)) == stat.st_mtime_ns
            and int(inspection.get("max_len", -1)) == max_len
            and selected_chains == expected_chains
        )
    except (OSError, TypeError, ValueError):
        return False


def _max_pairwise_distance(coords: np.ndarray) -> float:
    """Return cluster diameter without allocating a full N-by-N distance matrix."""
    if len(coords) < 2:
        return 0.0
    maximum = 0.0
    for index in range(len(coords) - 1):
        distances = np.linalg.norm(coords[index + 1 :] - coords[index], axis=1)
        if len(distances):
            maximum = max(maximum, float(distances.max()))
    return maximum


def _preflight_prediction_structure(
    pdb_file: Path,
    *,
    chain_id: Optional[str],
    max_len: int,
    allow_truncation: bool,
) -> dict[str, object]:
    from protcross.data.inspection import inspect_structure

    inspection = inspect_structure(pdb_file, chain_id=chain_id, max_len=max_len)
    if inspection["requires_truncation"] and not allow_truncation:
        longest = int(inspection["longest_chain_context"])
        raise ValueError(
            f"Input has an ESM chain context of {longest} scored residues, which exceeds --max-len={max_len}. "
            "Pass allow_truncation=True or CLI --allow-truncation to score only the leading residues of each long chain."
        )
    return inspection
