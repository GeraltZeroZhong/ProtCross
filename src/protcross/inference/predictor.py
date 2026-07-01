"""Lightweight prediction API for applying ProtCross to one structure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from protcross.assets import PredictorAssets, require_esm_license_acceptance, resolve_prediction_assets
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


@dataclass(frozen=True)
class ResiduePrediction:
    residue_id: str
    probability: float
    is_binding: bool


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
    _records_cache: list[dict[str, str | int | float | None]] | None = field(default=None, init=False, repr=False)
    _cluster_indices_cache: list[np.ndarray] | None = field(default=None, init=False, repr=False)
    _cluster_id_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _pocket_dict_cache: dict | None = field(default=None, init=False, repr=False)
    _summary_dict_cache: dict | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.input_pdb = Path(self.input_pdb)
        self.probabilities = np.asarray(self.probabilities, dtype=float)
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
        if not 0.0 <= float(self.threshold) <= 1.0:
            raise ValueError("Prediction threshold must be in [0, 1].")
        if self.cluster_cutoff <= 0:
            raise ValueError("Pocket cluster cutoff must be greater than 0.")
        if self.unscored_bfactor_policy not in {"keep", "zero"}:
            raise ValueError("unscored_bfactor_policy must be 'keep' or 'zero'.")
        if self.ca_coords is not None:
            coords = np.asarray(self.ca_coords, dtype=float)
            if coords.shape != (len(self.residue_ids), 3):
                raise ValueError(
                    "Original CA coordinates must have shape "
                    f"({len(self.residue_ids)}, 3); got {coords.shape}."
                )
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

    @property
    def binding_residues(self) -> list[ResiduePrediction]:
        return [
            ResiduePrediction(residue_id, float(probability), bool(probability > self.threshold))
            for residue_id, probability in zip(self.residue_ids, self.probabilities)
            if probability > self.threshold
        ]

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
        write_bfactor_pdb(
            self.input_pdb,
            output_pdb,
            self.residue_ids,
            self.probabilities,
            missing_value=missing_value,
            residue_metadata=self.residue_metadata,
        )

    def write_scores_tsv(self, output_tsv: str | Path, *, schema: str = "extended") -> None:
        output_tsv = Path(output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
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
                "input_bfactor\tprobability\tis_binding\tx\ty\tz\tcluster_id\tis_scored\t"
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
                    f"{record['probability']:.6f}\t{record['is_binding']}\t"
                    f"{self._format_optional_float(record['x'])}\t"
                    f"{self._format_optional_float(record['y'])}\t"
                    f"{self._format_optional_float(record['z'])}\t{cluster_id}\t"
                    f"{record['is_scored']}\t{record['rank_global']}\t{record['rank_within_chain']}\n"
                )

    def write_pocket_json(self, output_json: str | Path) -> None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(self.to_pocket_dict(), indent=2), encoding="utf-8")

    def write_summary_json(self, output_json: str | Path) -> None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(self.to_summary_dict(), indent=2), encoding="utf-8")

    def to_records(self) -> list[dict[str, str | int | float | None]]:
        if self._records_cache is not None:
            return self._records_cache
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
        return records

    def to_pocket_dict(self) -> dict:
        if self._pocket_dict_cache is not None:
            return self._pocket_dict_cache
        coords = self._coordinate_array(required=True)
        selected_indices = self._selected_indices()
        clusters = self._cluster_indices()
        records = self.to_records()
        payload = {
            "schema_version": "protcross-pocket-v1",
            "protcross_version": self._protcross_version(),
            "asset_version": self.asset_version,
            "assets": self.asset_metadata or {},
            "input_structure": str(self.input_pdb),
            "threshold": float(self.threshold),
            "threshold_operator": ">",
            "score_type": "softmax_probability",
            "coordinate_frame": "input_structure",
            "coordinate_units": "angstrom",
            "atom_basis": "CA",
            "center_type": "probability_weighted_ca_centroid",
            "cluster_cutoff": float(self.cluster_cutoff),
            "cluster_rank_basis": "residue_count_then_score_mean_then_score_max",
            "truncated": bool(self.truncated),
            "original_residue_count": self.original_residue_count,
            "residues_scored": len(self.residue_ids),
            "selected_residue_count": int(len(selected_indices)),
            "chains_analyzed": self.chains_analyzed,
            "residue_id_namespaces": self.residue_id_namespaces,
            "warnings": self.warnings,
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
        return payload

    def to_summary_dict(self) -> dict:
        if self._summary_dict_cache is not None:
            return self._summary_dict_cache
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
                "probability": record["probability"],
                "rank_global": record["rank_global"],
            }
            for record in sorted(self.to_records(), key=lambda record: int(record["rank_global"]))[:10]
        ]
        payload = {
            "schema_version": "protcross-summary-v1",
            "protcross_version": self._protcross_version(),
            "asset_version": self.asset_version,
            "assets": self.asset_metadata or {},
            "input_structure": str(self.input_pdb),
            "device": self.device,
            "threshold": float(self.threshold),
            "threshold_operator": ">",
            "score_type": "softmax_probability",
            "cluster_cutoff": float(self.cluster_cutoff),
            "chains_analyzed": self.chains_analyzed,
            "residue_id_namespaces": self.residue_id_namespaces,
            "residues_scored": len(self.residue_ids),
            "original_residue_count": self.original_residue_count,
            "selected_residue_count": int(len(self._selected_indices())),
            "selected_residue_fraction": (
                float(len(self._selected_indices()) / len(self.residue_ids)) if self.residue_ids else 0.0
            ),
            "probability_min": float(self.probabilities.min()) if len(self.probabilities) else None,
            "probability_max": float(self.probabilities.max()) if len(self.probabilities) else None,
            "probability_mean": float(self.probabilities.mean()) if len(self.probabilities) else None,
            "truncated": bool(self.truncated),
            "max_len": self.max_len,
            "unscored_bfactor_policy": self.unscored_bfactor_policy,
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
        }
        self._summary_dict_cache = payload
        return payload

    def format_summary(self, *, max_items: int = 50) -> str:
        summary = self.to_summary_dict()
        hits = self.binding_residues
        lines = [
            f"Input: {self.input_pdb}",
            f"Asset version: {self.asset_version or 'unspecified'}",
            f"Device: {self.device or 'unspecified'}",
            f"Chains analyzed: {', '.join(summary['chains_analyzed']) or 'none'}",
            f"Residues scored: {summary['residues_scored']} / {summary['original_residue_count']}",
            f"Threshold: {self.threshold:.2f} (binary calls, TSV is_binding, pocket selection, clustering, summary)",
            f"Cluster cutoff: {self.cluster_cutoff:.2f} A",
            f"Unscored B-factor policy: {summary['unscored_bfactor_policy']}",
            f"Predicted binding residues: {len(hits)}",
        ]
        if summary["probability_min"] is not None:
            lines.append(
                "Probability range: "
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
            lines.append(f"Pocket center: {center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}")
            lines.append(f"Top pocket residues: {summary['top_pocket']['residue_count']}")
            lines.append(f"Clustered pockets: {summary['cluster_count']}")
        if summary["output_files"]:
            for label, path in summary["output_files"].items():
                lines.append(f"Wrote {label}: {path}")
        return "\n".join(lines)

    @property
    def original_residue_count(self) -> int:
        return int(self.original_length or len(self.residue_ids))

    @property
    def chains_analyzed(self) -> list[str]:
        return sorted({str(metadata.get("chain_id", "")) for metadata in self.residue_metadata if metadata.get("chain_id")})

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
        chain_ids = self.chains_analyzed or [""]
        for chain_id in chain_ids:
            indices = [
                index
                for index, metadata in enumerate(self.residue_metadata)
                if str(metadata.get("chain_id", "")) == chain_id
            ]
            if not indices:
                indices = list(range(len(self.probabilities)))
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
    ) -> None:
        self.device = self._resolve_device(device)
        if max_len <= 0 or max_len > MAX_ESM_RESIDUES:
            raise ValueError(f"max_len must be between 1 and {MAX_ESM_RESIDUES}.")
        self.max_len = max_len
        if pca_dim <= 0:
            raise ValueError("pca_dim must be greater than 0.")
        self.pca_dim = pca_dim
        self.embedding_cache_dir = Path(embedding_cache_dir).expanduser() if embedding_cache_dir else None
        self.asset_version = asset_version
        self.asset_metadata = asset_metadata
        esm_weights_path = self._optional_existing_path(esm_weights, "esm_weights") if esm_weights is not None else None
        pca_path_obj = self._optional_existing_path(pca_path, "pca_path") if pca_path is not None else None
        ckpt_path_obj = self._optional_existing_path(ckpt_path, "ckpt_path") if ckpt_path is not None else None
        self._embedding_cache_asset_identity = self._embedding_cache_asset_identity_for(
            esm_weights_path,
            pca_path_obj,
        )
        self.structure_parser = structure_parser or StructureParser()
        if esm_extractor is None:
            require_esm_license_acceptance(accept_esm_license)
        self.esm_extractor = esm_extractor or ESMFeatureExtractor(
            self._require_path(esm_weights_path, "esm_weights"),
            self.device,
        )
        self.pca_reducer = pca_reducer or self._load_pca(self._require_path(pca_path_obj, "pca_path"), pca_dim)
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
        return cls.from_files(
            ckpt_path=assets.checkpoint,
            esm_weights=assets.esm_weights,
            pca_path=assets.pca,
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
            asset_version=assets.asset_version,
            asset_metadata=None,
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
    ) -> "ProtCrossPredictor":
        return cls.from_assets(
            PredictorAssets.from_env(),
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=accept_esm_license,
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
        return cls.from_assets(
            PredictorAssets.from_default_dir(asset_version=asset_version),
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
            embedding_cache_dir=embedding_cache_dir,
            accept_esm_license=accept_esm_license,
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
    ) -> PredictionResult:
        _validate_prediction_options(
            threshold=threshold,
            pocket_cluster_cutoff=pocket_cluster_cutoff,
            max_len=self.max_len,
            unscored_bfactor_policy=unscored_bfactor_policy,
        )
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"Input structure not found: {pdb_file}")

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
        start_time = time.perf_counter()
        features = self._featurize_parsed(parsed)
        data = Data(
            x=features,
            pos=torch.from_numpy(parsed["coords"]),
            batch=torch.zeros(len(parsed["coords"]), dtype=torch.long),
        ).to(self.device)

        probabilities = self._infer(data)
        elapsed_seconds = time.perf_counter() - start_time
        result = PredictionResult(
            input_pdb=pdb_file,
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
            output_format_warning=self._output_format_warning(pdb_file, output_pdb),
            unscored_bfactor_policy=unscored_bfactor_policy,
            elapsed_seconds=elapsed_seconds,
            warnings=list(parsed.get("structure_warnings", [])),
        )

        if output_pdb:
            result.write_pdb(output_pdb)
        if scores_tsv:
            result.write_scores_tsv(scores_tsv)
        if pocket_json:
            result.write_pocket_json(pocket_json)
        if summary_json:
            result.write_summary_json(summary_json)

        return result

    def predict_many(
        self,
        structures: list[str | Path] | tuple[str | Path, ...],
        **kwargs,
    ) -> list[PredictionResult]:
        """Predict multiple structures while reusing loaded ESM, PCA, and model assets."""
        return [self.predict(structure, **kwargs) for structure in structures]

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
                return torch.load(cache_path, map_location="cpu", weights_only=True).float()
            except TypeError:
                return torch.load(cache_path, map_location="cpu").float()

        raw_embeddings = self.esm_extractor.extract_residue_embeddings(sequence)
        reduced_embeddings = self.pca_reducer.transform(raw_embeddings)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(cache_path.suffix + ".part")
            torch.save(reduced_embeddings.cpu(), tmp_path)
            tmp_path.replace(cache_path)
        return reduced_embeddings.float()

    def _featurize_parsed(self, parsed: dict) -> torch.Tensor:
        sequence = parsed["sequence"]
        chunks = parsed_structure_sequence_chunks(parsed)
        if len(chunks) <= 1:
            return self._featurize(sequence)

        features = [self._featurize(sequence[start:end]) for start, end in chunks if start < end]
        return torch.cat(features, dim=0)

    def _embedding_cache_path(self, sequence: str) -> Path | None:
        if self.embedding_cache_dir is None:
            return None
        payload = "\n".join(
            [
                sequence,
                f"pca_dim={self.pca_dim}",
                f"max_len={self.max_len}",
                f"asset_version={self.asset_version or 'unknown'}",
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
        return reducer

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device in (None, "auto"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

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
    def _embedding_cache_asset_identity_for(esm_weights: Path | None, pca_path: Path | None) -> str:
        return "\n".join(
            [
                f"esm={ProtCrossPredictor._file_cache_identity(esm_weights)}",
                f"pca={ProtCrossPredictor._file_cache_identity(pca_path)}",
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
    asset_version: str = "default",
    refresh_assets: bool = False,
    offline: bool = False,
    accept_esm_license: bool = False,
    trust_unverified_assets: bool = False,
    max_len: int = MAX_ESM_RESIDUES,
    allow_truncation: bool = False,
    unscored_bfactor_policy: str = "zero",
    embedding_cache_dir: str | Path | None = None,
) -> PredictionResult:
    pdb_file = Path(pdb_file)
    if not pdb_file.exists():
        raise FileNotFoundError(f"Input structure not found: {pdb_file}")
    _validate_prediction_options(
        threshold=threshold,
        pocket_cluster_cutoff=pocket_cluster_cutoff,
        max_len=max_len,
        unscored_bfactor_policy=unscored_bfactor_policy,
    )
    _preflight_prediction_structure(
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
        device=device,
        max_len=max_len,
        asset_version=resolved.asset_version,
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
    )


def _resolve_predict_pdb_assets(
    ckpt_path: str | Path | None,
    esm_weights: str | Path | None,
    pca_path: str | Path | None,
    *,
    assets_dir: str | Path | None = None,
    auto_setup_assets: bool = True,
    asset_version: str = "default",
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


def _preflight_prediction_structure(
    pdb_file: Path,
    *,
    chain_id: Optional[str],
    max_len: int,
    allow_truncation: bool,
) -> None:
    parser = StructureParser()
    parsed = parser.parse_file_with_labels(pdb_file, chain_id=chain_id)
    if not parsed:
        if chain_id:
            any_chain = parser.parse_file_with_labels(pdb_file, chain_id=None)
            if any_chain:
                raise ValueError(f"No standard amino-acid residues with CA atoms found for chain {chain_id!r}.")
        raise ValueError(f"No standard amino-acid residues with CA atoms found in {pdb_file}.")
    long_chunks = parsed_structure_long_chunks(parsed, max_len)
    if long_chunks and not allow_truncation:
        longest = max(end - start for start, end in long_chunks)
        raise ValueError(
            f"Input has an ESM chain context of {longest} scored residues, which exceeds --max-len={max_len}. "
            "Pass allow_truncation=True or CLI --allow-truncation to score only the leading residues of each long chain."
        )
