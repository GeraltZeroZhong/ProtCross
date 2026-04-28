"""Lightweight prediction API for applying ProtCross to one structure."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch_geometric.data import Data

from evopoint_da.data import MAX_ESM_RESIDUES, PCAReducer, StructureParser, truncate_parsed_structure
from evopoint_da.data.esm import ESMFeatureExtractor
from evopoint_da.models import EvoPointDALitModule

from .pdb import write_bfactor_pdb


@dataclass(frozen=True)
class ResiduePrediction:
    residue_id: str
    probability: float
    is_binding: bool


@dataclass(frozen=True)
class PredictorAssets:
    checkpoint: Path
    esm_weights: Path
    pca: Path

    @classmethod
    def from_dir(cls, assets_dir: str | Path) -> "PredictorAssets":
        from evopoint_da.assets import (
            DEFAULT_CHECKPOINT_FILENAME,
            DEFAULT_PCA_FILENAME,
            LEGACY_CHECKPOINT_FILENAME,
            LEGACY_PCA_FILENAME,
        )

        assets_dir = Path(assets_dir)
        checkpoint = assets_dir / DEFAULT_CHECKPOINT_FILENAME
        if not checkpoint.exists() and (assets_dir / LEGACY_CHECKPOINT_FILENAME).exists():
            checkpoint = assets_dir / LEGACY_CHECKPOINT_FILENAME
        pca = assets_dir / DEFAULT_PCA_FILENAME
        if not pca.exists() and (assets_dir / LEGACY_PCA_FILENAME).exists():
            pca = assets_dir / LEGACY_PCA_FILENAME
        return cls(
            checkpoint=checkpoint,
            esm_weights=assets_dir / "esmc_600m_2024_12_v0.pth",
            pca=pca,
        )

    @classmethod
    def from_default_dir(cls) -> "PredictorAssets":
        from evopoint_da.assets import get_default_assets_dir

        return cls.from_dir(get_default_assets_dir())

    @classmethod
    def from_env(cls) -> "PredictorAssets":
        missing = [
            name
            for name in ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA")
            if not os.environ.get(name)
        ]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        return cls(
            checkpoint=Path(os.environ["PROTCROSS_CHECKPOINT"]),
            esm_weights=Path(os.environ["PROTCROSS_ESM_WEIGHTS"]),
            pca=Path(os.environ["PROTCROSS_PCA"]),
        )

    def missing_files(self) -> list[Path]:
        return [path for path in (self.checkpoint, self.esm_weights, self.pca) if not path.exists()]

    def is_complete(self) -> bool:
        return not self.missing_files()


@dataclass
class PredictionResult:
    input_pdb: Path
    residue_ids: list[str]
    probabilities: np.ndarray
    threshold: float = 0.5
    truncated: bool = False
    original_length: int | None = None

    @property
    def binding_residues(self) -> list[ResiduePrediction]:
        return [
            ResiduePrediction(residue_id, float(probability), bool(probability > self.threshold))
            for residue_id, probability in zip(self.residue_ids, self.probabilities)
            if probability > self.threshold
        ]

    def write_pdb(self, output_pdb: str | Path, *, missing_value: float = 0.0) -> None:
        write_bfactor_pdb(
            self.input_pdb,
            output_pdb,
            self.residue_ids,
            self.probabilities,
            missing_value=missing_value,
        )

    def write_scores_tsv(self, output_tsv: str | Path) -> None:
        output_tsv = Path(output_tsv)
        output_tsv.parent.mkdir(parents=True, exist_ok=True)
        with output_tsv.open("w", encoding="utf-8") as file:
            file.write("residue_id\tchain_id\tresidue_number\tprobability\tis_binding\n")
            for record in self.to_records():
                file.write(
                    f"{record['residue_id']}\t{record['chain_id']}\t{record['residue_number']}\t"
                    f"{record['probability']:.6f}\t{record['is_binding']}\n"
                )

    def to_records(self) -> list[dict[str, str | int | float]]:
        records = []
        for residue_id, probability in zip(self.residue_ids, self.probabilities):
            chain_id, residue_number = self._split_residue_id(residue_id)
            records.append(
                {
                    "residue_id": residue_id,
                    "chain_id": chain_id,
                    "residue_number": residue_number,
                    "probability": float(probability),
                    "is_binding": int(probability > self.threshold),
                }
            )
        return records

    def format_summary(self, *, max_items: int = 50) -> str:
        hits = self.binding_residues
        lines = [
            f"Residues scored: {len(self.residue_ids)}",
            f"Threshold: {self.threshold:.2f}",
            f"Predicted binding residues: {len(hits)}",
            (
                "Probability range: "
                f"{float(self.probabilities.min()):.4f} - {float(self.probabilities.max()):.4f} "
                f"(mean {float(self.probabilities.mean()):.4f})"
            ),
        ]
        if self.truncated:
            lines.append(f"Input was truncated from {self.original_length} to {len(self.residue_ids)} residues.")
        if hits:
            preview = ", ".join(f"{hit.residue_id}({hit.probability:.2f})" for hit in hits[:max_items])
            if len(hits) > max_items:
                preview += f", ... ({len(hits)} total)"
            lines.append(preview)
        return "\n".join(lines)

    @staticmethod
    def _split_residue_id(residue_id: str) -> tuple[str, str]:
        if "_" not in residue_id:
            return "", residue_id
        chain_id, residue_number = residue_id.split("_", 1)
        return chain_id, residue_number


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
        esm_extractor: ESMFeatureExtractor | None = None,
        pca_reducer: PCAReducer | None = None,
        structure_parser: StructureParser | None = None,
        model: EvoPointDALitModule | None = None,
    ) -> None:
        self.device = self._resolve_device(device)
        self.max_len = max_len
        self.structure_parser = structure_parser or StructureParser()
        self.esm_extractor = esm_extractor or ESMFeatureExtractor(self._require_path(esm_weights, "esm_weights"), self.device)
        self.pca_reducer = pca_reducer or self._load_pca(self._require_path(pca_path, "pca_path"), pca_dim)
        self.model = model or self._load_model(self._require_path(ckpt_path, "ckpt_path"))

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
    ) -> "ProtCrossPredictor":
        return cls(
            ckpt_path=ckpt_path,
            esm_weights=esm_weights,
            pca_path=pca_path,
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
        )

    @classmethod
    def from_assets(
        cls,
        assets: PredictorAssets,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
    ) -> "ProtCrossPredictor":
        return cls.from_files(
            ckpt_path=assets.checkpoint,
            esm_weights=assets.esm_weights,
            pca_path=assets.pca,
            device=device,
            pca_dim=pca_dim,
            max_len=max_len,
        )

    @classmethod
    def from_env(
        cls,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
    ) -> "ProtCrossPredictor":
        return cls.from_assets(PredictorAssets.from_env(), device=device, pca_dim=pca_dim, max_len=max_len)

    @classmethod
    def from_default_assets(
        cls,
        *,
        device: str | None = None,
        pca_dim: int = 128,
        max_len: int = MAX_ESM_RESIDUES,
    ) -> "ProtCrossPredictor":
        return cls.from_assets(PredictorAssets.from_default_dir(), device=device, pca_dim=pca_dim, max_len=max_len)

    def predict(
        self,
        pdb_file: str | Path,
        *,
        chain_id: Optional[str] = None,
        threshold: float = 0.5,
        output_pdb: str | Path | None = None,
        scores_tsv: str | Path | None = None,
    ) -> PredictionResult:
        pdb_file = Path(pdb_file)
        if not pdb_file.exists():
            raise FileNotFoundError(f"Input structure not found: {pdb_file}")

        parsed = self.structure_parser.parse_file_with_labels(pdb_file, chain_id=chain_id)
        if not parsed:
            raise ValueError(f"No standard amino-acid residues with CA atoms found in {pdb_file}")

        parsed = truncate_parsed_structure(parsed, self.max_len)
        features = self._featurize(parsed["sequence"])
        data = Data(
            x=features,
            pos=torch.from_numpy(parsed["coords"]),
            batch=torch.zeros(len(parsed["coords"]), dtype=torch.long),
        ).to(self.device)

        probabilities = self._infer(data)
        result = PredictionResult(
            input_pdb=pdb_file,
            residue_ids=list(parsed["residue_ids"]),
            probabilities=probabilities,
            threshold=threshold,
            truncated=bool(parsed.get("truncated", False)),
            original_length=parsed.get("original_length"),
        )

        if output_pdb:
            result.write_pdb(output_pdb)
        if scores_tsv:
            result.write_scores_tsv(scores_tsv)

        return result

    def _featurize(self, sequence: str) -> torch.Tensor:
        raw_embeddings = self.esm_extractor.extract_residue_embeddings(sequence)
        reduced_embeddings = self.pca_reducer.transform(raw_embeddings)
        return reduced_embeddings.float()

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
    def _require_path(value: str | Path | None, name: str) -> Path:
        if value is None:
            raise ValueError(f"{name} is required.")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")
        return path


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
    max_len: int = MAX_ESM_RESIDUES,
) -> PredictionResult:
    ckpt_path, esm_weights, pca_path = _resolve_predict_pdb_assets(
        ckpt_path,
        esm_weights,
        pca_path,
        assets_dir=assets_dir,
    )
    predictor = ProtCrossPredictor.from_files(
        ckpt_path=ckpt_path,
        esm_weights=esm_weights,
        pca_path=pca_path,
        device=device,
        max_len=max_len,
    )
    return predictor.predict(
        pdb_file,
        chain_id=chain_id,
        threshold=threshold,
        output_pdb=output_pdb,
        scores_tsv=scores_tsv,
    )


def _resolve_predict_pdb_assets(
    ckpt_path: str | Path | None,
    esm_weights: str | Path | None,
    pca_path: str | Path | None,
    *,
    assets_dir: str | Path | None = None,
) -> tuple[str | Path, str | Path, str | Path]:
    assets: PredictorAssets | None = None
    if assets_dir:
        assets = PredictorAssets.from_dir(assets_dir)
    elif not (ckpt_path and esm_weights and pca_path):
        required_env = ("PROTCROSS_CHECKPOINT", "PROTCROSS_ESM_WEIGHTS", "PROTCROSS_PCA")
        if all(os.environ.get(name) for name in required_env):
            assets = PredictorAssets.from_env()
        else:
            default_assets = PredictorAssets.from_default_dir()
            if default_assets.is_complete():
                assets = default_assets

    if assets:
        ckpt_path = ckpt_path or assets.checkpoint
        esm_weights = esm_weights or assets.esm_weights
        pca_path = pca_path or assets.pca

    missing = [
        name
        for name, value in (
            ("ckpt_path", ckpt_path),
            ("esm_weights", esm_weights),
            ("pca_path", pca_path),
        )
        if value is None
    ]
    if missing:
        raise ValueError(
            "Missing prediction assets: "
            f"{', '.join(missing)}. Run `protcross setup-assets`, pass assets_dir, "
            "or provide ckpt_path, esm_weights, and pca_path explicitly."
        )

    return ckpt_path, esm_weights, pca_path
