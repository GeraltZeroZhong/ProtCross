"""PyG datasets for preprocessed ProtCross point-cloud files."""

from __future__ import annotations

import glob
import json
import os
import random
import uuid
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


PREPROCESS_MANIFEST = "protcross-preprocess-manifest.json"
DATASET_CACHE_SCHEMA = "protcross-dataset-cache-v3"


class EvoPointDataset(InMemoryDataset):
    def __init__(
        self,
        root: str | os.PathLike,
        split: str = "train",
        augment: bool = False,
        *,
        require_labels: bool = True,
        require_positive_labels: bool = True,
        split_seed: int = 42,
    ) -> None:
        self.split = split
        self.augment = augment
        self.require_labels = require_labels
        self.require_positive_labels = require_positive_labels
        self.split_seed = split_seed
        # Reject an interrupted preprocessing run before PyG can build and
        # persist a cache from its partial output directory.
        self.root = str(root)
        self._validate_preprocess_manifest()
        super().__init__(str(root))

        if not os.path.exists(self.processed_paths[0]):
            self.process()
        elif self._cache_is_stale():
            print(f"[Dataset] Cache is stale for split '{self.split}', rebuilding.")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        if self.require_labels and self.require_positive_labels:
            filter_suffix = ""
        elif self.require_labels:
            filter_suffix = "_labeled"
        elif self.require_positive_labels:
            filter_suffix = "_positive"
        else:
            filter_suffix = "_all"
        seed_suffix = "" if self.split_seed == 42 else f"_seed{self.split_seed}"
        stem = f"data_cache_{self.split}{filter_suffix}{seed_suffix}"
        return [f"{stem}.pt", f"{stem}.manifest.json"]

    def _augment(self, pos: torch.Tensor) -> torch.Tensor:
        theta = np.random.uniform(0, 2 * np.pi)
        rot = torch.tensor(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=pos.dtype,
            device=pos.device,
        )
        return pos @ rot + torch.randn_like(pos) * 0.01

    def get(self, idx):
        data = super().get(idx)
        if self.augment and self.split == "train":
            data.pos = self._augment(data.pos)
        return data

    def process(self) -> None:
        raw_files = self._split_files()
        print(f"[Dataset] Processing split: '{self.split}'")
        print(f"   - Source Directory: {self.root}")
        print(f"   - Candidate Files: {len(raw_files)}")

        data_list = []
        for file_path in tqdm(raw_files, desc=f"Loading {self.split}"):
            data = self._load_data_file(file_path)
            if data is not None:
                data_list.append(data)

        if not data_list:
            raise RuntimeError(
                f"No valid data found for split '{self.split}' in {self.root}. "
                "Check preprocessing output and dataset filtering settings."
            )

        data, slices = self.collate(data_list)
        cache_path = Path(self.processed_paths[0])
        temporary_path = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.part")
        try:
            torch.save((data, slices), temporary_path)
            temporary_path.replace(cache_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        self._write_cache_manifest(raw_files)
        print(f"   - Final Valid Samples: {len(data_list)} / {len(raw_files)}")
        print(f"   - Cache saved to: {self.processed_paths[0]}")

    def _split_files(self) -> list[str]:
        raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        if self.split == "all":
            return raw_files

        rng = random.Random(self.split_seed)
        rng.shuffle(raw_files)

        num_files = len(raw_files)
        train_count = max(1, int(num_files * 0.8)) if num_files else 0
        if num_files >= 3:
            train_count = min(train_count, num_files - 2)
        elif num_files == 2:
            train_count = 1
        remaining = num_files - train_count
        val_count = min(max(1, int(num_files * 0.1)), remaining - 1) if remaining >= 2 else remaining
        train_end = train_count
        val_end = train_count + val_count

        if self.split == "train":
            return raw_files[:train_end]
        if self.split == "val":
            return raw_files[train_end:val_end]
        if self.split == "test":
            return raw_files[val_end:]
        raise ValueError(f"Unknown split: {self.split}")

    def _cache_is_stale(self) -> bool:
        manifest_path = Path(self.processed_paths[1])
        if not manifest_path.exists():
            return True
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return True
        expected_config = {
            "schema_version": DATASET_CACHE_SCHEMA,
            "split": self.split,
            "require_labels": self.require_labels,
            "require_positive_labels": self.require_positive_labels,
            "split_seed": self.split_seed,
        }
        if any(manifest.get(key) != value for key, value in expected_config.items()):
            return True
        return manifest.get("signature") != self._cache_signature(self._split_files())

    def _validate_preprocess_manifest(self) -> None:
        manifest_path = Path(self.root) / PREPROCESS_MANIFEST
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid preprocessing manifest: {manifest_path}") from exc
        if manifest.get("complete") is False:
            raise RuntimeError(
                f"Preprocessing manifest is incomplete: {manifest_path}. "
                "Re-run protcross preprocess successfully before loading this dataset."
            )
        if manifest.get("append_mode"):
            return
        if "produced_outputs" in manifest:
            allowed = set(manifest.get("produced_outputs") or [])
        else:
            allowed = set(manifest.get("expected_outputs") or [])
        if not allowed:
            raise RuntimeError(f"Preprocessing manifest has no expected outputs: {manifest_path}")
        actual = {Path(path).name for path in glob.glob(os.path.join(self.root, "*.pt"))}
        orphaned = sorted(actual - allowed)
        if orphaned:
            raise RuntimeError(
                "Processed data directory contains .pt files not listed in "
                f"{PREPROCESS_MANIFEST}: {', '.join(orphaned)}. "
                "Re-run protcross preprocess without --append or remove stale files."
            )

    def _write_cache_manifest(self, raw_files: list[str]) -> None:
        manifest = {
            "schema_version": DATASET_CACHE_SCHEMA,
            "split": self.split,
            "require_labels": self.require_labels,
            "require_positive_labels": self.require_positive_labels,
            "split_seed": self.split_seed,
            "signature": self._cache_signature(raw_files),
        }
        manifest_path = Path(self.processed_paths[1])
        temporary_path = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.part")
        try:
            temporary_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            temporary_path.replace(manifest_path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _cache_signature(self, raw_files: list[str]) -> dict:
        all_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        signatures_by_path = {
            os.path.abspath(path): self._file_signature(path)
            for path in dict.fromkeys([*all_files, *raw_files])
        }
        return {
            "all_files": [signatures_by_path[os.path.abspath(path)] for path in all_files],
            "split_files": [signatures_by_path[os.path.abspath(path)] for path in raw_files],
        }

    @staticmethod
    def _file_signature(path: str) -> dict[str, int | str]:
        stat = os.stat(path)
        return {
            "path": os.path.basename(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }

    def _load_data_file(self, file_path: str) -> Data | None:
        try:
            raw = torch.load(file_path, weights_only=False)
            if "x" not in raw or "pos" not in raw:
                return None

            x_tensor = self._to_tensor(raw["x"])
            pos_tensor = self._to_tensor(raw["pos"])
            if pos_tensor.ndim != 2 or pos_tensor.shape[1] != 3:
                raise ValueError(f"pos must have shape (N, 3); got {tuple(pos_tensor.shape)}")
            residue_count = int(pos_tensor.shape[0])
            if residue_count < 1:
                raise ValueError("pos must contain at least one residue")
            if x_tensor.ndim != 2 or x_tensor.shape[0] != residue_count or x_tensor.shape[1] < 1:
                raise ValueError(
                    f"x must have shape ({residue_count}, D) with D >= 1; got {tuple(x_tensor.shape)}"
                )
            if not torch.isfinite(pos_tensor).all() or not torch.isfinite(x_tensor).all():
                raise ValueError("x and pos must contain only finite values")

            y = raw.get("y")
            if y is None:
                if self.require_labels:
                    return None
                y = torch.zeros(residue_count, dtype=torch.float)

            y_tensor = self._to_tensor(y)
            if y_tensor.ndim != 1 or y_tensor.shape[0] != residue_count:
                raise ValueError(f"y must have shape ({residue_count},); got {tuple(y_tensor.shape)}")
            if not torch.isfinite(y_tensor).all():
                raise ValueError("y must contain only finite values")
            if self.require_positive_labels and y_tensor.sum() == 0:
                return None

            plddt = raw.get("plddt")
            if plddt is None:
                plddt = torch.ones(residue_count, dtype=torch.float)
            plddt_tensor = self._to_tensor(plddt)
            if plddt_tensor.shape not in {(residue_count,), (residue_count, 1)}:
                raise ValueError(
                    f"plddt must have shape ({residue_count},) or ({residue_count}, 1); "
                    f"got {tuple(plddt_tensor.shape)}"
                )
            if not torch.isfinite(plddt_tensor).all():
                raise ValueError("plddt must contain only finite values")

            return Data(
                x=x_tensor,
                pos=pos_tensor,
                plddt=plddt_tensor,
                y=y_tensor,
                protein_id=Path(file_path).stem,
            )
        except Exception as exc:
            print(f"Warning: error loading {file_path}: {exc}")
            return None

    @staticmethod
    def _to_tensor(value) -> torch.Tensor:
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).float()
        if isinstance(value, torch.Tensor):
            return value.float()
        return torch.tensor(value).float()
