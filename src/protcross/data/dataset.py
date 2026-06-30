"""PyG datasets for preprocessed ProtCross point-cloud files."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm


PREPROCESS_MANIFEST = "protcross-preprocess-manifest.json"


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
        super().__init__(str(root))
        self._validate_preprocess_manifest()

        if not os.path.exists(self.processed_paths[0]):
            self.process()
        elif self._cache_is_stale():
            print(f"[Dataset] Cache is stale for split '{self.split}', rebuilding.")
            self.process()

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        suffix = "" if self.require_labels and self.require_positive_labels else "_all"
        return [f"data_cache_{self.split}{suffix}.pt", f"data_cache_{self.split}{suffix}.manifest.json"]

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
        torch.save((data, slices), self.processed_paths[0])
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
        if num_files < 10:
            return raw_files

        train_end = int(num_files * 0.8)
        val_end = int(num_files * 0.9)

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
        return manifest.get("signature") != self._cache_signature(self._split_files())

    def _validate_preprocess_manifest(self) -> None:
        manifest_path = Path(self.root) / PREPROCESS_MANIFEST
        if not manifest_path.exists():
            return
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Invalid preprocessing manifest: {manifest_path}") from exc
        if manifest.get("append_mode"):
            return
        allowed = set(manifest.get("produced_outputs") or manifest.get("expected_outputs") or [])
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
            "schema_version": "protcross-dataset-cache-v1",
            "split": self.split,
            "require_labels": self.require_labels,
            "require_positive_labels": self.require_positive_labels,
            "split_seed": self.split_seed,
            "signature": self._cache_signature(raw_files),
        }
        Path(self.processed_paths[1]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _cache_signature(self, raw_files: list[str]) -> dict:
        all_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        return {
            "all_files": [self._file_signature(path) for path in all_files],
            "split_files": [self._file_signature(path) for path in raw_files],
        }

    @staticmethod
    def _file_signature(path: str) -> dict[str, int | str]:
        stat = os.stat(path)
        return {
            "path": os.path.basename(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "sha256": EvoPointDataset._file_sha256(path),
        }

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_data_file(self, file_path: str) -> Data | None:
        try:
            raw = torch.load(file_path, weights_only=False)
            if "x" not in raw or "pos" not in raw:
                return None

            y = raw.get("y")
            if y is None:
                if self.require_labels:
                    return None
                y = torch.zeros(len(raw["pos"]), dtype=torch.float)

            y_tensor = self._to_tensor(y)
            if self.require_positive_labels and y_tensor.sum() == 0:
                return None

            plddt = raw.get("plddt")
            if plddt is None:
                plddt = torch.ones(len(raw["pos"]), dtype=torch.float)

            return Data(
                x=self._to_tensor(raw["x"]),
                pos=self._to_tensor(raw["pos"]),
                plddt=self._to_tensor(plddt),
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
