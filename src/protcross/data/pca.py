"""PCA helpers for ESM-C residue embeddings."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import torch
from sklearn.decomposition import PCA


class PCAReducer:
    def __init__(self, n_components: int = 128, random_state: int | None = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.is_fitted = False

    def fit(self, data_list: List[torch.Tensor]) -> None:
        x = torch.cat(data_list, dim=0).numpy()
        self.pca.fit(x)
        self.is_fitted = True

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA reducer is not fitted.")
        return torch.from_numpy(self.pca.transform(x.numpy())).float()

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self.pca, file)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as file:
            self.pca = pickle.load(file)
        actual_components = getattr(
            self.pca,
            "n_components_",
            getattr(self.pca, "n_components", self.n_components),
        )
        self.n_components = int(actual_components)
        self.is_fitted = True
