"""Lightning data module for ProtCross training."""

from __future__ import annotations

import glob
import os
from typing import Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from .dataset import EvoPointDataset


class EvoPointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir_pdb: str = "./data/processed_pdb",
        data_dir_af2: str = "./data/processed_af2",
        batch_size: int = 16,
        num_workers: int = 0,
        use_target_domain: bool = True,
        train_split: str = "train",
        target_split: str = "train",
        val_split: str | None = "val",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.train_set_pdb = None
        self.train_set_af2 = None
        self.val_set = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in ("fit", None):
            self.train_set_pdb = EvoPointDataset(
                root=self.hparams.data_dir_pdb,
                split=self.hparams.train_split,
                augment=True,
                require_labels=True,
                require_positive_labels=True,
            )
            if self.hparams.val_split:
                self.val_set = EvoPointDataset(
                    root=self.hparams.data_dir_pdb,
                    split=self.hparams.val_split,
                    require_labels=True,
                    require_positive_labels=True,
                )
            if self.hparams.use_target_domain and self._has_pt_files(self.hparams.data_dir_af2):
                self.train_set_af2 = EvoPointDataset(
                    root=self.hparams.data_dir_af2,
                    split=self.hparams.target_split,
                    augment=True,
                    require_labels=False,
                    require_positive_labels=False,
                )

    def train_dataloader(self):
        loaders = {
            "source": DataLoader(
                self.train_set_pdb,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
            )
        }
        if self.train_set_af2 is not None:
            loaders["target"] = DataLoader(
                self.train_set_af2,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
            )
        return loaders

    def val_dataloader(self):
        if self.val_set is None:
            return None
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def _has_pt_files(path: str) -> bool:
        return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "*.pt")))
