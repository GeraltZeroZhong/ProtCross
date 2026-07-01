"""Training orchestration shared by CLI wrappers."""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger


def run_training(cfg: DictConfig) -> None:
    seed_everything(cfg.seed, workers=True)

    try:
        project_root = hydra.utils.get_original_cwd()
    except Exception:
        project_root = os.getcwd()

    print(f"[Train] Project Root: {project_root}")

    if not os.path.isabs(cfg.data.data_dir_pdb):
        cfg.data.data_dir_pdb = os.path.join(project_root, cfg.data.data_dir_pdb)
    if not os.path.isabs(cfg.data.data_dir_af2):
        cfg.data.data_dir_af2 = os.path.join(project_root, cfg.data.data_dir_af2)
    _preflight_training_data(cfg)

    print(f"[Train] Instantiating DataModule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    print(f"[Train] Instantiating Model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    output_root = str(cfg.get("paths", {}).get("output_dir", os.path.join(project_root, "outputs")))
    if not os.path.isabs(output_root):
        output_root = os.path.join(project_root, output_root)
    output_root = os.path.abspath(output_root)

    checkpoint_dir = os.path.join(output_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"[Train] Checkpoints will be saved to: {checkpoint_dir}")

    monitor_metric = "val/loss" if cfg.data.get("val_split") else "train/loss_total"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best-{epoch:02d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )
    print(f"[Train] Best checkpoint monitor: {monitor_metric}")
    logger = CSVLogger(save_dir=os.path.join(output_root, "logs"), name="protcross")

    print(f"[Train] Instantiating Trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[checkpoint_callback, RichProgressBar()],
        logger=logger,
        enable_checkpointing=True,
    )

    print("[Train] Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    print(f"[Train] Done. Checkpoints saved in {checkpoint_dir}")


def _preflight_training_data(cfg: DictConfig) -> None:
    source_dir = Path(str(cfg.data.data_dir_pdb))
    if not source_dir.exists():
        raise FileNotFoundError(f"Training source data directory not found: {source_dir}")
    if not any(source_dir.glob("*.pt")):
        raise FileNotFoundError(f"No source .pt files found in {source_dir}")

    if bool(cfg.data.get("use_target_domain")):
        target_dir = Path(str(cfg.data.data_dir_af2))
        if not target_dir.exists():
            raise FileNotFoundError(f"Target-domain data directory not found: {target_dir}")
        if not any(target_dir.glob("*.pt")):
            raise FileNotFoundError(
                f"Target-domain training is enabled but no AF2 .pt files were found in {target_dir}. "
                "Disable data.use_target_domain or preprocess target samples."
            )
