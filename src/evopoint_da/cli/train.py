"""Hydra CLI entry point for standard ProtCross training."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig
from pathlib import Path

from evopoint_da.training import run_training

CONFIG_PATH = str(Path(__file__).resolve().parents[3] / "configs")

@hydra.main(version_base="1.3", config_path=CONFIG_PATH, config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
