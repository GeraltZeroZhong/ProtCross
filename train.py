"""Backward-compatible training entry point.

Prefer ``protcross-train`` after installing the package.
"""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.training import run_training


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()

