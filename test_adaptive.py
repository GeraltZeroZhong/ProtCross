"""Backward-compatible adaptive evaluation entry point."""

import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.evaluation.adaptive import evaluate_adaptive, print_metrics_block


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    metrics = evaluate_adaptive(cfg)
    print_metrics_block(metrics)


if __name__ == "__main__":
    main()

