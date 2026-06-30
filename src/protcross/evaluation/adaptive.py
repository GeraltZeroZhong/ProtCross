"""Adaptive threshold evaluation for ProtCross checkpoints."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from protcross.data.dataset import EvoPointDataset
from protcross.evaluation.metrics import calculate_plddt_metrics, compute_best_iou
from protcross.models import EvoPointDALitModule


def find_checkpoint(project_root: str | Path, configured_ckpt: str | None = None) -> str | None:
    project_root = Path(project_root)
    if configured_ckpt:
        path = Path(configured_ckpt)
        return str(path if path.is_absolute() else project_root / path)

    candidates = []
    for checkpoint_dir in (project_root / "checkpoints", project_root / "checkpoint"):
        candidates.extend(glob.glob(str(checkpoint_dir / "best-*.ckpt")))
        candidates.extend(glob.glob(str(checkpoint_dir / "last.ckpt")))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def evaluate_adaptive(cfg: DictConfig, *, project_root: str | Path | None = None) -> dict[str, float | int]:
    if project_root is None:
        try:
            project_root = hydra.utils.get_original_cwd()
        except Exception:
            project_root = os.getcwd()

    checkpoint = find_checkpoint(project_root, cfg.get("ckpt_path"))
    if not checkpoint:
        raise FileNotFoundError("No checkpoints found. Set ckpt_path=/path/to/model.ckpt.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Loading checkpoint: {checkpoint}")
    model = EvoPointDALitModule.load_from_checkpoint(checkpoint, map_location=device)
    model.eval().to(device)

    if "data_dir_af2" in cfg.data and cfg.data.data_dir_af2:
        test_path = cfg.data.data_dir_af2
        domain_name = "AlphaFold2 (target)"
    else:
        test_path = cfg.data.data_dir_pdb
        domain_name = "PDB (source)"

    if not os.path.isabs(test_path):
        test_path = os.path.join(project_root, test_path)
    print(f"[Eval] Domain: {domain_name}")
    print(f"[Eval] Path: {test_path}")

    try:
        dataset = EvoPointDataset(root=test_path, split="test")
    except Exception as exc:
        evaluation_cfg = cfg.get("evaluation", {}) or {}
        allow_train_fallback = bool(evaluation_cfg.get("allow_train_fallback", False))
        if not allow_train_fallback:
            raise RuntimeError(
                "Test split unavailable for evaluation. Fix the processed test data or set "
                "evaluation.allow_train_fallback=true for legacy exploratory runs."
            ) from exc
        print("[Eval] Test split unavailable; falling back to train split by explicit configuration.")
        dataset = EvoPointDataset(root=test_path, split="train")

    loader = DataLoader(
        dataset,
        batch_size=cfg.data.get("batch_size", 1),
        shuffle=False,
        num_workers=cfg.data.get("num_workers", 2),
    )

    labels, probabilities, plddts = [], [], []
    print("[Eval] Running inference.")
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels.append(batch.y.cpu().numpy())
            probabilities.append(probs.cpu().numpy())
            plddts.append(batch.plddt.cpu().numpy().flatten())

    y_true = np.concatenate(labels)
    y_probs = np.concatenate(probabilities)
    y_plddt = np.concatenate(plddts)

    best_iou, best_threshold = compute_best_iou(y_true, y_probs)
    detailed = calculate_plddt_metrics(y_true, y_probs, y_plddt, best_threshold)
    return {
        "Overall_IoU": best_iou * 100,
        "Best_Threshold": best_threshold,
        **detailed,
    }


def print_metrics_block(metrics: dict[str, float | int]) -> None:
    print("\n<<<METRICS_START>>>")
    print(f"Overall_IoU: {metrics['Overall_IoU']:.2f}%")
    print(f"Best_Threshold: {metrics['Best_Threshold']:.2f}")
    print(f"High_IoU: {metrics['High_IoU']:.2f}%")
    print(f"High_Total: {metrics['High_Total']}")
    print(f"Med_IoU: {metrics['Med_IoU']:.2f}%")
    print(f"Med_Total: {metrics['Med_Total']}")
    print(f"Low_FPR: {metrics['Low_FPR']:.2f}%")
    print(f"Low_Total: {metrics['Low_Total']}")
    print("<<<METRICS_END>>>")
