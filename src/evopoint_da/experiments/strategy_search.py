"""Confidence weighting strategy search used for paper reproduction."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.data.dataset import EvoPointDataset
from evopoint_da.evaluation.metrics import compute_best_iou
from evopoint_da.models import EvoPointDALitModule


SEEDS = [42, 2025, 1]
MAX_EPOCHS = 70
BATCH_SIZE = 16
DA_WEIGHT = 0.2
STRATEGY_GRID = {
    "Baselines": ["Standard_DANN"],
    "Polynomial": ["pLDDT^1", "pLDDT^2", "pLDDT^3", "pLDDT^4", "pLDDT^5", "pLDDT^6"],
    "Hard_Threshold": ["threshold_0.5", "threshold_0.6", "threshold_0.7", "threshold_0.8", "threshold_0.9"],
    "Sigmoid": ["sigmoid_0.6_20", "sigmoid_0.7_20", "sigmoid_0.8_20", "sigmoid_0.7_10", "sigmoid_0.7_30"],
    "Exponential": ["exponential_2", "exponential_4", "exponential_6", "exponential_8"],
    "Trigonometric": ["sine", "cosine"],
}


def evaluate_iou(ckpt_path: str, test_folder: str, device: str = "cuda") -> tuple[float, float]:
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as exc:
        print(f"Warning: model load failed for {ckpt_path}: {exc}")
        return 0.0, 0.0

    try:
        dataset = EvoPointDataset(root=test_folder, split="test")
    except Exception:
        dataset = EvoPointDataset(root=test_folder, split="train")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            labels.append(batch.y.cpu().numpy())
            probabilities.append(probs.cpu().numpy())

    if not labels:
        return 0.0, 0.0

    best_iou, best_threshold = compute_best_iou(np.concatenate(labels), np.concatenate(probabilities))
    return best_iou * 100, best_threshold


def run_strategy_search() -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir_pdb = os.path.abspath("data/processed_pdb")
    data_dir_af2 = os.path.abspath("data/processed_af2")
    logs_root = "logs/grand_comparison_multiseed"
    tasks = [(family, strategy) for family, strategies in STRATEGY_GRID.items() for strategy in strategies]
    results = []

    print("Multi-seed strategy analysis starting.")
    print(f"Seeds: {SEEDS}")
    print(f"Configurations per seed: {len(tasks)}")

    for seed in SEEDS:
        print(f"\nRunning seed {seed}.")
        for index, (family, strategy) in enumerate(tasks, start=1):
            seed_everything(seed, workers=True)
            print(f"\n[Seed {seed}] [{index}/{len(tasks)}] Strategy: {strategy}")

            exp_name = f"{strategy.replace('^', '').replace('>', 'gt')}_seed{seed}"
            ckpt_dir = os.path.join(logs_root, exp_name, "checkpoints")
            best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
            use_plddt_weight = strategy != "Standard_DANN"

            if os.path.exists(best_ckpt):
                print("Found existing checkpoint; skipping training.")
            else:
                datamodule = EvoPointDataModule(data_dir_pdb, data_dir_af2, BATCH_SIZE, 2)
                model = EvoPointDALitModule(
                    weight_strategy=strategy,
                    lr=3e-4,
                    weight_decay=1e-4,
                    da_weight=DA_WEIGHT,
                    feature_dim=128,
                    pos_noise=0.08,
                    use_esm=True,
                    use_da=True,
                    use_plddt_weight=use_plddt_weight,
                )
                logger = CSVLogger(save_dir=logs_root, name=exp_name)
                checkpoint = ModelCheckpoint(
                    dirpath=ckpt_dir,
                    filename="best",
                    monitor="train/loss_total",
                    mode="min",
                    save_top_k=1,
                )
                trainer = pl.Trainer(
                    max_epochs=MAX_EPOCHS,
                    accelerator=device,
                    devices=1,
                    logger=logger,
                    callbacks=[checkpoint],
                    enable_checkpointing=True,
                    log_every_n_steps=5,
                )
                trainer.fit(model, datamodule=datamodule)
                best_ckpt = checkpoint.best_model_path

            if best_ckpt and os.path.exists(best_ckpt):
                iou, threshold = evaluate_iou(best_ckpt, data_dir_af2, device)
                print(f"Result: IoU={iou:.2f}% (threshold={threshold:.2f})")
                results.append({"Family": family, "Strategy": strategy, "Seed": seed, "IoU": iou, "Best_Thresh": threshold})
            else:
                print("Failed: checkpoint not found.")

    if not results:
        return pd.DataFrame()

    raw = pd.DataFrame(results)
    raw.to_csv("multiseed_raw_results.csv", index=False)
    summary = (
        raw.groupby(["Family", "Strategy"])
        .agg(Mean_IoU=("IoU", "mean"), Std_IoU=("IoU", "std"), Min_IoU=("IoU", "min"), Max_IoU=("IoU", "max"))
        .reset_index()
        .sort_values("Mean_IoU", ascending=False)
    )
    print("\n" + "=" * 60)
    print("Aggregated report (mean +/- std)")
    print("=" * 60)
    print(summary.to_string(float_format="%.2f"))
    summary.to_csv("multiseed_summary_report.csv", index=False)
    print("All strategy runs completed. Reports saved.")
    return summary


def main() -> int:
    run_strategy_search()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

