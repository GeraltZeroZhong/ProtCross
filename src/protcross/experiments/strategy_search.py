"""Confidence weighting strategy search used for paper reproduction."""

from __future__ import annotations

import os
import json
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from protcross.data.datamodule import EvoPointDataModule
from protcross.data.dataset import EvoPointDataset
from protcross.evaluation.metrics import compute_best_iou
from protcross.models import EvoPointDALitModule


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


def evaluate_iou(
    ckpt_path: str,
    test_folder: str,
    device: str = "cuda",
    *,
    allow_train_fallback: bool = False,
) -> tuple[float, float]:
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path, map_location=device)
        model.eval().to(device)
    except Exception as exc:
        raise RuntimeError(f"Model load failed for {ckpt_path}: {exc}") from exc

    try:
        dataset = EvoPointDataset(root=test_folder, split="test")
    except Exception as exc:
        if not allow_train_fallback:
            raise RuntimeError(
                f"Failed to load test split from {test_folder}; refusing to evaluate on train split implicitly."
            ) from exc
        print(f"Warning: test split failed for {test_folder}: {exc}; falling back to train split.")
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
        raise RuntimeError(f"No evaluation samples found in {test_folder}.")

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
            signature = strategy_run_signature(
                strategy=strategy,
                seed=seed,
                data_dir_pdb=data_dir_pdb,
                data_dir_af2=data_dir_af2,
            )

            if os.path.exists(best_ckpt) and should_resume_checkpoint(best_ckpt, signature):
                print("Found matching checkpoint manifest; skipping training.")
            else:
                if os.path.exists(best_ckpt):
                    print("Existing checkpoint has no matching manifest; retraining.")
                    os.remove(best_ckpt)
                    manifest_path = checkpoint_manifest_path(best_ckpt)
                    if os.path.exists(manifest_path):
                        os.remove(manifest_path)
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
                    monitor="val/loss",
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
                if not best_ckpt or not os.path.exists(best_ckpt):
                    raise RuntimeError(f"Training finished without a best checkpoint for {strategy} seed {seed}.")
                write_checkpoint_manifest(best_ckpt, signature)

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


def should_resume_checkpoint(checkpoint_path: str, signature: dict) -> bool:
    if os.environ.get("PROTCROSS_STRATEGY_RESUME") != "1":
        return False
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    if not os.path.exists(manifest_path):
        return False
    try:
        manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    except Exception:
        return False
    if manifest.get("signature") != signature:
        return False
    recorded_sha = manifest.get("checkpoint_sha256")
    return bool(recorded_sha and os.path.exists(checkpoint_path) and file_sha256(checkpoint_path) == recorded_sha)


def write_checkpoint_manifest(checkpoint_path: str, signature: dict) -> None:
    manifest = {
        "schema_version": "protcross-strategy-checkpoint-v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "signature": signature,
    }
    with open(checkpoint_manifest_path(checkpoint_path), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def checkpoint_manifest_path(checkpoint_path: str) -> str:
    return checkpoint_path + ".protcross-strategy.json"


def strategy_run_signature(*, strategy: str, seed: int, data_dir_pdb: str, data_dir_af2: str) -> dict:
    return {
        "strategy": strategy,
        "seed": seed,
        "max_epochs": MAX_EPOCHS,
        "batch_size": BATCH_SIZE,
        "da_weight": DA_WEIGHT,
        "data_dir_pdb": dir_signature(data_dir_pdb),
        "data_dir_af2": dir_signature(data_dir_af2),
    }


def dir_signature(directory: str) -> list[dict]:
    if not os.path.isdir(directory):
        return []
    records = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".pt"):
            continue
        path = os.path.join(directory, name)
        stat = os.stat(path)
        records.append(
            {
                "name": name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": file_sha256(path),
            }
        )
    return records


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
