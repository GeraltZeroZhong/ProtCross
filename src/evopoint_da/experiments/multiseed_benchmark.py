"""Multi-seed ablation benchmark used for paper reproduction."""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from collections import defaultdict

import numpy as np


SEEDS = [42, 2025, 1]
MAX_EPOCHS = 70
EXPERIMENTS = [
    {"id": "A", "name": "Baseline (Pure Geometry)", "args": "model.use_esm=False model.use_da=False model.feature_dim=128"},
    {"id": "B", "name": "ProtCross (No DA)", "args": "model.use_esm=True model.use_da=False"},
    {"id": "C", "name": "ProtCross (Standard DANN)", "args": "model.use_esm=True model.use_da=True model.use_plddt_weight=False"},
    {"id": "D", "name": "ProtCross (Confidence-Aware)", "args": "model.use_esm=True model.use_da=True model.use_plddt_weight=True"},
]


def clean_checkpoints(checkpoint_dir: str = "checkpoints") -> None:
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)


def backup_checkpoints(exp_id: str, seed: int, checkpoint_dir: str = "checkpoints", output_root: str = "saved_weights") -> None:
    backup_dir = os.path.join(output_root, f"{exp_id}_{seed}")
    os.makedirs(backup_dir, exist_ok=True)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    for checkpoint in checkpoints:
        shutil.copy(checkpoint, backup_dir)
        print(f"Backup: {checkpoint} -> {backup_dir}/")
    if not checkpoints:
        print(f"Warning: no checkpoints found for experiment {exp_id}, seed {seed}.")


def run_command(command: str, log_file: str) -> str:
    print(f"Exec: {command}")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    output_buffer = ""
    with open(log_file, "w", encoding="utf-8") as file:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            file.write(line)
            output_buffer += line
        process.wait()
        if process.returncode != 0:
            print(f"Command failed with return code {process.returncode}")
            print("Last 5 log lines:")
            print("   " + "\n   ".join(output_buffer.splitlines()[-5:]))
    return output_buffer


def parse_metrics(output: str) -> dict[str, float]:
    metrics = {"Overall_IoU": 0.0, "Best_Threshold": 0.0}
    if "<<<METRICS_START>>>" not in output:
        return metrics
    try:
        block = output.split("<<<METRICS_START>>>")[1].split("<<<METRICS_END>>>")[0]
        for line in block.strip().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip().replace("%", ""))
            except ValueError:
                continue
    except Exception as exc:
        print(f"Warning: failed to parse metrics: {exc}")
    return metrics


def run_multiseed_benchmark() -> dict[str, list[dict[str, float]]]:
    results = defaultdict(list)
    print("=" * 60)
    print("Multi-seed benchmark")
    print("=" * 60)

    total_runs = len(SEEDS) * len(EXPERIMENTS)
    current_run = 0
    for seed in SEEDS:
        print(f"\nStarting seed {seed}\n")
        for experiment in EXPERIMENTS:
            current_run += 1
            exp_id = experiment["id"]
            print(f"[{current_run}/{total_runs}] Running experiment {exp_id} with seed {seed}.")

            clean_checkpoints()
            train_log = f"logs/benchmark/train_{exp_id}_seed_{seed}.txt"
            train_command = " ".join(
                [
                    "python train.py",
                    experiment["args"],
                    f"seed={seed}",
                    f"trainer.max_epochs={MAX_EPOCHS}",
                    f"+trainer.default_root_dir=logs/benchmark/{exp_id}_{seed}",
                    "trainer.accelerator=gpu",
                    "data.batch_size=16",
                ]
            )
            run_command(train_command, train_log)

            test_log = f"logs/benchmark/test_{exp_id}_seed_{seed}.txt"
            test_output = run_command("python test_adaptive.py", test_log)
            backup_checkpoints(exp_id, seed)

            metrics = parse_metrics(test_output)
            results[exp_id].append(metrics)
            print(
                f"Experiment {exp_id}, seed {seed}: "
                f"IoU={metrics['Overall_IoU']}%, threshold={metrics['Best_Threshold']}"
            )

    print_report(results)
    return dict(results)


def print_report(results: dict[str, list[dict[str, float]]]) -> None:
    print("\n\n" + "=" * 80)
    print("Final benchmark report")
    print("=" * 80)

    def format_stat(exp_id: str, key: str) -> str:
        values = [entry[key] for entry in results[exp_id]]
        if not values:
            return "N/A"
        return f"{np.mean(values):.2f} +/- {np.std(values):.2f}"

    print("\n### Ablation Study")
    print("| ID | Model | IoU (%) | Best Threshold |")
    print("|---|---|---|---|")
    for experiment in EXPERIMENTS:
        exp_id = experiment["id"]
        print(f"| {exp_id} | {experiment['name']} | **{format_stat(exp_id, 'Overall_IoU')}** | {format_stat(exp_id, 'Best_Threshold')} |")
    print("\nDone. Weights were saved under 'saved_weights/'.")


def main() -> int:
    run_multiseed_benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

