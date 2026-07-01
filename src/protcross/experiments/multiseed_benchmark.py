"""Multi-seed ablation benchmark used for paper reproduction."""

from __future__ import annotations

import glob
import os
import shlex
import shutil
import signal
import subprocess
import sys
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
    args = shlex.split(command)
    with open(log_file, "w", encoding="utf-8") as file:
        process = subprocess.Popen(
            args,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=(sys.platform != "win32"),
            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0),
        )
        try:
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
                raise subprocess.CalledProcessError(process.returncode, args, output=output_buffer)
        except BaseException:
            _terminate_process(process)
            raise
    return output_buffer


def _terminate_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


def parse_metrics(output: str) -> dict[str, float]:
    if "<<<METRICS_START>>>" not in output:
        raise RuntimeError("Evaluation output did not contain a metrics block.")
    metrics = {"Overall_IoU": 0.0, "Best_Threshold": 0.0}
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


def find_run_checkpoint(output_dir: str) -> str:
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "best-*.ckpt")))
    if not candidates:
        candidates = sorted(glob.glob(os.path.join(checkpoint_dir, "last.ckpt")))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint produced under {checkpoint_dir}")
    checkpoint = max(candidates, key=os.path.getmtime)
    if not os.path.abspath(checkpoint).startswith(os.path.abspath(checkpoint_dir) + os.sep):
        raise RuntimeError(f"Resolved checkpoint is outside the current run directory: {checkpoint}")
    return checkpoint


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

            train_log = f"logs/benchmark/train_{exp_id}_seed_{seed}.txt"
            run_output_dir = f"outputs/benchmark/{exp_id}_{seed}"
            train_command = " ".join(
                [
                    "protcross train",
                    experiment["args"],
                    f"seed={seed}",
                    f"trainer.max_epochs={MAX_EPOCHS}",
                    f"paths.output_dir={run_output_dir}",
                    "trainer.accelerator=gpu",
                    "data.batch_size=16",
                ]
            )
            run_command(train_command, train_log)
            run_checkpoint = find_run_checkpoint(run_output_dir)

            test_log = f"logs/benchmark/test_{exp_id}_seed_{seed}.txt"
            test_output = run_command(
                f"python reproduction/legacy/test_adaptive.py ckpt_path={run_checkpoint}",
                test_log,
            )
            backup_checkpoints(exp_id, seed, checkpoint_dir=os.path.dirname(run_checkpoint))

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
