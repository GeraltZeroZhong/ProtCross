import subprocess
import os
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict
import glob  # 新增: 用于查找文件

SEEDS = [42, 2025, 1]
MAX_EPOCHS = 100

EXPERIMENTS = [
    {
        "id": "A",
        "name": "Baseline (Pure Geom)",
        "args": "model.use_esm=False model.use_da=False model.feature_dim=128"
    },
    {
        "id": "B",
        "name": "Ours (No DA)",
        "args": "model.use_esm=True model.use_da=False"
    },
    {
        "id": "C",
        "name": "Ours (Standard DANN)",
        "args": "model.use_esm=True model.use_da=True model.use_plddt_weight=False"
    },
    {
        "id": "D",
        "name": "Ours (Confidence-Aware)",
        "args": "model.use_esm=True model.use_da=True model.use_plddt_weight=True"
    }
]

def clean_checkpoints():
    """训练前清空临时目录，防止混淆"""
    if os.path.exists("checkpoints"):
        try:
            shutil.rmtree("checkpoints")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean checkpoints: {e}")
    os.makedirs("checkpoints", exist_ok=True)

def backup_checkpoints(exp_id, seed):
    """
    [新增功能] 将训练好的权重备份到 saved_weights/ 目录
    结构: saved_weights/A_42/best-epoch=xx.ckpt
    """
    backup_dir = os.path.join("saved_weights", f"{exp_id}_{seed}")
    os.makedirs(backup_dir, exist_ok=True)
    
    # 查找 checkpoints 目录下的所有 .ckpt 文件
    found = False
    for ckpt in glob.glob(os.path.join("checkpoints", "*.ckpt")):
        shutil.copy(ckpt, backup_dir)
        print(f"📦 Backup: {ckpt} -> {backup_dir}/")
        found = True
    
    if not found:
        print(f"⚠️ Warning: No checkpoints found to backup for Exp {exp_id} Seed {seed}")

def run_command(cmd, log_file):
    print(f"👉 Exec: {cmd}")
    output_buffer = ""
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                print(line, end="")
                f.write(line)
                output_buffer += line
            process.wait()
            if process.returncode != 0:
                print(f"❌ Command failed with return code {process.returncode}")
                # 打印最后几行错误日志
                print("   [Last 5 lines of log]")
                print("   " + "\n   ".join(output_buffer.splitlines()[-5:]))
    except Exception as e:
        print(f"❌ Execution Error: {e}")
    return output_buffer

def parse_metrics(output):
    metrics = {"Overall_IoU": 0.0, "Best_Threshold": 0.0}
    if "<<<METRICS_START>>>" in output:
        try:
            block = output.split("<<<METRICS_START>>>")[1].split("<<<METRICS_END>>>")[0]
            for line in block.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":")
                    key = k.strip()
                    val_str = v.strip().replace('%', '')
                    try:
                        metrics[key] = float(val_str)
                    except:
                        pass
        except Exception as e:
            print(f"⚠️ Error parsing metrics: {e}")
    return metrics

def main():
    results = defaultdict(list)
    print("="*60)
    print(f"🧬 EvoPoint-DA Multi-Seed Benchmark (100 Epochs, GPU Enabled)")
    print("="*60)

    total_runs = len(SEEDS) * len(EXPERIMENTS)
    current_run = 0

    for seed in SEEDS:
        print(f"\n🌱 === Starting Loop for SEED {seed} ===\n")
        
        for exp in EXPERIMENTS:
            current_run += 1
            exp_id = exp['id']
            print(f"[{current_run}/{total_runs}] Running Experiment {exp_id} (Seed {seed})...")
            
            # 1. 清理临时权重目录 (准备开始新训练)
            clean_checkpoints()
            
            log_train = f"logs/benchmark/train_{exp_id}_seed_{seed}.txt"
            os.makedirs(os.path.dirname(log_train), exist_ok=True)
            
            # 2. 训练
            train_cmd = " ".join([
                "python train.py",
                exp['args'],
                f"+seed_everything={seed}",
                f"trainer.max_epochs={MAX_EPOCHS}",
                f"+trainer.default_root_dir=logs/benchmark/{exp_id}_{seed}",
                "trainer.accelerator=cpu" # 如果有GPU请改为 gpu
            ])
            run_command(train_cmd, log_train)
            
            # 3. 测试
            log_test = f"logs/benchmark/test_{exp_id}_seed_{seed}.txt"
            test_cmd = "python test_adaptive.py" 
            test_output = run_command(test_cmd, log_test)
            
            # 4. [修改点] 备份权重！
            # 必须在 clean_checkpoints 之前 (即下一次循环开始前) 执行
            backup_checkpoints(exp_id, seed)

            # 5. 记录结果
            m = parse_metrics(test_output)
            results[exp_id].append(m)
            print(f"✅ Exp {exp_id} (Seed {seed}) Result: IoU={m['Overall_IoU']}% (Thresh={m['Best_Threshold']})")

    # === Report ===
    print("\n\n" + "="*80)
    print("🏆 FINAL BENCHMARK REPORT (100 Epochs)")
    print("="*80)

    def fmt_stat(exp_id, key):
        vals = [r[key] for r in results[exp_id]]
        if not vals: return "N/A"
        return f"{np.mean(vals):.2f} ± {np.std(vals):.2f}"

    print("\n### Table 2: Ablation Study")
    print("| ID | Model | IoU (%) | Best Threshold |")
    print("|---|---|---|---|")
    
    for exp in EXPERIMENTS:
        iou_str = fmt_stat(exp['id'], 'Overall_IoU')
        thresh_str = fmt_stat(exp['id'], 'Best_Threshold')
        print(f"| {exp['id']} | {exp['name']} | **{iou_str}** | {thresh_str} |")

    print("\n✅ Done. All weights saved to 'saved_weights/' directory.")

if __name__ == "__main__":
    main()
