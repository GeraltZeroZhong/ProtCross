import subprocess
import os
import shutil
import numpy as np
import pandas as pd
from collections import defaultdict

# === 配置区域 ===
SEEDS = [42, 1234, 2024]
MAX_EPOCHS = 50

# 定义四个关键实验 (与 run_benchmark.py 一致)
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
    """清理 checkpoints 文件夹，防止 test 读取到旧的权重"""
    if os.path.exists("checkpoints"):
        try:
            shutil.rmtree("checkpoints")
        except Exception as e:
            print(f"⚠️ Warning: Failed to clean checkpoints: {e}")
    os.makedirs("checkpoints", exist_ok=True)

def run_command(cmd, log_file):
    """运行命令并将输出同时写入文件和控制台 (简化版输出)"""
    print(f"👉 Exec: {cmd}")
    output_buffer = ""
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in process.stdout:
                f.write(line)
                output_buffer += line
            process.wait()
            if process.returncode != 0:
                print(f"❌ Command failed with return code {process.returncode}")
    except Exception as e:
        print(f"❌ Execution Error: {e}")
    return output_buffer

def parse_metrics(output):
    """从 test.py 的输出中提取指标"""
    metrics = {
        "Overall_IoU": 0.0, 
        "High_IoU": 0.0, 
        "Med_IoU": 0.0, 
        "Low_FPR": 0.0
    }
    
    # 查找 <<<METRICS_START>>> 块 (兼容你的 test.py 输出格式)
    if "<<<METRICS_START>>>" in output:
        try:
            block = output.split("<<<METRICS_START>>>")[1].split("<<<METRICS_END>>>")[0]
            for line in block.strip().split("\n"):
                if ":" in line:
                    k, v = line.split(":")
                    key = k.strip()
                    val = float(v.strip().replace('%', '')) # 移除%并转float
                    if key in metrics:
                        metrics[key] = val
        except Exception as e:
            print(f"⚠️ Error parsing metrics: {e}")
            
    return metrics

def main():
    # 存储结构: results[exp_id] = list of dicts (one per seed)
    results = defaultdict(list)
    
    print("="*60)
    print(f"🧬 EvoPoint-DA Multi-Seed Benchmark (Seeds: {SEEDS})")
    print("="*60)

    total_runs = len(SEEDS) * len(EXPERIMENTS)
    current_run = 0

    for seed in SEEDS:
        print(f"\n🌱 === Starting Loop for SEED {seed} ===\n")
        
        for exp in EXPERIMENTS:
            current_run += 1
            exp_id = exp['id']
            print(f"[{current_run}/{total_runs}] Running Experiment {exp_id} (Seed {seed})...")
            
            # 1. 清理权重
            clean_checkpoints()
            
            # 2. 训练 (加入 +seed_everything)
            log_train = f"logs/benchmark/train_{exp_id}_seed_{seed}.txt"
            os.makedirs(os.path.dirname(log_train), exist_ok=True)
            
            train_cmd = (
                f"python train.py {exp['args']} "
                f"+seed_everything={seed} "
                f"trainer.max_epochs={MAX_EPOCHS} "
                f"trainer.default_root_dir=logs/benchmark/{exp_id}_{seed}"
            )
            run_command(train_cmd, log_train)
            
            # 3. 测试
            log_test = f"logs/benchmark/test_{exp_id}_seed_{seed}.txt"
            test_cmd = "python test.py" # test.py 会自动找 checkpoints 里最新的
            test_output = run_command(test_cmd, log_test)
            
            # 4. 记录数据
            m = parse_metrics(test_output)
            results[exp_id].append(m)
            print(f"✅ Exp {exp_id} (Seed {seed}) Result: IoU={m['Overall_IoU']}%")

    # ==========================================
    # 📊 FINAL REPORT GENERATION
    # ==========================================
    print("\n\n" + "="*80)
    print("🏆 FINAL MULTI-SEED BENCHMARK REPORT")
    print("="*80)

    # --- Helper to format Mean ± Std ---
    def fmt_stat(exp_id, metric_key):
        vals = [r[metric_key] for r in results[exp_id]]
        if not vals: return "N/A"
        mean = np.mean(vals)
        std = np.std(vals)
        return f"{mean:.2f} ± {std:.2f}"

    # --- Table 2: Ablation Study ---
    print("\n### Table 2: Ablation Study (Mean ± Std over 3 runs)")
    print("| ID | Model | ESM | DA | pLDDT | AF2 IoU (%) |")
    print("|---|---|---|---|---|---|")
    
    for exp in EXPERIMENTS:
        args = exp['args']
        esm = "❌" if "use_esm=False" in args else "✅"
        da = "❌" if "use_da=False" in args else "✅"
        plddt = "❌" if "use_plddt_weight=False" in args or "use_da=False" in args else "✅"
        
        iou_str = fmt_stat(exp['id'], 'Overall_IoU')
        print(f"| {exp['id']} | {exp['name']} | {esm} | {da} | {plddt} | **{iou_str}** |")

    # --- Table 3: Confidence Analysis (Experiment D only) ---
    print("\n### Table 3: Analysis by Confidence (Experiment D, Mean ± Std)")
    print("| Region Type | Metric | Value |")
    print("|---|---|---|")
    
    if 'D' in results and results['D']:
        high = fmt_stat('D', 'High_IoU')
        med = fmt_stat('D', 'Med_IoU')
        low_fpr = fmt_stat('D', 'Low_FPR')
        
        print(f"| High Conf (>90) | IoU | {high} |")
        print(f"| Med Conf (70-90) | IoU | {med} |")
        print(f"| Low Conf (<70) | False Positive Rate | **{low_fpr}** |")
    else:
        print("| Experiment D data missing | - | - |")

    print("\n✅ Benchmark Completed. Logs are saved in 'logs/benchmark/'.")

if __name__ == "__main__":
    main()