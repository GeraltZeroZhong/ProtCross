# File: run_benchmark.py
import subprocess
import re
import time
import os
import shutil

# 定义四个关键实验
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

def run_command(cmd, log_file):
    """运行命令并将输出同时写入文件和控制台"""
    print(f"🚀 Running: {cmd}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        output_buffer = ""
        for line in process.stdout:
            print(line, end="") # 实时显示
            f.write(line)
            output_buffer += line
        process.wait()
    return output_buffer

def parse_metrics(output):
    """从 test.py 的输出中提取指标"""
    metrics = {"Overall_IoU": "N/A", "High_IoU": "N/A", "Med_IoU": "N/A", "Low_FPR": "N/A"}
    
    # 查找 <<<METRICS_START>>> 块
    if "<<<METRICS_START>>>" in output:
        block = output.split("<<<METRICS_START>>>")[1].split("<<<METRICS_END>>>")[0]
        for line in block.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":")
                metrics[k.strip()] = v.strip()
    return metrics

def clean_checkpoints():
    """清理 checkpoints 文件夹，防止 test 读取到旧的权重"""
    if os.path.exists("checkpoints"):
        shutil.rmtree("checkpoints")
    os.makedirs("checkpoints", exist_ok=True)

def main():
    results = {}
    
    print("="*60)
    print("🧬 EvoPoint-DA Automated Benchmark Runner")
    print("="*60)
    print(f"Total Experiments: {len(EXPERIMENTS)}")
    print("Note: Ensuring 'checkpoints' folder is clean before each run.\n")

    for i, exp in enumerate(EXPERIMENTS):
        exp_id = exp['id']
        print(f"\n[{i+1}/{4}] Starting Experiment {exp_id}: {exp['name']}")
        
        # 1. 清理旧权重 (确保 test.py 总是取到最新的)
        clean_checkpoints()
        
        # 2. 训练
        train_cmd = f"python train.py {exp['args']} trainer.max_epochs=50" # 建议跑50轮快速验证，正式跑可去掉max_epochs限制
        run_command(train_cmd, f"log_train_{exp_id}.txt")
        
        # 3. 测试
        # test.py 会自动读取 checkpoints 目录下最新的 best-*.ckpt
        test_cmd = "python test.py"
        test_output = run_command(test_cmd, f"log_test_{exp_id}.txt")
        
        # 4. 解析结果
        metrics = parse_metrics(test_output)
        results[exp_id] = metrics
        print(f"✅ Result {exp_id}: {metrics}")

    # ==========================================
    # Generate The Money Shot Tables
    # ==========================================
    print("\n\n" + "="*60)
    print("📊 FINAL RESULTS GENERATOR")
    print("="*60)

    # Table 2: Ablation Study
    print("\n### Table 2: Ablation Study")
    print("| ID | Model | ESM | DA | pLDDT | AF2 IoU (%) |")
    print("|---|---|---|---|---|---|")
    
    # Mapping booleans to symbols
    def get_mark(exp_args, key):
        return "✅" if f"{key}=True" in exp_args else ("❌" if f"{key}=False" in exp_args else "✅") # Default is True in config

    for exp in EXPERIMENTS:
        esm = "❌" if "use_esm=False" in exp['args'] else "✅"
        da = "❌" if "use_da=False" in exp['args'] else "✅"
        plddt = "❌" if "use_plddt_weight=False" in exp['args'] or "use_da=False" in exp['args'] else "✅"
        
        iou = results[exp['id']]['Overall_IoU']
        print(f"| {exp['id']} | {exp['name']} | {esm} | {da} | {plddt} | **{iou}** |")

    # Table 3: Confidence Analysis (Using Experiment D)
    print("\n### Table 3: Analysis by Confidence (Full Model D)")
    d_res = results['D']
    print("| Region Type | Metric | Value |")
    print("|---|---|---|")
    print(f"| High Conf (>90) | IoU | {d_res['High_IoU']}% |")
    print(f"| Med Conf (70-90) | IoU | {d_res['Med_IoU']}% |")
    print(f"| Low Conf (<70) | False Positive Rate | **{d_res['Low_FPR']}%** |")
    
    print("\nDone! Copy these tables to your paper.")

if __name__ == "__main__":
    main()