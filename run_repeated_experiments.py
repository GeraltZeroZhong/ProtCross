import os
import subprocess
import glob
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from torch_geometric.loader import DataLoader
import sys
import time

# === 配置区域 ===
SEEDS = [2025, 1224, 318]
MAX_EPOCHS = 50
AF2_DATA_FOLDER = "data/processed_af2"
OUTPUT_CSV = "experiment_results_summary.csv"

# 引入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.evopoint_da.models.module import EvoPointDALitModule
from scripts.plot_metrics import SimpleFolderDataset 

def run_training(seed):
    """调用 train.py 进行训练"""
    print(f"\n{'='*40}")
    print(f"🚀 Starting Training for SEED: {seed}")
    print(f"{'='*40}")
    
    # 虽然 Checkpoint 没存这儿，但日志可能还在这
    log_dir = f"logs/experiments/seed_{seed}"
    
    cmd = [
        "python", "train.py",
        f"+seed_everything={seed}",                 
        f"+trainer.default_root_dir={log_dir}",    
        f"trainer.max_epochs={MAX_EPOCHS}",
        "model.use_esm=True",
        "model.use_da=True",
        "model.use_plddt_weight=True",
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ Training finished for seed {seed}")
        # 稍微等一秒，确保文件系统写入完成，时间戳更新
        time.sleep(2)
        return log_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed for seed {seed}: {e}")
        return None

def find_best_checkpoint(log_dir):
    """
    修改版：直接在 checkpoints/ 目录下寻找最新的 .ckpt 文件
    """
    # 你的模型似乎固定保存到了根目录下的 checkpoints/
    ckpt_dir = "checkpoints"
    
    if not os.path.exists(ckpt_dir):
        print(f"❌ Error: Checkpoint directory '{ckpt_dir}' does not exist.")
        return None
    
    # 搜索所有 .ckpt 文件
    pattern = os.path.join(ckpt_dir, "*.ckpt")
    ckpts = glob.glob(pattern)
    
    if not ckpts:
        print(f"❌ No .ckpt files found in '{ckpt_dir}'.")
        return None
    
    # 关键：按“最后修改时间”排序，取最新的一个
    # 这样能自动匹配到 last.ckpt, last-v1.ckpt, last-v2.ckpt 中刚生成的那个
    latest_ckpt = max(ckpts, key=os.path.getmtime)
    
    print(f"📂 Found latest checkpoint (by time): {latest_ckpt}")
    return latest_ckpt

def evaluate_model(ckpt_path, data_folder, seed):
    """加载模型并评估核心指标"""
    print(f"📊 Evaluating seed {seed}...")
    
    model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    
    dataset = SimpleFolderDataset(data_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.cuda()
            
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            p = model._normalize_plddt(batch.plddt).squeeze()
            is_reliable = (p > 0.65).float()
            probs_gated = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs_gated.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    ap_score = average_precision_score(y_true, y_scores)
    
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_scores > thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    return {
        "Seed": seed,
        "AUC": auc_score,
        "AP": ap_score,
        "F1_Max": best_f1,
        "Best_Threshold": best_thresh,
        "Checkpoint": ckpt_path
    }

def main():
    results = []
    
    for seed in SEEDS:
        # 1. 训练
        log_dir = run_training(seed)
        
        if log_dir:
            # 2. 找刚才生成的最新 Checkpoint
            ckpt_path = find_best_checkpoint(log_dir)
            
            if ckpt_path:
                # 3. 评估
                metrics = evaluate_model(ckpt_path, AF2_DATA_FOLDER, seed)
                results.append(metrics)
                
                df_current = pd.DataFrame(results)
                df_current.to_csv(OUTPUT_CSV, index=False)
                print(f"✅ Results for seed {seed} saved to {OUTPUT_CSV}")
            else:
                print(f"❌ Evaluation skipped for seed {seed} (No checkpoint)")

    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("🏆 FINAL SUMMARY (Mean ± Std)")
        print("="*50)
        
        mean_auc = df["AUC"].mean()
        std_auc = df["AUC"].std()
        print(f"AUC : {mean_auc:.4f} ± {std_auc:.4f}")
        
        mean_ap = df["AP"].mean()
        std_ap = df["AP"].std()
        print(f"AP  : {mean_ap:.4f} ± {std_ap:.4f}")
        
        mean_f1 = df["F1_Max"].mean()
        std_f1 = df["F1_Max"].std()
        print(f"F1  : {mean_f1:.4f} ± {std_f1:.4f}")
        print("="*50)
        
        summary_row = {
            "Seed": "Mean ± Std",
            "AUC": f"{mean_auc:.4f} ± {std_auc:.4f}",
            "AP": f"{mean_ap:.4f} ± {std_ap:.4f}",
            "F1_Max": f"{mean_f1:.4f} ± {std_f1:.4f}",
            "Best_Threshold": "-",
            "Checkpoint": "-"
        }
        df_final = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"📁 Full report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
