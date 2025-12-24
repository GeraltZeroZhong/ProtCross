import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from torch_geometric.loader import DataLoader
import sys
import glob

# === 🛠️ 配置区域 (根据你的实际文件路径修改) ===
# 格式: {种子: "Checkpoint文件路径"}
CKPT_MAP = {
    1:   "checkpoints/last.ckpt",
    2025:   "checkpoints/last-v1.ckpt",      # 第1次跑的
    1224: "checkpoints/last-v2.ckpt",   # 第2次跑的
    318: "checkpoints/last-v3.ckpt"    # 第3次跑的
}

AF2_DATA_FOLDER = "data/processed_af2"
OUTPUT_CSV = "experiment_results_summary.csv"

# 引入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.evopoint_da.models.module import EvoPointDALitModule
from scripts.plot_metrics import SimpleFolderDataset 

def evaluate_model(ckpt_path, data_folder, seed):
    """加载模型并评估核心指标"""
    print(f"\n{'='*40}")
    print(f"📊 Evaluating SEED {seed}")
    print(f"📂 Checkpoint: {ckpt_path}")
    print(f"{'='*40}")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: File not found: {ckpt_path}")
        return None

    # 加载模型
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.cuda()
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # 加载数据
    dataset = SimpleFolderDataset(data_folder)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    all_probs = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            batch = batch.cuda()
            
            # Forward Pass
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Confidence Gating
            p = model._normalize_plddt(batch.plddt).squeeze()
            is_reliable = (p > 0.65).float()
            probs_gated = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs_gated.cpu().numpy())
            
            if i % 50 == 0:
                print(f"Processing {i}/{len(loader)}...")

    # 计算指标
    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    ap_score = average_precision_score(y_true, y_scores)
    
    # 寻找最佳 F1
    best_f1 = 0
    best_thresh = 0
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_scores > thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
            
    print(f"✅ Result: AUC={auc_score:.4f}, AP={ap_score:.4f}, F1={best_f1:.4f}")
    
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
    
    # 遍历预定义的 Checkpoint 列表
    for seed, ckpt_path in CKPT_MAP.items():
        metrics = evaluate_model(ckpt_path, AF2_DATA_FOLDER, seed)
        
        if metrics:
            results.append(metrics)
            # 实时保存
            df_current = pd.DataFrame(results)
            df_current.to_csv(OUTPUT_CSV, index=False)

    # === 最终汇总 ===
    if len(results) > 0:
        df = pd.DataFrame(results)
        print("\n" + "="*50)
        print("🏆 FINAL SUMMARY (Mean ± Std)")
        print("="*50)
        
        # 计算均值和标准差
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
        
        # 添加汇总行
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
    else:
        print("❌ No results obtained. Please check file paths.")

if __name__ == "__main__":
    main()
