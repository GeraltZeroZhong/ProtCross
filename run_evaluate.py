import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, average_precision_score, f1_score
from torch_geometric.loader import DataLoader
import sys
import glob

# === 🛠️ 配置区域 ===
WEIGHTS_DIR = "saved_weights"       # 自动扫描的根目录
AF2_DATA_FOLDER = "data/processed_af2"
OUTPUT_CSV = "evaluation_results_all.csv"

# 引入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evopoint_da.models.module import EvoPointDALitModule
# [修改 1] 引入 EvoPointDataset
from evopoint_da.data.dataset import EvoPointDataset

def get_checkpoints_from_dir(root_dir):
    """
    扫描 saved_weights 目录，解析子文件夹结构
    返回列表: [{'path': ..., 'exp_id': ..., 'seed': ...}, ...]
    """
    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory '{root_dir}' not found.")
        return []

    ckpt_list = []
    
    # 遍历 root_dir 下的所有子文件夹
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"📂 Found {len(subdirs)} experiment folders in '{root_dir}':")
    
    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)
        
        # 1. 尝试寻找 last.ckpt
        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            # 如果 last.ckpt 不存在，尝试找任意 .ckpt
            candidates = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if candidates:
                target_ckpt = candidates[0] # 取第一个
                print(f"   ⚠️ 'last.ckpt' missing in {folder_name}, using fallback: {os.path.basename(target_ckpt)}")
            else:
                print(f"   ❌ No .ckpt files found in {folder_name}, skipping.")
                continue
        
        # 2. 解析元数据 (Exp_Seed)
        # 假设文件夹命名格式为 "ExperimentID_Seed" (例如 "A_42")
        parts = folder_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            seed = int(parts[-1])
            exp_id = "_".join(parts[:-1]) # 处理名字中有下划线的情况
        else:
            seed = "Unknown"
            exp_id = folder_name
            
        ckpt_list.append({
            "path": target_ckpt,
            "exp_id": exp_id,
            "seed": seed,
            "folder": folder_name
        })
        print(f"   ✅ Found: {exp_id} (Seed {seed}) -> {os.path.basename(target_ckpt)}")

    return ckpt_list

def evaluate_model(ckpt_info, data_folder):
    """加载模型并评估核心指标"""
    ckpt_path = ckpt_info['path']
    seed = ckpt_info['seed']
    exp_id = ckpt_info['exp_id']
    
    print(f"\n{'='*60}")
    print(f"📊 Evaluating Exp: {exp_id} | Seed: {seed}")
    print(f"📂 Load: {ckpt_path}")
    print(f"{'='*60}")
    
    # 加载模型
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return None
    
    # 加载数据 [修改 2] 使用 EvoPointDataset
    try:
        dataset = EvoPointDataset(root=data_folder, split="test")
        print(f"   ✅ Loaded EvoPointDataset (split='test', n={len(dataset)})")
    except Exception as e:
        print(f"   ⚠️ 'test' split failed, trying 'train' split ({e})")
        dataset = EvoPointDataset(root=data_folder, split="train")
        print(f"   ✅ Loaded EvoPointDataset (split='train', n={len(dataset)})")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels = []
    all_probs = []
    
    print("   Running Inference...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            # Forward Pass
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # [注意] 此处保持逻辑一致，不开启硬阈值过滤 (Raw Probabilities)
            # 如果需要开启，请取消注释以下代码：
            # if hasattr(model, '_normalize_plddt') and getattr(model.hparams, 'use_plddt_weight', False):
            #     p = model._normalize_plddt(batch.plddt).squeeze()
            #     is_reliable = (p > 0.65).float()
            #     probs = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

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
            
    print(f"   ✅ Result: AUC={auc_score:.4f}, AP={ap_score:.4f}, F1={best_f1:.4f}")
    
    return {
        "Experiment": exp_id,
        "Seed": seed,
        "AUC": auc_score,
        "AP": ap_score,
        "F1_Max": best_f1,
        "Best_Threshold": best_thresh,
        "Checkpoint": ckpt_path
    }

def main():
    # 1. 扫描文件夹
    tasks = get_checkpoints_from_dir(WEIGHTS_DIR)
    
    if not tasks:
        print("No checkpoints found to evaluate.")
        return

    results = []
    
    # 2. 依次评估
    for task in tasks:
        metrics = evaluate_model(task, AF2_DATA_FOLDER)
        if metrics:
            results.append(metrics)
            # 实时保存，防止中断
            df_current = pd.DataFrame(results)
            df_current.to_csv(OUTPUT_CSV, index=False)

    # 3. 最终汇总
    if len(results) > 0:
        df = pd.DataFrame(results)
        
        # 按 Experiment 分组计算均值
        print("\n" + "="*80)
        print("🏆 GROUPED SUMMARY (Mean ± Std)")
        print("="*80)
        
        # 尝试按 Experiment 分组统计，如果 Experiment 都是一样的，就整体统计
        if "Experiment" in df.columns:
            groups = df.groupby("Experiment")
            summary_rows = []
            
            print(f"{'Experiment':<25} | {'AUC':<20} | {'AP':<20}")
            print("-" * 75)
            
            for name, group in groups:
                m_auc, s_auc = group["AUC"].mean(), group["AUC"].std()
                m_ap, s_ap = group["AP"].mean(), group["AP"].std()
                
                # 处理只有一个样本 std 为 NaN 的情况
                s_auc = 0.0 if np.isnan(s_auc) else s_auc
                s_ap = 0.0 if np.isnan(s_ap) else s_ap
                
                auc_str = f"{m_auc:.4f} ± {s_auc:.4f}"
                ap_str = f"{m_ap:.4f} ± {s_ap:.4f}"
                
                print(f"{name:<25} | {auc_str:<20} | {ap_str:<20}")
                
                summary_rows.append({
                    "Experiment": f"{name} (Mean)",
                    "Seed": "Aggregated",
                    "AUC": auc_str,
                    "AP": ap_str,
                    "F1_Max": f"{group['F1_Max'].mean():.4f}",
                    "Best_Threshold": "-",
                    "Checkpoint": "-"
                })
            
            # 将汇总行添加到 CSV 底部
            df_final = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
            df_final.to_csv(OUTPUT_CSV, index=False)
            
        else:
            # 旧逻辑：整体统计
            print(f"AUC : {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}")

        print(f"\n📁 Full report saved to: {OUTPUT_CSV}")
    else:
        print("❌ No results obtained.")

if __name__ == "__main__":
    main()
