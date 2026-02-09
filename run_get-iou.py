import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score
from torch_geometric.loader import DataLoader
import sys
import glob
from tqdm import tqdm
import pytorch_lightning as pl  # 1. 引入 PyTorch Lightning

# === 🛠️ 配置区域 ===
WEIGHTS_DIR = "saved_weights"       # 权重备份目录
AF2_DATA_FOLDER = "data/processed_af2"
OUTPUT_CSV = "standard_iou_results.csv"  # 结果保存文件
SEED = 42 # 固定种子

# 引入项目模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.evopoint_da.models.module import EvoPointDALitModule
from scripts.plot_metrics import SimpleFolderDataset 

def get_checkpoints_from_dir(root_dir):
    """自动扫描 saved_weights 目录"""
    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory '{root_dir}' not found.")
        return []

    ckpt_list = []
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"📂 Found {len(subdirs)} experiment folders in '{root_dir}':")
    
    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)
        
        # 优先找 last.ckpt
        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            candidates = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if candidates:
                target_ckpt = candidates[0]
            else:
                continue
        
        # 解析元数据
        parts = folder_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            seed = int(parts[-1])
            exp_id = "_".join(parts[:-1])
        else:
            seed = "Unknown"
            exp_id = folder_name
            
        ckpt_list.append({"path": target_ckpt, "exp_id": exp_id, "seed": seed})

    return ckpt_list

def evaluate_standard_iou(ckpt_info, data_folder):
    """仅计算最佳 Standard IoU (Binding Site)"""
    ckpt_path = ckpt_info['path']
    seed = ckpt_info['seed']
    exp_id = ckpt_info['exp_id']
    
    print(f"\n📊 Processing {exp_id} | Seed {seed}...")
    
    # 1. 加载模型
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return None
    
    # 2. 加载数据
    try:
        dataset = SimpleFolderDataset(data_folder)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
    except Exception as e:
        print(f"❌ Data Error: {e}")
        return None
    
    all_labels = []
    all_probs = []
    
    # 3. 推理
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", leave=False):
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # 这里的置信度过滤保持与模型训练时的配置一致
            if hasattr(model, '_normalize_plddt'):
                p = model._normalize_plddt(batch.plddt).squeeze()
            else:
                p = batch.plddt.squeeze() / 100.0
            
            # [修改] 注释掉硬截断逻辑，保留原始概率
            # 如果模型使用了 plddt 权重，则应用过滤逻辑
            # if model.hparams.use_plddt_weight:
            #     is_reliable = (p > 0.65).float()
            #     probs = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    
    # 4. 寻找最佳 Standard IoU
    best_iou = 0.0
    best_thresh = 0.0
    
    # 遍历阈值
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_scores > thresh).astype(int)
        
        # average='binary': 仅计算 Class 1 (Binding Site) 的 IoU
        iou = jaccard_score(y_true, y_pred, average='binary')
        
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    print(f"   ✅ Best Standard IoU: {best_iou*100:.2f}% (Thresh={best_thresh:.2f})")
    
    return {
        "Experiment": exp_id,
        "Seed": seed,
        "Best_Standard_IoU": best_iou * 100,
        "Best_Threshold": best_thresh,
        "Checkpoint": ckpt_path
    }

def main():
    # 2. 关键修复：固定全局随机种子
    # 这确保了 PointNet++ 中的采样 (FPS) 以及任何其他随机操作每次都是确定的
    pl.seed_everything(SEED, workers=True)
    print(f"🔒 Global seed set to {SEED}")

    tasks = get_checkpoints_from_dir(WEIGHTS_DIR)
    
    if not tasks:
        print("No checkpoints found. Please ensure 'saved_weights' exists.")
        return

    results = []
    
    for task in tasks:
        metrics = evaluate_standard_iou(task, AF2_DATA_FOLDER)
        if metrics:
            results.append(metrics)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if len(results) > 0:
        df = pd.DataFrame(results)
        
        print("\n" + "="*60)
        print("🏆 STANDARD IoU SUMMARY (Mean ± Std)")
        print("="*60)
        
        if "Experiment" in df.columns:
            groups = df.groupby("Experiment")
            summary_rows = []
            
            print(f"{'Experiment':<25} | {'Std IoU (Best)':<20}")
            print("-" * 50)
            
            for name, group in groups:
                m_iou, s_iou = group["Best_Standard_IoU"].mean(), group["Best_Standard_IoU"].std()
                s_iou = 0.0 if np.isnan(s_iou) else s_iou
                
                iou_str = f"{m_iou:.2f} ± {s_iou:.2f}"
                
                print(f"{name:<25} | {iou_str:<20}")
                
                summary_rows.append({
                    "Experiment": f"{name} (Mean)",
                    "Seed": "Aggregated",
                    "Best_Standard_IoU": iou_str,
                    "Best_Threshold": "-",
                    "Checkpoint": "-"
                })
            
            df_final = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
            df_final.to_csv(OUTPUT_CSV, index=False)
            
        print(f"\n📁 Results saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
