import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import sys
import os
import glob
import pandas as pd
import warnings
from tqdm import tqdm
import random

# 忽略警告
warnings.filterwarnings("ignore")

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evopoint_da.models.module import EvoPointDALitModule

# ==========================================
# 1. Custom Dataset for Evaluation
# ==========================================
class EvalDataset(Dataset):
    """
    专门用于评估的 Dataset。
    修改：逻辑与 EvoPointDataset (src/evopoint_da/data/dataset.py) 保持严格一致
    1. Sort -> Random Shuffle (Seed 42) -> Split
    2. 过滤掉 y.sum() == 0 的无效样本
    3. 保留 pdb_id 用于 external baselines
    """
    def __init__(self, root, split="test"):
        super().__init__()
        self.root = root
        
        # 1. 获取所有 .pt 文件并排序 (保证初始顺序一致)
        all_files = sorted(glob.glob(os.path.join(root, "*.pt")))
        
        # 2. 核心：固定随机种子并打乱 (复制 EvoPointDataset 逻辑)
        random.seed(42)
        random.shuffle(all_files)
        
        num = len(all_files)
        
        # 3. 按比例划分 (8:1:1)
        if split == "train":
            candidate_files = all_files[:int(num*0.8)]
        elif split == "val":
            candidate_files = all_files[int(num*0.8):int(num*0.9)]
        elif split == "test":
            candidate_files = all_files[int(num*0.9):]
        elif split == "all":
            candidate_files = all_files
        else:
            candidate_files = []

        print(f"[{split}] Scanning {len(candidate_files)} candidate files from {root}...")
        
        # 4. 过滤无效数据 (y.sum() == 0) 并保留有效文件列表
        self.files = []
        for fpath in tqdm(candidate_files, desc=f"Filtering {split}"):
            try:
                # 只读取 label 进行检查，减少内存消耗
                # 注意：如果文件很大，这里可能会慢，但为了保持一致性是必须的
                data = torch.load(fpath, weights_only=False)
                
                # 兼容字典或Data对象
                y = data['y'] if isinstance(data, dict) else data.y
                
                # 转换为 tensor 检查
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)
                
                # 严格过滤：丢弃标签全为 0 的样本 (与 EvoPointDataset 保持一致)
                if y.sum() > 0:
                    self.files.append(fpath)
            except Exception as e:
                print(f"Error checking {fpath}: {e}")
                continue
                
        print(f"[{split}] Final valid files: {len(self.files)} (Filtered {len(candidate_files) - len(self.files)})")

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = self.files[idx]
        try:
            payload = torch.load(file_path, weights_only=False)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return Data() # Return empty data to avoid crash
            
        if isinstance(payload, dict):
            data = Data(**payload)
        else:
            data = payload
            
        # [关键] 强制添加 pdb_id
        filename = os.path.basename(file_path)
        pdb_id = os.path.splitext(filename)[0]
        data.pdb_id = pdb_id
        
        return data

# ==========================================
# 2. Helpers: Data Parsers
# ==========================================
def get_residue_ids_from_pdb(pdb_path):
    res_ids = []
    seen = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21].strip()
                    if chain_id == '': chain_id = 'A'
                    res_seq = line[22:27].strip()
                    unique_id = f"{chain_id}_{res_seq}"
                    if unique_id not in seen:
                        res_ids.append(unique_id)
                        seen.add(unique_id)
        return res_ids
    except: return []

def load_pesto_scores(pdb_path):
    scores = []
    seen = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_seq = line[22:27].strip()
                    chain = line[21]
                    uid = f"{chain}_{res_seq}"
                    if uid not in seen:
                        try:
                            scores.append(float(line[60:66].strip()))
                            seen.add(uid)
                        except: continue
        return np.array(scores)
    except: return None

def load_p2rank_scores(csv_path, target_len, residue_ids_map):
    scores = np.zeros(target_len)
    if not os.path.exists(csv_path): return None
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        for _, row in df.iterrows():
            prob = row['probability']
            res_str_list = str(row['residue_ids']).strip().split()
            for res_str in res_str_list:
                target_id = res_str.strip()
                if target_id in residue_ids_map:
                    idx = residue_ids_map.index(target_id)
                    scores[idx] = max(scores[idx], prob)
        return scores
    except: return np.zeros(target_len)

# ==========================================
# 3. Checkpoint Scanner
# ==========================================
def get_grouped_checkpoints(root_dir):
    """
    扫描 saved_weights，将实验分为 'A' (Baseline) 和 'D' (ProtCross) 两组
    """
    if not os.path.exists(root_dir):
        print(f"❌ Error: {root_dir} not found.")
        return {'A': [], 'D': []}

    groups = {'A': [], 'D': []}
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    print(f"📂 Scanning {len(subdirs)} folders in '{root_dir}'...")
    
    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)
        
        # 找 ckpt
        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            cands = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if cands: target_ckpt = cands[0]
            else: continue
        
        # 分组
        if folder_name.startswith("A_"):
            groups['A'].append(target_ckpt)
        elif folder_name.startswith("D_"):
            groups['D'].append(target_ckpt)
            
    print(f"   ✅ Found {len(groups['A'])} checkpoints for Baseline (A)")
    print(f"   ✅ Found {len(groups['D'])} checkpoints for ProtCross (D)")
    return groups

# ==========================================
# 4. Inference Engine
# ==========================================
def run_inference(ckpt_path, loader, device='cuda'):
    """运行单个模型的推理"""
    try:
        import logging
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(ckpt_path)}: {e}")
        return None, None

    all_y = []
    all_p = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # 兼容处理
            src_x = batch.x if hasattr(model.hparams, 'use_esm') and model.hparams.use_esm else None
            
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_y.append(batch.y.cpu().numpy())
            all_p.append(probs.cpu().numpy())
            
    return np.concatenate(all_y), np.concatenate(all_p)

# ==========================================
# 5. Statistical Aggregator (Fixed)
# ==========================================
def aggregate_roc_results(ckpt_list, loader, device):
    """收集多个模型的 ROC 数据并计算 Mean ± Std"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    pbar = tqdm(ckpt_list, desc="Aggregating ROC", leave=False)
    for ckpt in pbar:
        y_true, y_score = run_inference(ckpt, loader, device)
        if y_true is None: continue
        
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # [Fix] 去重 FPR 以避免插值错误
        _, unique_indices = np.unique(fpr, return_index=True)
        fpr_unique = fpr[unique_indices]
        tpr_unique = tpr[unique_indices]

        interp_tpr = np.interp(mean_fpr, fpr_unique, tpr_unique)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    
    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc

def aggregate_pr_results(ckpt_list, loader, device):
    """收集多个模型的 PR 数据并计算 Mean ± Std (修复起点问题)"""
    precs = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)
    
    pbar = tqdm(ckpt_list, desc="Aggregating PR", leave=False)
    for ckpt in pbar:
        y_true, y_score = run_inference(ckpt, loader, device)
        if y_true is None: continue
        
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)
        
        # 1. 翻转为升序 recall
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]
        
        # 2. [关键修复] 去重
        _, unique_indices = np.unique(recall_rev, return_index=True)
        recall_unique = recall_rev[unique_indices]
        precision_unique = precision_rev[unique_indices]
        
        # 3. [关键修复] 强制起点 (Recall=0 -> Precision=1)
        if recall_unique[0] == 0:
            precision_unique[0] = 1.0
            
        interp_prec = np.interp(mean_recall, recall_unique, precision_unique)
        precs.append(interp_prec)
        
    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    return mean_recall, mean_prec, std_prec, mean_ap, std_ap

# ==========================================
# 6. External Baselines
# ==========================================
def get_external_baselines(loader, pesto_dir, p2rank_dir, af2_dir):
    """收集 PeSTo 和 P2Rank 的结果"""
    # 检查是否有数据
    if len(loader.dataset) == 0: return None, None, None
    sample = loader.dataset[0]
    if not hasattr(sample, 'pdb_id'): 
        print("⚠️ Warning: No 'pdb_id' in dataset. Skipping external baselines.")
        return None, None, None

    y_true_all = []
    pesto_scores = []
    p2rank_scores = []
    
    has_pesto = pesto_dir and os.path.exists(pesto_dir)
    has_p2rank = p2rank_dir and os.path.exists(p2rank_dir)
    
    print("   Running External Baselines...")
    for batch in tqdm(loader, desc="Ext Baselines", leave=False):
        # 处理 batch (batch_size=1)
        if isinstance(batch.pdb_id, list): pid = batch.pdb_id[0]
        else: pid = batch.pdb_id
        
        y_true = batch.y.numpy()
        gt_len = len(y_true)
        y_true_all.append(y_true)
        
        # PeSTo
        if has_pesto:
            p_path = os.path.join(pesto_dir, f"{pid}_i3.pdb")
            if not os.path.exists(p_path): p_path = os.path.join(pesto_dir, f"{pid}.pdb")
            sc = load_pesto_scores(p_path)
            if sc is not None and len(sc) >= gt_len: pesto_scores.append(sc[:gt_len])
            else: pesto_scores.append(np.zeros(gt_len))
            
        # P2Rank
        if has_p2rank:
            raw_pdb = os.path.join(af2_dir.replace("processed_af2", "raw_af2"), f"{pid}.pdb")
            if not os.path.exists(raw_pdb): raw_pdb = os.path.join(pesto_dir, f"{pid}.pdb")
            res_map = get_residue_ids_from_pdb(raw_pdb)
            
            csv_path = os.path.join(p2rank_dir, f"{pid}.pdb_predictions.csv")
            if res_map and len(res_map) >= gt_len:
                sc = load_p2rank_scores(csv_path, gt_len, res_map[:gt_len])
                p2rank_scores.append(sc)
            else:
                p2rank_scores.append(np.zeros(gt_len))

    y_cat = np.concatenate(y_true_all)
    p_cat = np.concatenate(pesto_scores) if pesto_scores else None
    p2_cat = np.concatenate(p2rank_scores) if p2rank_scores else None
    
    return y_cat, p_cat, p2_cat

# ==========================================
# 7. Main Logic
# ==========================================
def main():
    # --- Config ---
    WEIGHTS_DIR = "saved_weights"
    AF2_DATA = "data/processed_af2"
    PESTO_DIR = "data/PeSTo_results"
    P2RANK_DIR = "data/p2rank_results"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Dataset
    print(f"📦 Loading Dataset from {AF2_DATA}...")
    try:
        ds = EvalDataset(root=AF2_DATA, split="test")
        if len(ds) == 0: raise ValueError("Empty split")
    except:
        print("   ⚠️ Test split empty or failed, trying ALL files.")
        ds = EvalDataset(root=AF2_DATA, split="all")
    
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 2. Group Checkpoints
    groups = get_grouped_checkpoints(WEIGHTS_DIR)
    if not groups['A'] and not groups['D']:
        print("❌ No checkpoints found.")
        return

    # 3. Calculate Stats
    stats = {}
    print_table_rows = []

    for name, label in [('A', 'Baseline'), ('D', 'ProtCross')]:
        if not groups[name]: continue
        print(f"\n📈 Processing {label} ({len(groups[name])} runs)...")
        
        # ROC Stats
        fpr, tpr, tpr_std, auc_m, auc_s = aggregate_roc_results(groups[name], loader, device)
        # PR Stats
        rec, prec, prec_std, ap_m, ap_s = aggregate_pr_results(groups[name], loader, device)
        
        stats[name] = {
            'roc': (fpr, tpr, tpr_std, auc_m, auc_s),
            'pr': (rec, prec, prec_std, ap_m, ap_s)
        }
        
        print_table_rows.append({
            "Model": label,
            "AUC": f"{auc_m:.4f} ± {auc_s:.4f}",
            "AP": f"{ap_m:.4f} ± {ap_s:.4f}"
        })

    # 4. External Baselines
    y_ext, p_ext, p2_ext = get_external_baselines(loader, PESTO_DIR, P2RANK_DIR, AF2_DATA)
    if p_ext is not None:
        p_auc = auc(roc_curve(y_ext, p_ext)[0], roc_curve(y_ext, p_ext)[1])
        p_ap = average_precision_score(y_ext, p_ext)
        print_table_rows.append({"Model": "PeSTo", "AUC": f"{p_auc:.4f}", "AP": f"{p_ap:.4f}"})
        
    if p2_ext is not None:
        p2_auc = auc(roc_curve(y_ext, p2_ext)[0], roc_curve(y_ext, p2_ext)[1])
        p2_ap = average_precision_score(y_ext, p2_ext)
        print_table_rows.append({"Model": "P2Rank", "AUC": f"{p2_auc:.4f}", "AP": f"{p2_ap:.4f}"})

    # 5. Plotting
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 1.5, 'font.family': 'sans-serif'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    
    # Colors
    c_A = '#7f8c8d' # Grey (Baseline)
    c_D = '#d35400' # Orange (ProtCross)
    c_P = '#27ae60' # Green
    c_P2 = '#2980b9' # Blue

    # --- Plot ROC (On ax1) ---
    ax1.plot([0, 1], [0, 1], 'k:', alpha=0.5)
    
    if 'A' in stats:
        fpr, tpr, std, auc_m, auc_s = stats['A']['roc']
        ax1.plot(fpr, tpr, color=c_A, lw=2, label=f'Baseline (AUC={auc_m:.2f}±{auc_s:.2f})')
        ax1.fill_between(fpr, np.maximum(tpr-std, 0), np.minimum(tpr+std, 1), color=c_A, alpha=0.2)
        
    if 'D' in stats:
        fpr, tpr, std, auc_m, auc_s = stats['D']['roc']
        ax1.plot(fpr, tpr, color=c_D, lw=3, label=f'ProtCross (AUC={auc_m:.2f}±{auc_s:.2f})')
        ax1.fill_between(fpr, np.maximum(tpr-std, 0), np.minimum(tpr+std, 1), color=c_D, alpha=0.2)
        
    if p_ext is not None:
        fpr_p, tpr_p, _ = roc_curve(y_ext, p_ext)
        ax1.plot(fpr_p, tpr_p, color=c_P, lw=2, ls='--', label=f'PeSTo (AUC={p_auc:.2f})')
    if p2_ext is not None:
        fpr_2, tpr_2, _ = roc_curve(y_ext, p2_ext)
        ax1.plot(fpr_2, tpr_2, color=c_P2, lw=2, ls='--', label=f'P2Rank (AUC={p2_auc:.2f})')

    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # --- Plot PR (On ax2) ---
    if 'A' in stats:
        rec, prec, std, ap_m, ap_s = stats['A']['pr']
        ax2.plot(rec, prec, color=c_A, lw=2, label=f'Baseline (AP={ap_m:.2f}±{ap_s:.2f})')
        ax2.fill_between(rec, np.maximum(prec-std, 0), np.minimum(prec+std, 1), color=c_A, alpha=0.2)
        
    if 'D' in stats:
        rec, prec, std, ap_m, ap_s = stats['D']['pr']
        ax2.plot(rec, prec, color=c_D, lw=3, label=f'ProtCross (AP={ap_m:.2f}±{ap_s:.2f})')
        ax2.fill_between(rec, np.maximum(prec-std, 0), np.minimum(prec+std, 1), color=c_D, alpha=0.2)
        
    if p_ext is not None:
        p_p, r_p, _ = precision_recall_curve(y_ext, p_ext)
        ax2.plot(r_p, p_p, color=c_P, lw=2, ls='--', label=f'PeSTo (AP={p_ap:.2f})')
    if p2_ext is not None:
        p_2, r_2, _ = precision_recall_curve(y_ext, p2_ext)
        ax2.plot(r_2, p_2, color=c_P2, lw=2, ls='--', label=f'P2Rank (AP={p2_ap:.2f})')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    out_file = "aggregated_metrics_plot.png"
    plt.savefig(out_file)
    print(f"\n✅ Plot saved to {out_file}")
    
    # 6. Print Summary Table
    print("\n" + "="*50)
    print(f"{'Model':<20} | {'AUC':<18} | {'AP':<18}")
    print("-" * 60)
    for row in print_table_rows:
        print(f"{row['Model']:<20} | {row['AUC']:<18} | {row['AP']:<18}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
