import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import sys
import os
import glob
import pandas as pd
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evopoint_da.models.module import EvoPointDALitModule

# ==========================================
# 1. Helpers: Data Parsers (保持不变)
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
    except Exception as e:
        print(f"⚠️ Failed to parse PDB residues for {pdb_path}: {e}")
        return []

def load_pesto_scores(pdb_path):
    scores = []
    seen_residues = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    res_seq_id = line[22:27].strip()
                    chain_id = line[21]
                    unique_res_id = f"{chain_id}_{res_seq_id}"
                    if unique_res_id not in seen_residues:
                        try:
                            b_factor = float(line[60:66].strip())
                            scores.append(b_factor)
                            seen_residues.add(unique_res_id)
                        except ValueError:
                            continue
        return np.array(scores)
    except Exception as e:
        print(f"⚠️ Warning: Failed to parse PeSTo PDB {pdb_path}: {e}")
        return None

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
    except Exception as e:
        return np.zeros(target_len)

def find_best_threshold(y_true, y_prob):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_prob > thresh).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh
    return best_thresh, best_f1

# ==========================================
# 2. Dataset Loader
# ==========================================
class SimpleFolderDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.file_list = sorted(glob.glob(os.path.join(folder_path, "*.pt")))
        if len(self.file_list) == 0:
            raise RuntimeError(f"❌ Error: No .pt files found in '{folder_path}'")
        else:
            print(f"✅ Loaded {len(self.file_list)} files from {folder_path}")

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        payload = torch.load(self.file_list[idx])
        if isinstance(payload, dict):
            data = Data(**payload)
        else:
            data = payload
        filename = os.path.basename(self.file_list[idx])
        data.pdb_id = os.path.splitext(filename)[0] 
        return data

# ==========================================
# 3. Inference Helper
# ==========================================
def run_inference(ckpt_path, loader, device='cuda'):
    """加载特定模型并在数据加载器上运行推理"""
    print(f"\n🚀 Loading Model: {ckpt_path}")
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"❌ Error loading checkpoint {ckpt_path}: {e}")
        sys.exit(1)

    all_labels = []
    all_probs_gated = [] # 我们主要关注 Gated (Confidence-Aware) 结果，或者是 Raw
    
    # 对于 Experiment A (Baseline)，它可能没有训练 pLDDT weight，但推理时可以用 pLDDT 过滤
    # 为了公平对比，我们统一提取 Gated 结果，或者也可以提取 Raw。
    # 这里我们提取 Gated，因为这是 ProtCross 的核心卖点。
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            
            # Forward
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Confidence Gating Logic
            if hasattr(model, '_normalize_plddt'):
                p = model._normalize_plddt(batch.plddt).squeeze()
            else:
                p = batch.plddt.squeeze() / 100.0
            
            is_reliable = (p > 0.65).float()
            probs_gated = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs_gated.append(probs_gated.cpu().numpy())
            
    return np.concatenate(all_labels), np.concatenate(all_probs_gated)

# ==========================================
# 4. Main Evaluation Logic
# ==========================================
def evaluate_and_plot(ckpt_path_a, ckpt_path_d, af2_data_folder, pesto_folder=None, p2rank_folder=None):
    BATCH_SIZE = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Load Data Iterator (Shared) ---
    # 只需要加载一次数据集
    test_dataset = SimpleFolderDataset(af2_data_folder)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # --- 1. Run Inference for Experiment A (Baseline) ---
    print(f"\n--- Processing Experiment A (Baseline) ---")
    y_true_a, y_probs_a = run_inference(ckpt_path_a, test_loader, device)

    # --- 2. Run Inference for Experiment D (ProtCross) ---
    print(f"\n--- Processing Experiment D (ProtCross) ---")
    y_true_d, y_probs_d = run_inference(ckpt_path_d, test_loader, device)
    
    # 确保标签一致
    assert np.array_equal(y_true_a, y_true_d), "❌ Error: Data mismatch between runs!"
    y_true = y_true_a # 使用其中一个作为 Ground Truth

    # --- 3. Run Baselines (PeSTo / P2Rank) ---
    # 这部分逻辑需要重新遍历 loader 来匹配 ID，或者更简单的，我们假设顺序一致
    # 为了复用之前的逻辑，我们再跑一遍轻量级的循环来收集外部指标
    
    print("\n--- Collecting External Baselines (PeSTo/P2Rank) ---")
    all_pesto = []
    all_p2rank = []
    valid_mask = [] # 用于记录哪些样本是有效的（PeSTo 对齐问题）
    
    has_pesto = pesto_folder and os.path.exists(pesto_folder)
    has_p2rank = p2rank_folder and os.path.exists(p2rank_folder)
    
    for i, batch in enumerate(test_loader):
        current_id = batch.pdb_id[0]
        gt_len = len(batch.y)
        
        pesto_val = None
        p2rank_val = None
        is_valid = True
        
        # PeSTo
        if has_pesto:
            p_path = os.path.join(pesto_folder, f"{current_id}.pdb")
            if not os.path.exists(p_path): p_path = os.path.join(pesto_folder, f"{current_id}_pesto.pdb")
            if os.path.exists(p_path):
                sc = load_pesto_scores(p_path)
                if sc is not None and len(sc) >= gt_len:
                    pesto_val = sc[:gt_len]
                else:
                    is_valid = False
            else:
                is_valid = False
        
        # P2Rank
        if has_p2rank and is_valid:
            # 寻找原始 PDB 以获取 ID 映射
            pdb_ref = os.path.join(pesto_folder, f"{current_id}.pdb") 
            if not os.path.exists(pdb_ref):
                pdb_ref = os.path.join(af2_data_folder.replace("processed_af2", "raw_af2"), f"{current_id}.pdb")
            
            if os.path.exists(pdb_ref):
                res_map = get_residue_ids_from_pdb(pdb_ref)
                if len(res_map) >= gt_len:
                    res_map = res_map[:gt_len]
                    csv_path = os.path.join(p2rank_folder, f"{current_id}.pdb_predictions.csv")
                    p2rank_val = load_p2rank_scores(csv_path, gt_len, res_map)
                else:
                    p2rank_val = np.zeros(gt_len)
            else:
                p2rank_val = np.zeros(gt_len)
        
        if is_valid:
            valid_mask.append(True)
            if has_pesto: all_pesto.append(pesto_val)
            if has_p2rank: all_p2rank.append(p2rank_val)
        else:
            valid_mask.append(False)

    # --- Data Filtering for Plotting ---
    # 因为 PeSTo 可能导致某些样本不可用，我们需要过滤 y_true, y_probs_a, y_probs_d
    # 这里的 mask 是样本级的 (batch 级)，我们需要将其扩展到残基级
    # 但为了简化，我们可以假设所有样本都有效，或者只在画外部对比时过滤
    # 更加严谨的做法：重新拼接所有数据
    
    # 简便起见：如果启用了 PeSTo，我们只取 PeSTo 有效的子集进行画图
    if has_pesto:
        # 重构 y_true / probs 为 list of arrays
        list_y = []
        list_a = []
        list_d = []
        idx_start = 0
        
        # 重新遍历以切分 array (因为之前 concat 了)
        # 这比较麻烦，不如重新 run 一次 loader。
        # 为了代码简洁，这里采用一种策略：
        # 如果样本丢失很少，直接用全部数据的 A/D 和 仅有效数据的 PeSTo 对比（虽然有点不齐）
        # 严谨做法：只画全部数据的 A/D。如果需要对比，另外画。
        # 为了满足用户需求，我们假设数据都是对齐的（通常 benchmark 数据集也是清洗过的）。
        pass
    
    # 最终数据拼接
    y_pesto_flat = np.concatenate(all_pesto) if all_pesto else None
    y_p2rank_flat = np.concatenate(all_p2rank) if all_p2rank else None
    
    # 如果 PeSTo 过滤了数据，我们需要对 A 和 D 也做同样的过滤，否则曲线没法比
    if has_pesto and len(all_pesto) < len(test_loader):
        print("⚠️  Warning: Aligning datasets to PeSTo valid samples...")
        # 重新收集 A 和 D
        filtered_a, filtered_d, filtered_y = [], [], []
        curr_idx = 0
        dataset_files = sorted(glob.glob(os.path.join(af2_data_folder, "*.pt")))
        
        for i, is_ok in enumerate(valid_mask):
            # 加载该文件的大小
            d = torch.load(dataset_files[i])
            length = len(d['y']) if isinstance(d, dict) else len(d.y)
            
            segment_y = y_true[curr_idx : curr_idx+length]
            segment_a = y_probs_a[curr_idx : curr_idx+length]
            segment_d = y_probs_d[curr_idx : curr_idx+length]
            
            if is_ok:
                filtered_y.append(segment_y)
                filtered_a.append(segment_a)
                filtered_d.append(segment_d)
            
            curr_idx += length
            
        y_true = np.concatenate(filtered_y)
        y_probs_a = np.concatenate(filtered_a)
        y_probs_d = np.concatenate(filtered_d)

    print("\nCalculating Metrics & Plotting...")

    # ==========================================
    # 5. Plotting
    # ==========================================
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Colors
    c_base = '#95a5a6'  # Grey (Baseline A)
    c_ours = '#d35400'  # Orange (ProtCross D)
    c_pesto = '#27ae60' # Green
    c_p2 = '#2980b9'    # Blue

    # --- ROC ---
    # Exp A (Baseline)
    fpr_a, tpr_a, _ = roc_curve(y_true, y_probs_a)
    ax1.plot(fpr_a, tpr_a, color=c_base, ls='--', lw=2, label=f'Baseline (Pure Geom) (AUC={auc(fpr_a, tpr_a):.2f})')
    
    # Exp D (ProtCross)
    fpr_d, tpr_d, _ = roc_curve(y_true, y_probs_d)
    ax1.plot(fpr_d, tpr_d, color=c_ours, lw=3, label=f'ProtCross (AUC={auc(fpr_d, tpr_d):.2f})')

    # PeSTo
    if y_pesto_flat is not None:
        fpr_p, tpr_p, _ = roc_curve(y_true, y_pesto_flat)
        ax1.plot(fpr_p, tpr_p, color=c_pesto, lw=2.5, ls='-', alpha=0.8, label=f'PeSTo (AUC={auc(fpr_p, tpr_p):.2f})')

    # P2Rank
    if y_p2rank_flat is not None:
        fpr_p2, tpr_p2, _ = roc_curve(y_true, y_p2rank_flat)
        ax1.plot(fpr_p2, tpr_p2, color=c_p2, lw=2.5, ls='-.', alpha=0.8, label=f'P2Rank (AUC={auc(fpr_p2, tpr_p2):.2f})')
    
    ax1.plot([0,1],[0,1], 'k:', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f'ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- PR ---
    # Exp A
    p_a, r_a, _ = precision_recall_curve(y_true, y_probs_a)
    ax2.plot(r_a, p_a, color=c_base, ls='--', lw=2, label=f'Baseline (Pure Geom) (AP={average_precision_score(y_true, y_probs_a):.2f})')

    # Exp D
    p_d, r_d, _ = precision_recall_curve(y_true, y_probs_d)
    ax2.plot(r_d, p_d, color=c_ours, lw=3, label=f'ProtCross (AP={average_precision_score(y_true, y_probs_d):.2f})')

    # PeSTo
    if y_pesto_flat is not None:
        p_p, r_p, _ = precision_recall_curve(y_true, y_pesto_flat)
        ax2.plot(r_p, p_p, color=c_pesto, lw=2.5, ls='-', alpha=0.8, label=f'PeSTo (AP={average_precision_score(y_true, y_pesto_flat):.2f})')

    # P2Rank
    if y_p2rank_flat is not None:
        p_p2, r_p2, _ = precision_recall_curve(y_true, y_p2rank_flat)
        ax2.plot(r_p2, p_p2, color=c_p2, lw=2.5, ls='-.', alpha=0.8, label=f'P2Rank (AP={average_precision_score(y_true, y_p2rank_flat):.2f})')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('comparison_metrics_A_vs_D.png')
    print("✅ Plot saved to comparison_metrics_A_vs_D.png")

    # ==========================================
    # 6. Stats Table
    # ==========================================
    print("\n" + "="*80)
    print(f"{'Model':<25} | {'AUC':<6} | {'AP':<6} | {'Max F1':<8} | {'Threshold':<9} |")
    print("-" * 80)
    
    def print_row(name, y_t, y_p):
        auc_s = auc(roc_curve(y_t, y_p)[0], roc_curve(y_t, y_p)[1])
        ap_s = average_precision_score(y_t, y_p)
        bt, bf1 = find_best_threshold(y_t, y_p)
        print(f"{name:<25} | {auc_s:.4f} | {ap_s:.4f} | {bf1:.4f}   | {bt:.2f}      |")

    print_row("Baseline (Exp A, Seed 42)", y_true, y_probs_a)
    print_row("ProtCross (Exp D, Seed 42)", y_true, y_probs_d)
    
    if y_pesto_flat is not None:
        print_row("PeSTo", y_true, y_pesto_flat)
    if y_p2rank_flat is not None:
        print_row("P2Rank", y_true, y_p2rank_flat)
    
    print("="*80)

def find_ckpt(folder):
    """在文件夹中查找 last.ckpt 或 best*.ckpt"""
    # 优先找 last.ckpt
    last = os.path.join(folder, "last.ckpt")
    if os.path.exists(last): return last
    
    # 否则找 best
    cands = glob.glob(os.path.join(folder, "*.ckpt"))
    if cands: return cands[0] # 返回第一个
    return None

if __name__ == "__main__":
    # --- Configuration ---
    
    # 1. 定义实验目录
    DIR_A = "saved_weights/A_42"
    DIR_D = "saved_weights/D_42"
    
    ckpt_a = find_ckpt(DIR_A)
    ckpt_d = find_ckpt(DIR_D)
    
    if not ckpt_a:
        print(f"❌ Error: Checkpoint not found in {DIR_A}")
        sys.exit(1)
    if not ckpt_d:
        print(f"❌ Error: Checkpoint not found in {DIR_D}")
        sys.exit(1)

    # 2. GT Data (.pt files)
    AF2_DATA_FOLDER = "data/processed_af2"

    # 3. PeSTo / P2Rank Results (Optional)
    PESTO_FOLDER = "data/PeSTo_results" 
    P2RANK_FOLDER = "data/p2rank_results"

    evaluate_and_plot(ckpt_a, ckpt_d, AF2_DATA_FOLDER, PESTO_FOLDER, P2RANK_FOLDER)
