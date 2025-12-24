import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, f1_score, accuracy_score
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import sys
import os
import glob
import pandas as pd
import warnings

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evopoint_da.models.module import EvoPointDALitModule

# ==========================================
# 1. Helpers: Data Parsers
# ==========================================

def get_residue_ids_from_pdb(pdb_path):
    """
    读取 PDB 文件并返回残基 ID 列表 (用于 P2Rank 对齐)。
    格式: ["A_1", "A_2", "B_1", ...]
    """
    res_ids = []
    seen = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith(('ATOM', 'HETATM')):
                    chain_id = line[21].strip()
                    if chain_id == '': chain_id = 'A' # 默认链 A
                    res_seq = line[22:27].strip()
                    
                    # 唯一标识符: Chain_ResNum (e.g., A_10)
                    unique_id = f"{chain_id}_{res_seq}"
                    
                    if unique_id not in seen:
                        res_ids.append(unique_id)
                        seen.add(unique_id)
        return res_ids
    except Exception as e:
        print(f"⚠️ Failed to parse PDB residues for {pdb_path}: {e}")
        return []

def load_pesto_scores(pdb_path):
    """
    读取 PeSTo 生成的 PDB 文件 (B-factor 字段)。
    """
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
                            # PeSTo 概率在 B-factor (60-66)
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
    """
    读取 P2Rank 的 CSV 输出并映射到残基级概率。
    """
    scores = np.zeros(target_len)
    
    if not os.path.exists(csv_path):
        return None # 文件不存在，视为空

    try:
        # P2Rank CSV columns: name, rank, score, probability, ..., residue_ids
        df = pd.read_csv(csv_path, skipinitialspace=True)
        
        # 遍历每个预测的口袋
        for _, row in df.iterrows():
            prob = row['probability']
            
            # residue_ids 格式通常是 "A_1 A_2 A_3" (空格分隔)
            # 或者 "1 2 3" (如果没有链ID)
            res_str_list = str(row['residue_ids']).strip().split()
            
            for res_str in res_str_list:
                # 尝试匹配 residue_ids_map
                # P2Rank 输出通常包含链 (e.g. A_123)
                target_id = res_str.strip()
                
                if target_id in residue_ids_map:
                    idx = residue_ids_map.index(target_id)
                    # 如果残基属于多个口袋，取最大概率
                    scores[idx] = max(scores[idx], prob)
                    
        return scores
    except Exception as e:
        # print(f"⚠️ Error parsing P2Rank {csv_path}: {e}")
        return np.zeros(target_len) # 解析失败返回全0

def find_best_threshold(y_true, y_prob):
    """自动寻找最佳 F1 阈值"""
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
# 3. Main Evaluation Logic
# ==========================================
def evaluate_and_plot(ckpt_path, af2_data_folder, pesto_folder=None, p2rank_folder=None):
    # --- Settings ---
    BATCH_SIZE = 1 # 必须为 1 以便进行 PDB 对齐
    
    # --- Load Model ---
    print(f"Loading checkpoint: {ckpt_path}")
    model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    
    # --- Load Data ---
    test_dataset = SimpleFolderDataset(af2_data_folder)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Storage
    all_labels = []
    all_probs_raw = []
    all_probs_gated = []
    
    all_probs_pesto = []
    all_probs_p2rank = []
    
    has_pesto = pesto_folder is not None and os.path.exists(pesto_folder)
    has_p2rank = p2rank_folder is not None and os.path.exists(p2rank_folder)

    if has_pesto: print(f"🔹 Comparison: PeSTo enabled ({pesto_folder})")
    if has_p2rank: print(f"🔹 Comparison: P2Rank enabled ({p2rank_folder})")

    print("\nRunning Inference...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.cuda()
            current_id = batch.pdb_id[0]
            
            # --- 1. ProtCross Inference ---
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Confidence Gating
            p = model._normalize_plddt(batch.plddt).squeeze()
            is_reliable = (p > 0.65).float()
            probs_gated = probs * is_reliable
            
            # --- 2. Baseline Alignment Prep ---
            gt_len = len(batch.y)
            pesto_val = None
            p2rank_val = None
            valid_sample = True
            
            # --- 3. PeSTo Handling ---
            if has_pesto:
                pesto_path = os.path.join(pesto_folder, f"{current_id}.pdb")
                # Fallback check
                if not os.path.exists(pesto_path):
                     pesto_path = os.path.join(pesto_folder, f"{current_id}_pesto.pdb")
                
                if os.path.exists(pesto_path):
                    p_scores = load_pesto_scores(pesto_path)
                    
                    if p_scores is not None and len(p_scores) >= gt_len:
                        pesto_val = p_scores[:gt_len] # Truncate if necessary
                    else:
                        # PeSTo mismatch (too short), skip this sample for fairness
                        valid_sample = False
                else:
                    valid_sample = False # Missing PeSTo file

            # --- 4. P2Rank Handling ---
            if has_p2rank and valid_sample:
                # We need the Original PDB to map Residue IDs
                # Assumption: PDB exists in PeSTo folder (it's the same PDB)
                # OR we use the .pt file logic. Best is to use the PDB we used for PeSTo reading
                # If no PDB available easily, we assume batch.residue_ids exists? 
                # Let's rely on reading the PDB file from pesto_folder (which contains the PDBs)
                
                pdb_ref_path = os.path.join(pesto_folder, f"{current_id}.pdb")
                if not os.path.exists(pdb_ref_path):
                    # Try raw AF2 folder if pesto folder doesn't have it
                    pdb_ref_path = os.path.join(af2_data_folder.replace("processed_af2", "raw_af2"), f"{current_id}.pdb")

                if os.path.exists(pdb_ref_path):
                    # Get Mapping: Index -> "A_1"
                    res_map = get_residue_ids_from_pdb(pdb_ref_path)
                    
                    # Ensure map length matches GT
                    if len(res_map) >= gt_len:
                         res_map = res_map[:gt_len]
                         
                         # Load CSV
                         csv_name = f"{current_id}.pdb_predictions.csv" # Standard P2Rank output
                         csv_path = os.path.join(p2rank_folder, csv_name)
                         
                         p2rank_val = load_p2rank_scores(csv_path, gt_len, res_map)
                    else:
                        # PDB parsing mismatch
                        p2rank_val = np.zeros(gt_len)
                else:
                    # Cannot map, default to 0
                    p2rank_val = np.zeros(gt_len)

            # --- 5. Aggregation ---
            # Only add if valid (mainly for PeSTo strict alignment)
            if valid_sample:
                all_labels.append(batch.y.cpu().numpy())
                all_probs_raw.append(probs.cpu().numpy())
                all_probs_gated.append(probs_gated.cpu().numpy())
                
                if has_pesto:
                    all_probs_pesto.append(pesto_val)
                if has_p2rank:
                    all_probs_p2rank.append(p2rank_val)
            
            if i % 20 == 0:
                print(f"Processing {i} / {len(test_loader)}...")

    # Concatenate
    y_true = np.concatenate(all_labels)
    y_raw = np.concatenate(all_probs_raw)
    y_gated = np.concatenate(all_probs_gated)
    
    y_pesto = np.concatenate(all_probs_pesto) if has_pesto and len(all_probs_pesto) > 0 else None
    y_p2rank = np.concatenate(all_probs_p2rank) if has_p2rank and len(all_probs_p2rank) > 0 else None
    
    print("\nCalculating Metrics...")

    # ==========================================
    # 4. Plotting
    # ==========================================
    plt.rcParams.update({'font.size': 12, 'axes.linewidth': 1.5})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Colors
    c_raw = '#95a5a6'   # Grey
    c_ours = '#d35400'  # Orange
    c_pesto = '#27ae60' # Green
    c_p2 = '#2980b9'    # Blue

    # --- ROC ---
    # Ours Raw
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_raw)
    ax1.plot(fpr_raw, tpr_raw, color=c_raw, ls='--', lw=1.5, label=f'PointNet++ with ESM (AUC={auc(fpr_raw, tpr_raw):.2f})')
    
    # PeSTo
    if y_pesto is not None:
        fpr_p, tpr_p, _ = roc_curve(y_true, y_pesto)
        ax1.plot(fpr_p, tpr_p, color=c_pesto, lw=2.5, label=f'PeSTo (AUC={auc(fpr_p, tpr_p):.2f})')

    # P2Rank
    if y_p2rank is not None:
        fpr_p2, tpr_p2, _ = roc_curve(y_true, y_p2rank)
        ax1.plot(fpr_p2, tpr_p2, color=c_p2, lw=2.5, ls='-.', label=f'P2Rank (AUC={auc(fpr_p2, tpr_p2):.2f})')

    # Ours Gated
    fpr_g, tpr_g, _ = roc_curve(y_true, y_gated)
    ax1.plot(fpr_g, tpr_g, color=c_ours, lw=3, label=f'ProtCross (AUC={auc(fpr_g, tpr_g):.2f})')
    
    ax1.plot([0,1],[0,1], 'k:', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # --- PR ---
    # Ours Raw
    p_raw, r_raw, _ = precision_recall_curve(y_true, y_raw)
    ax2.plot(r_raw, p_raw, color=c_raw, ls='--', lw=1.5, label=f'Ours (Raw) (AP={average_precision_score(y_true, y_raw):.2f})')

    # PeSTo
    if y_pesto is not None:
        p_p, r_p, _ = precision_recall_curve(y_true, y_pesto)
        ax2.plot(r_p, p_p, color=c_pesto, lw=2.5, label=f'PeSTo (AP={average_precision_score(y_true, y_pesto):.2f})')

    # P2Rank
    if y_p2rank is not None:
        p_p2, r_p2, _ = precision_recall_curve(y_true, y_p2rank)
        ax2.plot(r_p2, p_p2, color=c_p2, lw=2.5, ls='-.', label=f'P2Rank (AP={average_precision_score(y_true, y_p2rank):.2f})')

    # Ours Gated
    p_g, r_g, _ = precision_recall_curve(y_true, y_gated)
    ax2.plot(r_g, p_g, color=c_ours, lw=3, label=f'ProtCross (AP={average_precision_score(y_true, y_gated):.2f})')

    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('comparison_metrics.png')
    print("✅ Plot saved to comparison_metrics.png")

    # ==========================================
    # 5. Stats Table
    # ==========================================
    print("\n" + "="*70)
    print(f"{'Model':<20} | {'AUC':<6} | {'AP':<6} | {'Max F1':<8} | {'Threshold':<9} |")
    print("-" * 70)
    
    # Ours Raw
    bt, bf1 = find_best_threshold(y_true, y_raw)
    print(f"{'Ours (Raw)':<20} | {auc(fpr_raw, tpr_raw):.4f} | {average_precision_score(y_true, y_raw):.4f} | {bf1:.4f}   | {bt:.2f}      |")
    
    # ProtCross
    bt, bf1 = find_best_threshold(y_true, y_gated)
    print(f"{'ProtCross (Gated)':<20} | {auc(fpr_g, tpr_g):.4f} | {average_precision_score(y_true, y_gated):.4f} | {bf1:.4f}   | {bt:.2f}      |")

    # PeSTo
    if y_pesto is not None:
        bt, bf1 = find_best_threshold(y_true, y_pesto)
        print(f"{'PeSTo':<20} | {auc(fpr_p, tpr_p):.4f} | {average_precision_score(y_true, y_pesto):.4f} | {bf1:.4f}   | {bt:.2f}      |")

    # P2Rank
    if y_p2rank is not None:
        bt, bf1 = find_best_threshold(y_true, y_p2rank)
        print(f"{'P2Rank':<20} | {auc(fpr_p2, tpr_p2):.4f} | {average_precision_score(y_true, y_p2rank):.4f} | {bf1:.4f}   | {bt:.2f}      |")
    
    print("="*70)

if __name__ == "__main__":
    # --- Configuration ---
    
    # 1. Checkpoint
    CKPT_PATH = "checkpoints/last.ckpt" 

    # 2. GT Data (.pt files)
    AF2_DATA_FOLDER = "data/processed_af2"

    # 3. PeSTo Results (Folder containing .pdb files with B-factor)
    PESTO_FOLDER = "data/PeSTo_results" 
    
    # 4. P2Rank Results (Folder containing .csv files)
    P2RANK_FOLDER = "data/p2rank_results"

    evaluate_and_plot(CKPT_PATH, AF2_DATA_FOLDER, PESTO_FOLDER, P2RANK_FOLDER)
