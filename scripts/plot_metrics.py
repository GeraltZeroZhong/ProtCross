import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import sys
import os
import glob

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evopoint_da.models.module import EvoPointDALitModule

# ==========================================
# 1. Dataset Loader
# ==========================================
class SimpleFolderDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.file_list = glob.glob(os.path.join(folder_path, "*.pt"))
        if len(self.file_list) == 0:
            raise RuntimeError(f"❌ Error: No .pt files found in '{folder_path}'")
        else:
            print(f"✅ Loaded {len(self.file_list)} files from {folder_path}")

    def len(self):
        return len(self.file_list)

    def get(self, idx):
        payload = torch.load(self.file_list[idx])
        if isinstance(payload, dict):
            return Data(**payload)
        return payload

# ==========================================
# 2. Evaluation Logic
# ==========================================
def evaluate_and_plot(ckpt_path, af2_data_folder):
    # --- Settings ---
    BATCH_SIZE = 8
    
    # --- Load Model ---
    print(f"Loading checkpoint: {ckpt_path}")
    model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()
    
    # --- Load Data ---
    print(f"Loading Test Data...")
    test_dataset = SimpleFolderDataset(af2_data_folder)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_labels = []
    all_probs_raw = []
    all_probs_gated = []
    
    print("Running Inference...")
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.cuda()
            
            # Forward
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Confidence Gating
            p = model._normalize_plddt(batch.plddt).squeeze()
            is_reliable = (p > 0.65).float()
            probs_gated = probs * is_reliable
            
            # Store
            all_labels.append(batch.y.cpu().numpy())
            all_probs_raw.append(probs.cpu().numpy())
            all_probs_gated.append(probs_gated.cpu().numpy())
            
            if i % 20 == 0:
                print(f"Processing batch {i} / {len(test_loader)}...")

    # Concatenate
    y_true = np.concatenate(all_labels)
    y_raw = np.concatenate(all_probs_raw)
    y_gated = np.concatenate(all_probs_gated)
    
    print("\nCalculating Metrics...")

    # ==========================================
    # 3. Professional Plotting Style
    # ==========================================
    # Set global font styles
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 12,
        'axes.linewidth': 1.5,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # --- Colors ---
    c_raw = '#95a5a6'       # Muted Grey for baseline
    c_ours = '#d35400'      # Professional Burnt Orange/Red for Ours
    c_grid = '#ecf0f1'

    # -----------------------------
    # PLOT 1: ROC Curve
    # -----------------------------
    fpr_raw, tpr_raw, _ = roc_curve(y_true, y_raw)
    auc_raw = auc(fpr_raw, tpr_raw)
    
    fpr_gated, tpr_gated, _ = roc_curve(y_true, y_gated)
    auc_gated = auc(fpr_gated, tpr_gated)
    
    ax1.plot(fpr_raw, tpr_raw, color=c_raw, linestyle='--', lw=2.5, label=f'Baseline (Raw) (AUC = {auc_raw:.2f})')
    ax1.plot(fpr_gated, tpr_gated, color=c_ours, lw=3, label=f'Ours (Confidence-Aware) (AUC = {auc_gated:.2f})')
    
    # Fill under curve for emphasis
    ax1.fill_between(fpr_gated, tpr_gated, alpha=0.1, color=c_ours)
    
    # Styling
    ax1.plot([0, 1], [0, 1], color='navy', lw=1, linestyle=':', alpha=0.5)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('Receiver Operating Characteristic (ROC)', fontsize=14, pad=15)
    ax1.legend(loc="lower right", frameon=False, fontsize=11)
    ax1.grid(color=c_grid, linestyle='--', linewidth=1)
    
    # Remove top and right spines
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # -----------------------------
    # PLOT 2: Precision-Recall Curve
    # -----------------------------
    prec_raw, rec_raw, _ = precision_recall_curve(y_true, y_raw)
    ap_raw = average_precision_score(y_true, y_raw)
    
    prec_gated, rec_gated, _ = precision_recall_curve(y_true, y_gated)
    ap_gated = average_precision_score(y_true, y_gated)
    
    ax2.plot(rec_raw, prec_raw, color=c_raw, linestyle='--', lw=2.5, label=f'Baseline (Raw) (AP = {ap_raw:.2f})')
    ax2.plot(rec_gated, prec_gated, color=c_ours, lw=3, label=f'Ours (Confidence-Aware) (AP = {ap_gated:.2f})')
    
    # Fill under curve
    ax2.fill_between(rec_gated, prec_gated, alpha=0.1, color=c_ours)
    
    # Baseline Random
    baseline = y_true.sum() / len(y_true)
    ax2.plot([0, 1], [baseline, baseline], color='black', linestyle=':', lw=1.5, alpha=0.5, label=f'Random ({baseline:.2f})')
    
    # Styling
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curve (PR)', fontsize=14, pad=15)
    ax2.legend(loc="upper right", frameon=False, fontsize=11)
    ax2.grid(color=c_grid, linestyle='--', linewidth=1)
    
    # Remove top and right spines
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Final layout
    plt.tight_layout(pad=3.0)
    output_file = 'publication_metrics.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Success! Publication-quality plot saved to '{output_file}'")
    print("-" * 50)
    print(f"RAW   -> ROC AUC: {auc_raw:.4f} | PR AP: {ap_raw:.4f}")
    print(f"OURS  -> ROC AUC: {auc_gated:.4f} | PR AP: {ap_gated:.4f}")
    print("-" * 50)

if __name__ == "__main__":
    # ==========================================
    # Configuration
    # ==========================================
    
    # Path to your BEST checkpoint (Run 10)
    CKPT_PATH = "checkpoints/last.ckpt" 

    # Path to AF2 data
    AF2_DATA_FOLDER = "data/processed_af2"  

    evaluate_and_plot(CKPT_PATH, AF2_DATA_FOLDER)
