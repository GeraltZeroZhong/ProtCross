import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
import torch
import os
import sys
import glob
from tqdm import tqdm

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evopoint_da.data.dataset import EvoPointDataset
from evopoint_da.models.module import EvoPointDALitModule

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # 1. Handle Working Directory
    try: orig_cwd = hydra.utils.get_original_cwd()
    except: orig_cwd = os.getcwd()
    
    # 2. Find Best Checkpoint
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "best-*.ckpt"))
    if not ckpts:
        if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
            target_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        else:
            print("❌ No checkpoints found.")
            return
    else:
        target_ckpt = sorted(ckpts, key=os.path.getmtime)[-1]

    print(f"[{os.path.basename(__file__)}] Loading: {target_ckpt}")
    
    # 3. Load Model
    model = EvoPointDALitModule.load_from_checkpoint(target_ckpt, map_location='cpu')
    model.eval()

    # 4. Determine Test Dataset Path
    if "data_dir_af2" in cfg.data and cfg.data.data_dir_af2:
        test_path = cfg.data.data_dir_af2
        domain_name = "AlphaFold2 (Target)"
    else:
        print("\n⚠️  WARNING: 'data_dir_af2' not found. Testing on PDB (Source).")
        test_path = cfg.data.data_dir_pdb
        domain_name = "PDB (Source)"

    if not os.path.isabs(test_path): test_path = os.path.join(orig_cwd, test_path)

    print(f"🔎 Domain: {domain_name}")
    print(f"📂 Path: {test_path}")

    # 5. Load Dataset
    try:
        dataset = EvoPointDataset(root=test_path, split="test")
    except:
        print(f"⚠️  'test' split not found. Trying 'train' split.")
        dataset = EvoPointDataset(root=test_path, split="train")

    # 6. Setup Loader
    batch_size = cfg.data.get("batch_size", 1)
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )

    # ==========================================
    # 🔬 DATASET INSPECTION (pLDDT Check)
    # ==========================================
    print("\n" + "="*50)
    print("🔬 INSPECTING DATASET (Calculating pLDDT Stats...)")
    print("="*50)
    
    total_pts = 0
    p_min, p_max = float('inf'), float('-inf')
    counts = {'high': 0, 'med': 0, 'low': 0}
    
    # Use a separate loop to inspect data without affecting metrics
    for batch in tqdm(loader, desc="Scanning"):
        p = batch.plddt.view(-1)
        
        # 1. Raw Range
        curr_min, curr_max = p.min().item(), p.max().item()
        if curr_min < p_min: p_min = curr_min
        if curr_max > p_max: p_max = curr_max
        
        # 2. Simulate Normalization
        # Logic matches module.py: if max > 1.0, divide by 100
        is_0_100 = (curr_max > 1.0) or (p_max > 1.0) 
        p_norm = p / 100.0 if is_0_100 else p
        
        # 3. Bin Counts
        counts['high'] += (p_norm >= 0.9).sum().item()
        counts['med'] += ((p_norm >= 0.7) & (p_norm < 0.9)).sum().item()
        counts['low'] += (p_norm < 0.7).sum().item()
        total_pts += p.numel()

    print(f"\n📊 INSPECTION RESULTS:")
    print(f"  - Total Points: {total_pts}")
    print(f"  - Raw pLDDT Range: [{p_min:.4f}, {p_max:.4f}]")
    
    if p_max > 1.0:
        print("  - Detected Scale: 0-100 (✅ Logic will normalize this)")
    else:
        print("  - Detected Scale: 0-1 (✅ Already normalized)")
        
    print(f"  - Bin Distribution:")
    print(f"    - High (>0.9): {counts['high']} ({counts['high']/total_pts*100:.1f}%)")
    print(f"    - Med (0.7-0.9): {counts['med']} ({counts['med']/total_pts*100:.1f}%)")
    print(f"    - Low (<0.7): {counts['low']} ({counts['low']/total_pts*100:.1f}%)")
    
    if counts['med'] == 0 and counts['low'] == 0:
        print("\n❌ CRITICAL WARNING: No Medium or Low confidence points found!")
        print("   -> Are you pointing to PDB files instead of AlphaFold?")
        print("   -> Or is your 'data_dir_af2' actually pointing to processed PDBs?")
    
    print("="*50 + "\n")

    # 7. Run Actual Test
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False
    )
    trainer.test(model=model, dataloaders=loader)
    
    # 8. Extract Metrics
    overall = model.test_iou.compute()[1].item() * 100
    
    def get_res(m_iou, m_tot, m_pred):
        try:
            tot = int(m_tot.compute().item())
            iou = m_iou.compute()[1].item() * 100 if tot > 0 else 0.0
            pred = int(m_pred.compute().item())
            return iou, tot, pred
        except: return 0.0, 0, 0

    h_iou, h_tot, h_pred = get_res(model.test_iou_high, model.count_total_high, model.count_pred_high)
    m_iou, m_tot, m_pred = get_res(model.test_iou_med, model.count_total_med, model.count_pred_med)
    l_iou, l_tot, l_pred = get_res(model.test_iou_low, model.count_total_low, model.count_pred_low)

    l_fpr = (l_pred / l_tot * 100) if l_tot > 0 else 0.0
    l_fpr_str = f"{l_fpr:.2f}" if l_tot > 0 else "N/A"

    print("\n<<<METRICS_START>>>")
    print(f"Overall_IoU:{overall:.2f}")
    print(f"High_IoU:{h_iou:.2f}")
    print(f"High_Total:{h_tot}")
    print(f"Med_IoU:{m_iou:.2f}")
    print(f"Med_Total:{m_tot}")
    print(f"Low_IoU:{l_iou:.2f}") # Usually 0.0
    print(f"Low_FPR:{l_fpr_str}")
    print(f"Low_Total:{l_tot}")
    print("<<<METRICS_END>>>\n")

if __name__ == "__main__":
    main()
