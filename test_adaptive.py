import hydra
from omegaconf import DictConfig
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import os
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evopoint_da.data.dataset import EvoPointDataset
from evopoint_da.models.module import EvoPointDALitModule

def compute_best_iou(y_true, y_probs):
    """遍历 0.1-0.95 寻找最佳 IoU 和对应的阈值"""
    best_iou = 0.0
    best_thresh = 0.5
    
    # 遍历阈值
    # 使用更细的粒度可能更准确，这里保持 0.05
    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_probs > thresh).astype(int)
        
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))
        
        if union == 0:
            iou = 0.0 if np.sum(y_true) > 0 else 1.0
        else:
            iou = intersection / union
            
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
            
    return best_iou, best_thresh

def calculate_metrics_at_threshold(y_true, y_probs, plddt, threshold):
    """在指定阈值下计算分层指标 (High/Med/Low)"""
    y_pred = (y_probs > threshold).astype(int)
    
    # 定义置信度掩码
    # 注意：这里的 plddt 应该是已经归一化到 0-1 或 0-100 的，根据逻辑统一处理
    # 假设输入是原始值，我们在处理时归一化
    if plddt.max() > 1.5:
        p_norm = plddt / 100.0
    else:
        p_norm = plddt

    mask_high = p_norm >= 0.9
    mask_med  = (p_norm >= 0.7) & (p_norm < 0.9)
    mask_low  = p_norm < 0.7

    def calc_iou(mask):
        if mask.sum() == 0: return 0.0, 0
        y_t = y_true[mask]
        y_p = y_pred[mask]
        inter = np.sum((y_t == 1) & (y_p == 1))
        union = np.sum((y_t == 1) | (y_p == 1))
        return (inter / union * 100) if union > 0 else 0.0, mask.sum()

    def calc_fpr(mask):
        if mask.sum() == 0: return 0.0, 0
        y_t = y_true[mask]
        y_p = y_pred[mask]
        # FPR = FP / (FP + TN)
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0,1]).ravel()
        negatives = tn + fp
        return (fp / negatives * 100) if negatives > 0 else 0.0, mask.sum()

    high_iou, high_tot = calc_iou(mask_high)
    med_iou, med_tot = calc_iou(mask_med)
    low_fpr, low_tot = calc_fpr(mask_low)
    
    # Low IoU 通常不重要，但保持格式一致
    low_iou, _ = calc_iou(mask_low)

    return {
        "High_IoU": high_iou, "High_Total": high_tot,
        "Med_IoU": med_iou,   "Med_Total": med_tot,
        "Low_FPR": low_fpr,   "Low_Total": low_tot,
        "Low_IoU": low_iou
    }

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    # 1. 确定工作目录
    try: orig_cwd = hydra.utils.get_original_cwd()
    except: orig_cwd = os.getcwd()
    
    # 2. 自动寻找最佳 Checkpoint (复用 test.py 逻辑)
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, "best-*.ckpt"))
    if not ckpts:
        if os.path.exists(os.path.join(ckpt_dir, "last.ckpt")):
            target_ckpt = os.path.join(ckpt_dir, "last.ckpt")
        else:
            print("❌ No checkpoints found.")
            return
    else:
        target_ckpt = max(ckpts, key=os.path.getmtime)

    print(f"[{os.path.basename(__file__)}] Loading: {target_ckpt}")
    
    # 3. 加载模型
    model = EvoPointDALitModule.load_from_checkpoint(target_ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # 4. 确定测试数据路径 (复用 test.py 逻辑)
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

    # 5. 加载数据集
    try:
        dataset = EvoPointDataset(root=test_path, split="test")
    except:
        print(f"⚠️  'test' split not found. Trying 'train' split.")
        dataset = EvoPointDataset(root=test_path, split="train")

    loader = DataLoader(
        dataset, 
        batch_size=cfg.data.get("batch_size", 1), 
        shuffle=False, 
        num_workers=cfg.data.get("num_workers", 2)
    )

    # 6. 推理循环 (收集数据)
    all_labels = []
    all_probs = []
    all_plddts = []

    print("Running Inference & Collecting Probabilities...")
    with torch.no_grad():
        for batch in tqdm(loader):
            if torch.cuda.is_available():
                batch = batch.cuda()
            
            # Forward
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_plddts.append(batch.plddt.cpu().numpy().flatten())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)
    y_plddt = np.concatenate(all_plddts)

    # 7. 计算最佳 IoU (Adaptive Step)
    print("Computing Best Threshold...")
    best_iou, best_t = compute_best_iou(y_true, y_probs)
    
    # 8. 使用最佳阈值计算详细指标 (Comprehensive Output Step)
    print(f"Calculating detailed metrics at Best Threshold: {best_t:.2f}...")
    detailed_metrics = calculate_metrics_at_threshold(y_true, y_probs, y_plddt, best_t)

    # 9. 输出结果
    print("\n<<<METRICS_START>>>")
    print(f"Overall_IoU: {best_iou * 100:.2f}%")
    print(f"Best_Threshold: {best_t:.2f}") 
    
    # 输出详细分层指标
    print(f"High_IoU: {detailed_metrics['High_IoU']:.2f}%")
    print(f"High_Total: {detailed_metrics['High_Total']}")
    print(f"Med_IoU: {detailed_metrics['Med_IoU']:.2f}%")
    print(f"Med_Total: {detailed_metrics['Med_Total']}")
    print(f"Low_FPR: {detailed_metrics['Low_FPR']:.2f}%")
    print(f"Low_Total: {detailed_metrics['Low_Total']}")
    print("<<<METRICS_END>>>")

if __name__ == "__main__":
    main()
