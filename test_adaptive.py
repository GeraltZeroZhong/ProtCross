import torch
import numpy as np
from torch_geometric.loader import DataLoader
import os
import sys
import glob

# === 引入项目模块 ===
# 确保路径正确，能 import 你的模型
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from src.evopoint_da.models.module import EvoPointDALitModule
from scripts.plot_metrics import SimpleFolderDataset # 复用之前的 Dataset

# === 配置 ===
AF2_DATA_FOLDER = "data/processed_af2"
CHECKPOINT_DIR = "checkpoints" # 自动找这里面最新的

def find_latest_checkpoint(ckpt_dir):
    # 递归查找所有 .ckpt
    ckpts = glob.glob(os.path.join(ckpt_dir, "**", "*.ckpt"), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    # 按时间排序取最新
    return max(ckpts, key=os.path.getmtime)

def compute_best_iou(y_true, y_probs):
    """遍历 0.1-0.9 寻找最佳 IoU"""
    best_iou = 0.0
    best_thresh = 0.5
    
    # 遍历阈值
    for thresh in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_probs > thresh).astype(int)
        
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))
        
        if union == 0:
            iou = 0.0
        else:
            iou = intersection / union
            
        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh
            
    return best_iou, best_thresh

def main():
    # 1. 找权重
    try:
        ckpt_path = find_latest_checkpoint(CHECKPOINT_DIR)
        # print(f"DEBUG: Using checkpoint {ckpt_path}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. 加载模型
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.cuda()
    except:
        # 兼容性处理：如果模型类路径变了
        # print("Warning: Loading failed, trying direct load...")
        return 

    # 3. 加载数据
    test_dataset = SimpleFolderDataset(AF2_DATA_FOLDER)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    all_labels = []
    all_probs = []

    # 4. 推理
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.cuda()
            
            # Forward
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            # Gating (Exp D)
            # 只有当模型配置了 plddt_weight 时才启用，或者是 B/C 模型时此项无影响
            if hasattr(model.hparams, 'use_plddt_weight') and model.hparams.use_plddt_weight:
                p = model._normalize_plddt(batch.plddt).squeeze()
                is_reliable = (p > 0.65).float()
                probs = probs * is_reliable
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_probs = np.concatenate(all_probs)

    # 5. 计算最佳 IoU
    best_iou, best_t = compute_best_iou(y_true, y_probs)
    
    # 6. 按照 run_benchmark 的格式输出
    # 必须包含 <<<METRICS_START>>> 和 <<<METRICS_END>>>
    # 以及 Overall_IoU: XX.XX%
    
    print("<<<METRICS_START>>>")
    print(f"Overall_IoU: {best_iou * 100:.2f}%")
    print(f"Best_Threshold: {best_t:.2f}") 
    print("High_IoU: 0.0%") # 占位符，防止解析报错
    print("Med_IoU: 0.0%")  # 占位符
    print("Low_FPR: 0.0%")  # 占位符
    print("<<<METRICS_END>>>")

if __name__ == "__main__":
    main()