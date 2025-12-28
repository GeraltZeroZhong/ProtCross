import os
import sys
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import pandas as pd
import numpy as np
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# === 引入项目模块 ===
# 确保脚本能找到 src 下的代码
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointDALitModule
from evopoint_da.data.dataset import EvoPointDataset

# ================= 配置区域 =================
SEED = 42
MAX_EPOCHS = 100
BATCH_SIZE = 16
DA_WEIGHT = 0.2

STRATEGY_GRID = {
    # 1. 多项式组 (Polynomial Family)
    "Polynomial": ["pLDDT^1", "pLDDT^2", "pLDDT^3", "pLDDT^4", "pLDDT^5", "pLDDT^6"],
    
    # 2. 硬阈值组 (Hard Threshold Family)
    "Hard_Threshold": ["threshold_0.5", "threshold_0.6", "threshold_0.7", "threshold_0.8", "threshold_0.9"],
    
    # 3. Sigmoid 软阈值组 (Sigmoid Family)
    # 格式: sigmoid_{tau}_{k} (center, steepness)
    "Sigmoid": [
        "sigmoid_0.6_20", "sigmoid_0.7_20", "sigmoid_0.8_20", # Fix k=20, vary tau
        "sigmoid_0.7_10", "sigmoid_0.7_30"                    # Fix tau=0.7, vary k
    ],
    
    # 4. 指数衰减组 (Exponential Family)
    # 格式: exponential_{beta}
    "Exponential": ["exponential_2", "exponential_5", "exponential_10"],
    
    # 5. 三角函数组 (Trigonometric Family)
    "Trigonometric": ["sine", "cosine"]
}
# ===========================================

class AdvancedWeightingModule(EvoPointDALitModule):
    """支持所有高级加权策略的模型"""
    def __init__(self, weight_strategy="pLDDT^4", **kwargs):
        super().__init__(**kwargs)
        self.weight_strategy = weight_strategy
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            src, tgt = batch["source"], batch.get("target")
        else:
            src, tgt = batch, None
        
        # --- Source Flow (不变) ---
        src_x = src.x if self.hparams.use_esm else None
        # 简单的增强
        src_pos = src.pos + torch.randn_like(src.pos) * self.hparams.pos_noise
        src_pos = self._random_rotate(src_pos) 
        
        src_feats, _ = self.backbone(src_x, src_pos, src.batch)
        seg_logits = self.seg_head(src_feats)
        loss_seg = self.focal_loss(seg_logits, src.y.long())
        
        # --- DA Flow (高级权重计算) ---
        loss_da = torch.tensor(0.0, device=self.device)
        
        if self.hparams.use_da and tgt is not None:
            # Domain Loss (Source)
            d_s = self.domain_disc(self.grl(src_feats))
            loss_d_s = F.binary_cross_entropy_with_logits(d_s, torch.ones_like(d_s))
            
            # Domain Loss (Target)
            tgt_x = tgt.x if self.hparams.use_esm else None
            tgt_pos = tgt.pos + torch.randn_like(tgt.pos) * (self.hparams.pos_noise * 0.7)
            tgt_pos = self._random_rotate(tgt_pos)
            tgt_feats, _ = self.backbone(tgt_x, tgt_pos, tgt.batch)
            d_t = self.domain_disc(self.grl(tgt_feats))
            
            # === 核心：解析并计算权重 ===
            if self.hparams.use_plddt_weight:
                p = self._normalize_plddt(tgt.plddt) # 0-1
                ws = self.weight_strategy
                
                # 1. Polynomial: pLDDT^k
                if ws.startswith('pLDDT'):
                    try: k = float(ws.split('^')[1]) if '^' in ws else 1.0
                    except: k = 1.0
                    weights = p ** k
                
                # 2. Hard Threshold: threshold_tau
                elif ws.startswith('threshold'):
                    try: tau = float(ws.split('_')[1])
                    except: tau = 0.7
                    weights = (p > tau).float()
                
                # 3. Sigmoid: sigmoid_tau_k
                elif ws.startswith('sigmoid'):
                    try: 
                        parts = ws.split('_')
                        tau, k = float(parts[1]), float(parts[2])
                    except: tau, k = 0.7, 20.0
                    weights = torch.sigmoid(k * (p - tau))
                
                # 4. Exponential: exponential_beta
                elif ws.startswith('exponential'):
                    try: beta = float(ws.split('_')[1])
                    except: beta = 5.0
                    weights = torch.exp(beta * (p - 1.0))
                
                # 5. Trigonometric
                elif ws == 'sine':
                    weights = torch.sin(torch.tensor(np.pi/2, device=self.device) * p)
                elif ws == 'cosine':
                    weights = 1.0 - torch.cos(torch.tensor(np.pi/2, device=self.device) * p)
                
                # Fallback
                else:
                    weights = p ** 4.0

                weights = weights.view(-1, 1)
                
                # 计算加权 Loss
                raw_bce = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t), reduction='none')
                loss_d_t = (raw_bce * weights).sum() / (weights.sum() + 1e-6)
            else:
                loss_d_t = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t))
            
            loss_da = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_seg + self.hparams.da_weight * loss_da

        # === 关键修复：显式记录日志，确保 ModelCheckpoint 能监控到 ===
        self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return total_loss

def evaluate_iou(ckpt_path, test_folder, device='cuda'):
    """快速评估 Standard IoU"""
    try:
        model = AdvancedWeightingModule.load_from_checkpoint(ckpt_path)
        model.eval().to(device)
    except: return 0.0, 0.0

    try: dataset = EvoPointDataset(root=test_folder, split="test")
    except: dataset = EvoPointDataset(root=test_folder, split="train")
    
    # 强制 batch_size=1 以进行准确的评估
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            # Forward
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            probs = torch.softmax(model.seg_head(feats), dim=1)[:, 1]
            
            # 统一评估标准：Hard Cutoff at 0.65 (Standard)
            p = batch.plddt.squeeze() / 100.0 if batch.plddt.max() > 1.5 else batch.plddt.squeeze()
            probs = probs * (p > 0.65).float()
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    
    best_iou = 0.0
    best_thresh = 0.0
    for t in np.arange(0.1, 0.95, 0.05):
        y_pred = (y_scores > t).astype(int)
        inter = np.sum((y_true==1)&(y_pred==1))
        union = np.sum((y_true==1)|(y_pred==1))
        iou = inter/union if union>0 else 0.0
        if iou > best_iou: best_iou, best_thresh = iou, t
            
    return best_iou * 100, best_thresh

def main():
    seed_everything(SEED, workers=True)
    device = 'cpu'
    
    data_dir_pdb = os.path.abspath("./data/processed_pdb")
    data_dir_af2 = os.path.abspath("./data/processed_af2")
    logs_root = "./logs/grand_comparison"
    
    # 展平所有任务
    tasks = []
    for family, variants in STRATEGY_GRID.items():
        for v in variants:
            tasks.append((family, v))
    
    results = []
    
    print(f"⚔️  GRAND STRATEGY COMPARISON STARTING...")
    print(f"Total Configurations: {len(tasks)}")
    print(f"Device: {device}")
    
    for idx, (family, strategy) in enumerate(tasks):
        print(f"\n👉 [{idx+1}/{len(tasks)}] Group: {family} | Strategy: {strategy}")
        exp_name = strategy.replace("^", "").replace(">", "gt")
        
        # 检查是否已跑过 (断点续传)
        ckpt_dir = os.path.join(logs_root, exp_name, "checkpoints")
        best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
        
        if os.path.exists(best_ckpt):
            print(f"   ⏩ Found existing checkpoint, skipping training.")
        else:
            dm = EvoPointDataModule(data_dir_pdb, data_dir_af2, BATCH_SIZE, 2)
            model = AdvancedWeightingModule(
                weight_strategy=strategy,
                lr=3e-4, weight_decay=1e-4, da_weight=DA_WEIGHT,
                feature_dim=128, pos_noise=0.08, use_esm=True, use_da=True, use_plddt_weight=True
            )
            
            logger = CSVLogger(save_dir=logs_root, name=exp_name)
            ckpt_cb = ModelCheckpoint(
                dirpath=ckpt_dir, 
                filename="best", 
                monitor="train/loss_total", 
                mode="min", 
                save_top_k=1
            )
            
            trainer = pl.Trainer(
                max_epochs=MAX_EPOCHS,
                accelerator=device, 
                devices=1,
                logger=logger, 
                callbacks=[ckpt_cb],
                enable_checkpointing=True,
                log_every_n_steps=5 
            )
            trainer.fit(model, datamodule=dm)
            best_ckpt = ckpt_cb.best_model_path

        # Evaluate
        if best_ckpt and os.path.exists(best_ckpt):
            iou, thresh = evaluate_iou(best_ckpt, data_dir_af2, device)
            print(f"   ✅ Result: IoU={iou:.2f}%")
            results.append({
                "Family": family,
                "Strategy": strategy,
                "IoU": iou,
                "Best_Thresh": thresh
            })
        else:
            print("   ❌ Failed.")

    # === 生成最终报告 ===
    if not results: return
    df = pd.DataFrame(results)
    
    # 1. 保存原始数据
    df.to_csv("strategy_comparison_raw.csv", index=False)
    
    print("\n" + "="*60)
    print("🏆 BATTLE OF CHAMPIONS (Best of Each Family)")
    print("="*60)
    
    # 2. 分组取最大值
    champions = df.loc[df.groupby("Family")["IoU"].idxmax()].sort_values("IoU", ascending=False)
    
    # 打印冠军表 (使用 to_string 避免 tabulate 报错)
    print(champions.to_string(index=False))
    
    print("="*60)
    
    # 保存冠军表
    champions.to_csv("strategy_comparison_champions.csv", index=False)
    print("\n✅ Done! Reports saved to .csv files.")

if __name__ == "__main__":
    main()
