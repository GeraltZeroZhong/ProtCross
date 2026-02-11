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

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointDALitModule
from evopoint_da.data.dataset import EvoPointDataset


SEEDS = [42, 2025, 1]
MAX_EPOCHS = 70
BATCH_SIZE = 16
DA_WEIGHT = 0.2

STRATEGY_GRID = {

    "Baselines": ["Standard_DANN"],
    "Polynomial": ["pLDDT^1", "pLDDT^2", "pLDDT^3", "pLDDT^4", "pLDDT^5", "pLDDT^6"],
    "Hard_Threshold": ["threshold_0.5", "threshold_0.6", "threshold_0.7", "threshold_0.8", "threshold_0.9"],
    "Sigmoid": ["sigmoid_0.6_20", "sigmoid_0.7_20", "sigmoid_0.8_20", "sigmoid_0.7_10", "sigmoid_0.7_30"],
    "Exponential": ["exponential_2", "exponential_4", "exponential_6", "exponential_8"],
    "Trigonometric": ["sine", "cosine"]
}


class AdvancedWeightingModule(EvoPointDALitModule):
    def __init__(self, weight_strategy="pLDDT^4", **kwargs):
        super().__init__(**kwargs)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            src, tgt = batch["source"], batch.get("target")
        else:
            src, tgt = batch, None
        
        src_x = src.x if self.hparams.use_esm else None
        src_pos = src.pos + torch.randn_like(src.pos) * self.hparams.pos_noise
        src_pos = self._random_rotate(src_pos) 
        
        src_feats, _ = self.backbone(src_x, src_pos, src.batch)
        seg_logits = self.seg_head(src_feats)
        loss_seg = self.focal_loss(seg_logits, src.y.long())
        
        loss_da = torch.tensor(0.0, device=self.device)
        
        if self.hparams.use_da and tgt is not None:
            d_s = self.domain_disc(self.grl(src_feats))
            loss_d_s = F.binary_cross_entropy_with_logits(d_s, torch.ones_like(d_s))
            
            tgt_x = tgt.x if self.hparams.use_esm else None
            tgt_pos = tgt.pos + torch.randn_like(tgt.pos) * (self.hparams.pos_noise * 0.7)
            tgt_pos = self._random_rotate(tgt_pos)
            tgt_feats, _ = self.backbone(tgt_x, tgt_pos, tgt.batch)
            d_t = self.domain_disc(self.grl(tgt_feats))
            
            if self.hparams.use_plddt_weight:
                p_norm = self._normalize_plddt(tgt.plddt)
                ws = self.hparams.weight_strategy
                
                if ws == 'sine':
                    weights = torch.sin(torch.tensor(np.pi/2, device=self.device) * p_norm)
                elif ws == 'cosine':
                    weights = 1.0 - torch.cos(torch.tensor(np.pi/2, device=self.device) * p_norm)
                elif ws.startswith('pLDDT'):
                    try: k = float(ws.split('^')[1]) if '^' in ws else 1.0
                    except: k = 1.0
                    weights = p_norm ** k
                elif ws.startswith('threshold'):
                    try: tau = float(ws.split('_')[1])
                    except: tau = 0.7
                    weights = (p_norm > tau).float()
                elif ws.startswith('sigmoid'):
                    try: 
                        parts = ws.split('_')
                        tau, k = float(parts[1]), float(parts[2])
                    except: tau, k = 0.7, 20.0
                    weights = torch.sigmoid(k * (p_norm - tau))
                else:
                    weights = p_norm ** 4.0 

                weights = weights.view(-1, 1)
                
                raw_bce = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t), reduction='none')
                loss_d_t = (raw_bce * weights).sum() / (weights.sum() + 1e-6)
            else:
                # Standard DANN: 无权重
                loss_d_t = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t))
            
            loss_da = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_seg + self.hparams.da_weight * loss_da
        
        real_bs = src.num_graphs if hasattr(src, "num_graphs") else BATCH_SIZE
        self.log("train/loss_total", total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=real_bs)
        
        return total_loss

def compute_best_iou(y_true, y_probs):
    best_iou = 0.0
    best_thresh = 0.5
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

def evaluate_iou(ckpt_path, test_folder, device='cuda'):
    try:
        model = AdvancedWeightingModule.load_from_checkpoint(ckpt_path)
        model.eval().to(device)
    except Exception as e:
        print(f"Warning: Model load failed {ckpt_path}: {e}")
        return 0.0, 0.0

    try: dataset = EvoPointDataset(root=test_folder, split="test")
    except: dataset = EvoPointDataset(root=test_folder, split="train")
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_labels.append(batch.y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    if len(all_labels) == 0:
        return 0.0, 0.0

    y_true = np.concatenate(all_labels)
    y_scores = np.concatenate(all_probs)
    
    best_iou, best_thresh = compute_best_iou(y_true, y_scores)
    return best_iou * 100, best_thresh

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_dir_pdb = os.path.abspath("./data/processed_pdb")
    data_dir_af2 = os.path.abspath("./data/processed_af2")
    logs_root = "./logs/grand_comparison_multiseed" # 区分日志目录
    
    tasks = [(f, v) for f, vars_ in STRATEGY_GRID.items() for v in vars_]
    results = []
    
    print(f"MULTI-SEED STRATEGY ANALYSIS STARTING...")
    print(f"Seeds: {SEEDS}")
    print(f"Configurations per seed: {len(tasks)}")
    

    for seed in SEEDS:
        print(f"\n >>> Running with Seed: {seed} <<<")
   
        for idx, (family, strategy) in enumerate(tasks):
            seed_everything(seed, workers=True)
            print(f"\n👉 [Seed {seed}] [{idx+1}/{len(tasks)}] Strategy: {strategy}")
            
            # 这里的 exp_name 加上种子后缀，确保不混淆
            base_exp_name = strategy.replace("^", "").replace(">", "gt")
            exp_name = f"{base_exp_name}_seed{seed}"
            
            ckpt_dir = os.path.join(logs_root, exp_name, "checkpoints")
            best_ckpt = os.path.join(ckpt_dir, "best.ckpt")
            
            use_da_flag = True
            use_plddt_flag = True
            
            if strategy == "Standard_DANN":
                use_da_flag = True
                use_plddt_flag = False
            
            if os.path.exists(best_ckpt):
                print(f" Found existing checkpoint, skipping training.")
            else:
                dm = EvoPointDataModule(data_dir_pdb, data_dir_af2, BATCH_SIZE, 2)
                model = AdvancedWeightingModule(
                    weight_strategy=strategy,
                    lr=3e-4, 
                    weight_decay=1e-4, 
                    da_weight=DA_WEIGHT,
                    feature_dim=128, 
                    pos_noise=0.08, 
                    use_esm=True, 
                    use_da=use_da_flag,             
                    use_plddt_weight=use_plddt_flag 
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

            if best_ckpt and os.path.exists(best_ckpt):
                iou, thresh = evaluate_iou(best_ckpt, data_dir_af2, device)
                print(f"   ✅ Result: IoU={iou:.2f}% (Thresh={thresh:.2f})")
                results.append({
                    "Family": family,
                    "Strategy": strategy,
                    "Seed": seed,
                    "IoU": iou,
                    "Best_Thresh": thresh
                })
            else:
                print("   ❌ Failed.")

    if not results: return
    

    df = pd.DataFrame(results)
    df.to_csv("multiseed_raw_results.csv", index=False)
    
    print("\n" + "="*60)
    print("AGGREGATED REPORT (Mean ± Std)")
    print("="*60)
    
    agg_df = df.groupby(["Family", "Strategy"]).agg(
        Mean_IoU=("IoU", "mean"),
        Std_IoU=("IoU", "std"),
        Min_IoU=("IoU", "min"),
        Max_IoU=("IoU", "max")
    ).reset_index()
    
    # 按 Mean_IoU 降序排列
    agg_df = agg_df.sort_values("Mean_IoU", ascending=False)
    
    print(agg_df.to_string(float_format="%.2f"))
    print("="*60)
    
    agg_df.to_csv("multiseed_summary_report.csv", index=False)
    print("\n All runs completed. Reports saved.")

if __name__ == "__main__":
    main()
