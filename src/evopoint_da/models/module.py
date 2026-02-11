import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Function
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import SumMetric
import numpy as np

from .backbones.pointnet2 import PointNet2Backbone
from .heads.classifier import SegmentationHead, DomainDiscriminator

# ==========================================
# 1. Gradient Reversal Layer (GRL)
# ==========================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    def forward(self, x):
        return GradientReversalFn.apply(x, self.alpha)

# ==========================================
# 2. EvoPointDA Lightning Module
# ==========================================
class EvoPointDALitModule(pl.LightningModule):
    def __init__(self, 
                 lr: float = 3e-4, 
                 weight_decay: float = 1e-4,
                 da_weight: float = 0.2,      
                 feature_dim: int = 128,
                 pos_noise: float = 0.08,
                 weight_strategy: str = "pLDDT^4",
                 # --- Experimental Flags ---
                 use_esm: bool = True,        
                 use_da: bool = True,         
                 use_plddt_weight: bool = True 
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # Determine backbone input channels
        backbone_in_channels = feature_dim if use_esm else 0
        
        # Architecture
        self.backbone = PointNet2Backbone(in_channels=backbone_in_channels, out_channels=128)
        self.seg_head = SegmentationHead(in_channels=128, out_channels=2)
        
        # Domain Adaptation Components
        if use_da:
            self.domain_disc = DomainDiscriminator(in_channels=128)
            self.grl = GradientReversal(alpha=1.0)
        
        # --- Metrics Setup ---
        self.test_iou = MulticlassJaccardIndex(num_classes=2, average='none')
        self.test_iou_high = MulticlassJaccardIndex(num_classes=2, average='none')
        self.test_iou_med = MulticlassJaccardIndex(num_classes=2, average='none')
        self.test_iou_low = MulticlassJaccardIndex(num_classes=2, average='none')

        self.count_total_high = SumMetric()
        self.count_pred_high  = SumMetric()
        self.count_total_med  = SumMetric()
        self.count_pred_med   = SumMetric()
        self.count_total_low  = SumMetric()
        self.count_pred_low   = SumMetric()

    def _normalize_plddt(self, plddt_tensor):
        if plddt_tensor.max() > 1.0:
            return plddt_tensor / 100.0
        return plddt_tensor

    def focal_loss(self, logits, targets, alpha=0.5, gamma=2.0):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt)**gamma * ce_loss
        return focal_loss.mean()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            src, tgt = batch["source"], batch.get("target")
        else:
            src, tgt = batch, None
        
        # === 1. Source Flow ===
        src_x = src.x if self.hparams.use_esm else None
        
        # Augmentation
        src_pos = src.pos + torch.randn_like(src.pos) * self.hparams.pos_noise
        src_pos = self._random_rotate(src_pos) 
        
        src_feats, _ = self.backbone(src_x, src_pos, src.batch)
        seg_logits = self.seg_head(src_feats)
        loss_seg = self.focal_loss(seg_logits, src.y.long())
        
        # === 2. DA Flow ===
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
                ws = self.hparams.weight_strategy  # 获取策略名称

                if ws == 'sine':
                    # Sine: sin(pi/2 * p)
                    weights = torch.sin(torch.tensor(np.pi/2, device=self.device) * p_norm)
                elif ws == 'cosine':
                    # Cosine: 1 - cos(pi/2 * p)
                    weights = 1.0 - torch.cos(torch.tensor(np.pi/2, device=self.device) * p_norm)
                elif ws.startswith('pLDDT'):
                    # 多项式(pLDDT^4)
                    try: k = float(ws.split('^')[1]) if '^' in ws else 4.0
                    except: k = 4.0
                    weights = p_norm ** k
                elif ws.startswith('threshold'):
                    # 硬阈值策略 (例如 threshold_0.6)
                    try: tau = float(ws.split('_')[1])
                    except: tau = 0.6
                    weights = (p_norm > tau).float()
                elif ws.startswith('exponential'):
                    # 指数策略 (例如 exponential_4)
                    try: beta = float(ws.split('_')[1])
                    except: beta = 4.0
                    weights = torch.exp(beta * (p_norm - 1.0))
                elif ws.startswith('sigmoid'):
                    # 格式: sigmoid_0.7_30 (中心_斜率)
                    try: 
                        parts = ws.split('_')
                        tau, k = float(parts[1]), float(parts[2])
                    except: tau, k = 0.7, 20.0
                    # 公式: 1 / (1 + exp(-k * (p - tau)))
                    weights = torch.sigmoid(k * (p_norm - tau))
                else:
                    # 默认回退
                    weights = p_norm ** 4.0
                
                weights = weights.view(-1, 1)
                
                raw_bce = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t), reduction='none')
                loss_d_t = (raw_bce * weights).sum() / (weights.sum() + 1e-6)
            else:
                loss_d_t = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t))
            
            loss_da = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_seg + self.hparams.da_weight * loss_da
        
        self.log("train/loss_total", total_loss, prog_bar=True)
        self.log("train/loss_seg", loss_seg)
        self.log("train/loss_da", loss_da)
        
        return total_loss
    def validation_step(self, batch, batch_idx):
        """计算验证集损失和指标"""
        # 1. 前向传播
        src_x = batch.x if self.hparams.use_esm else None
        feats, _ = self.backbone(src_x, batch.pos, batch.batch)
        logits = self.seg_head(feats)
        
        # 2. 计算分割损失
        # 强制将 y 转为 long 类型，防止 float 报错
        loss = torch.nn.functional.cross_entropy(logits, batch.y.long())
        
        # 3. 记录日志
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        batch_x = batch.x if self.hparams.use_esm else None
        feats, _ = self.backbone(batch_x, batch.pos, batch.batch)
        logits = self.seg_head(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        
        p = self._normalize_plddt(batch.plddt).squeeze() 
        preds = (probs > 0.35).long() 
        
        # Metrics
        self.test_iou(preds, batch.y)
        self.log("test/iou_overall", self.test_iou.compute()[1])

        # Analysis
        masks = {
            "high": p >= 0.9,
            "med":  (p >= 0.7) & (p < 0.9),
            "low":  p < 0.7
        }
        
        def update_bin(name, metric_iou, metric_total, metric_pred):
            m = masks[name]
            if m.sum() > 0:
                metric_iou(preds[m], batch.y[m])
                metric_total.update(m.sum())
                metric_pred.update(preds[m].sum())
        
        update_bin("high", self.test_iou_high, self.count_total_high, self.count_pred_high)
        update_bin("med",  self.test_iou_med,  self.count_total_med,  self.count_pred_med)
        update_bin("low",  self.test_iou_low,  self.count_total_low,  self.count_pred_low)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]

    def _random_rotate(self, pos):
        theta_x = np.random.uniform(0, 2 * np.pi)
        theta_y = np.random.uniform(0, 2 * np.pi)
        theta_z = np.random.uniform(0, 2 * np.pi)
        
        dev = pos.device
        dt = pos.dtype 
        
        def Rz(t): return torch.tensor([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0,0,1]], device=dev, dtype=dt)
        def Ry(t): return torch.tensor([[np.cos(t), 0, np.sin(t)], [0,1,0], [-np.sin(t), 0, np.cos(t)]], device=dev, dtype=dt)
        def Rx(t): return torch.tensor([[1,0,0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]], device=dev, dtype=dt)
        
        rot_mat = Rz(theta_z) @ Ry(theta_y) @ Rx(theta_x)
        return pos @ rot_mat
