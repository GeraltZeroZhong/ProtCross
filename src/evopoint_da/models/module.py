import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.autograd import Function
import torch.optim as optim
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics import SumMetric

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
                 da_weight: float = 1.0,      # Domain Adaptation Weight
                 feature_dim: int = 128,
                 pos_noise: float = 0.1,
                 # --- Experimental Flags ---
                 use_esm: bool = True,        # Table 1/2: Use ESM features?
                 use_da: bool = True,         # Table 2: Use Domain Adaptation?
                 use_plddt_weight: bool = True # Table 2: Use pLDDT weighting?
                 ):
        super().__init__()
        self.save_hyperparameters()
        
        # Determine backbone input channels
        # PointNet++ needs 0 extra channels if only using XYZ, else feature_dim
        backbone_in_channels = feature_dim if use_esm else 0
        
        # Architecture
        self.backbone = PointNet2Backbone(in_channels=backbone_in_channels, out_channels=128)
        self.seg_head = SegmentationHead(in_channels=128, out_channels=2)
        
        # Domain Adaptation Components (Only init if needed)
        if use_da:
            self.domain_disc = DomainDiscriminator(in_channels=128)
            self.grl = GradientReversal(alpha=1.0)
        
        # --- Metrics Setup ---
        # 1. Overall Global IoU (Aggregates all points then computes IoU)
        self.test_iou = MulticlassJaccardIndex(num_classes=2, average='none')
        
        # 2. Bin-wise Global IoU (Separate metrics ensure independent aggregation)
        self.test_iou_high = MulticlassJaccardIndex(num_classes=2, average='none')
        self.test_iou_med = MulticlassJaccardIndex(num_classes=2, average='none')
        self.test_iou_low = MulticlassJaccardIndex(num_classes=2, average='none')

        # 3. Counters (Total Points & Predicted Positives)
        # Used to verify data distribution and calculate FPR
        self.count_total_high = SumMetric()
        self.count_pred_high  = SumMetric()
        self.count_total_med  = SumMetric()
        self.count_pred_med   = SumMetric()
        self.count_total_low  = SumMetric()
        self.count_pred_low   = SumMetric()

    def _normalize_plddt(self, plddt_tensor):
        """
        Auto-detect and normalize pLDDT to [0, 1].
        If max value > 1.0, assume it's [0, 100] and divide by 100.
        """
        if plddt_tensor.max() > 1.0:
            return plddt_tensor / 100.0
        return plddt_tensor

    def focal_loss(self, logits, targets, alpha=0.5, gamma=3.0):
        """Focal Loss handles extreme class imbalance."""
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt)**gamma * ce_loss
        return focal_loss.mean()

    def training_step(self, batch, batch_idx):
        if isinstance(batch, dict):
            src, tgt = batch["source"], batch.get("target")
        else:
            src, tgt = batch, None
        
        # === 1. Source Flow (PDB - Labeled) ===
        # Use ESM features only if flag is True
        src_x = src.x if self.hparams.use_esm else None
        # Add noise for robustness
        src_pos = src.pos + torch.randn_like(src.pos) * self.hparams.pos_noise
        
        src_feats, _ = self.backbone(src_x, src_pos, src.batch)
        seg_logits = self.seg_head(src_feats)
        loss_seg = self.focal_loss(seg_logits, src.y.long())
        
        # === 2. Domain Adaptation (DA) Flow ===
        loss_da = torch.tensor(0.0, device=self.device)
        
        if self.hparams.use_da and tgt is not None:
            # --- Source Domain Discriminator (Real = 1) ---
            d_s = self.domain_disc(self.grl(src_feats))
            loss_d_s = F.binary_cross_entropy_with_logits(d_s, torch.ones_like(d_s))
            
            # --- Target Domain Discriminator (Fake = 0) ---
            tgt_x = tgt.x if self.hparams.use_esm else None
            tgt_pos = tgt.pos + torch.randn_like(tgt.pos) * (self.hparams.pos_noise * 0.7)
            
            tgt_feats, _ = self.backbone(tgt_x, tgt_pos, tgt.batch)
            d_t = self.domain_disc(self.grl(tgt_feats))
            
            if self.hparams.use_plddt_weight:
                # [Crucial Fix] Weighted Domain Adaptation
                # We align high-confidence regions strongly, ignore low-confidence noise.
                
                # 1. Normalize pLDDT
                p_norm = self._normalize_plddt(tgt.plddt)
                weights = p_norm.view(-1, 1)
                
                # 2. Weighted Mean Loss
                # Use .sum() / weights.sum() to preserve gradient magnitude
                raw_bce = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t), reduction='none')
                loss_d_t = (raw_bce * weights).sum() / (weights.sum() + 1e-6)
            else:
                # Standard DANN (Blind Alignment)
                loss_d_t = F.binary_cross_entropy_with_logits(d_t, torch.zeros_like(d_t))
            
            loss_da = 0.5 * (loss_d_s + loss_d_t)

        total_loss = loss_seg + self.hparams.da_weight * loss_da
        
        # Logging
        self.log("train/loss_total", total_loss, prog_bar=True)
        self.log("train/loss_seg", loss_seg)
        self.log("train/loss_da", loss_da)
        
        return total_loss

    def test_step(self, batch, batch_idx):
        # Inference respects the use_esm flag
        batch_x = batch.x if self.hparams.use_esm else None
        feats, _ = self.backbone(batch_x, batch.pos, batch.batch)
        logits = self.seg_head(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = (probs > 0.35).long() # Threshold tuning
        
        # 1. Update Overall IoU
        self.test_iou(preds, batch.y)
        self.log("test/iou_overall", self.test_iou.compute()[1])

        # 2. Confidence-based Analysis
        # Ensure normalization happens before binning!
        p = self._normalize_plddt(batch.plddt).squeeze()
        
        masks = {
            "high": p >= 0.9,
            "med":  (p >= 0.7) & (p < 0.9),
            "low":  p < 0.7
        }
        
        # Helper to update metrics safely
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