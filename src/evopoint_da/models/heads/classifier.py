import torch.nn as nn
from torch_geometric.nn import MLP

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        """
        分割头: Point Features -> Logits
        Args:
            out_channels: 2 (Background, Binding Site)
        """
        super().__init__()
        # 简单的 3 层 MLP
        self.mlp = MLP([in_channels, 64, 32, out_channels], dropout=0.5, norm=None)

    def forward(self, x):
        return self.mlp(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels):
        """
        域判别器: Features -> Domain Logit (Real/Fake)
        """
        super().__init__()
        # 输出 1 维 Logit (用于 BCEWithLogitsLoss)
        self.mlp = MLP([in_channels, 64, 32, 1], dropout=0.5, norm=None)

    def forward(self, x):
        return self.mlp(x)