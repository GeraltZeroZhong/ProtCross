import torch.nn as nn
from torch_geometric.nn import MLP

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, out_channels=2):
        """
        Point-wise segmentation head.
        Args:
            out_channels: 2 (background, binding site)
        """
        super().__init__()
        self.mlp = MLP([in_channels, 64, 32, out_channels], dropout=0.5, norm=None)

    def forward(self, x):
        return self.mlp(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, in_channels):
        """
        Domain discriminator used by DANN training.
        """
        super().__init__()
        self.mlp = MLP([in_channels, 64, 32, 1], dropout=0.5, norm=None)

    def forward(self, x):
        return self.mlp(x)
