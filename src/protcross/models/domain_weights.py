"""Confidence weighting strategies for domain adaptation."""

from __future__ import annotations

import math

import torch


def normalize_plddt(plddt_tensor: torch.Tensor) -> torch.Tensor:
    return plddt_tensor / 100.0 if plddt_tensor.max() > 1.0 else plddt_tensor


def confidence_weights(plddt_tensor: torch.Tensor, strategy: str = "pLDDT^4") -> torch.Tensor:
    """Convert normalized pLDDT scores into target-domain loss weights."""
    p_norm = normalize_plddt(plddt_tensor)

    if strategy == "sine":
        return torch.sin(torch.tensor(math.pi / 2, device=p_norm.device) * p_norm)
    if strategy == "cosine":
        return 1.0 - torch.cos(torch.tensor(math.pi / 2, device=p_norm.device) * p_norm)
    if strategy.startswith("pLDDT"):
        try:
            exponent = float(strategy.split("^")[1]) if "^" in strategy else 4.0
        except (IndexError, ValueError):
            exponent = 4.0
        return p_norm**exponent
    if strategy.startswith("threshold"):
        try:
            threshold = float(strategy.split("_")[1])
        except (IndexError, ValueError):
            threshold = 0.6
        return (p_norm > threshold).float()
    if strategy.startswith("exponential"):
        try:
            beta = float(strategy.split("_")[1])
        except (IndexError, ValueError):
            beta = 4.0
        return torch.exp(beta * (p_norm - 1.0))
    if strategy.startswith("sigmoid"):
        try:
            _, threshold, slope = strategy.split("_")
            threshold = float(threshold)
            slope = float(slope)
        except (ValueError, TypeError):
            threshold, slope = 0.7, 20.0
        return torch.sigmoid(slope * (p_norm - threshold))

    return p_norm**4.0

