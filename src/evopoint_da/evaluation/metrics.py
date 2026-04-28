"""Shared metrics for ProtCross evaluation scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_best_iou(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return the best IoU and threshold over a probability grid."""
    thresholds = thresholds if thresholds is not None else np.arange(0.1, 1.0, 0.05)
    best_iou = 0.0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_probs > threshold).astype(int)
        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))
        if union == 0:
            iou = 0.0 if np.sum(y_true) > 0 else 1.0
        else:
            iou = intersection / union
        if iou > best_iou:
            best_iou = float(iou)
            best_threshold = float(threshold)

    return best_iou, best_threshold


def calculate_plddt_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    plddt: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Compute high/medium/low pLDDT stratified metrics."""
    y_pred = (y_probs > threshold).astype(int)
    p_norm = plddt / 100.0 if plddt.max() > 1.5 else plddt

    masks = {
        "High": p_norm >= 0.9,
        "Med": (p_norm >= 0.7) & (p_norm < 0.9),
        "Low": p_norm < 0.7,
    }

    def iou(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        y_t = y_true[mask]
        y_p = y_pred[mask]
        intersection = np.sum((y_t == 1) & (y_p == 1))
        union = np.sum((y_t == 1) | (y_p == 1))
        return float(intersection / union * 100) if union > 0 else 0.0

    def fpr(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return 0.0
        y_t = y_true[mask]
        y_p = y_pred[mask]
        tn, fp, _, _ = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        negatives = tn + fp
        return float(fp / negatives * 100) if negatives > 0 else 0.0

    return {
        "High_IoU": iou(masks["High"]),
        "High_Total": int(masks["High"].sum()),
        "Med_IoU": iou(masks["Med"]),
        "Med_Total": int(masks["Med"].sum()),
        "Low_FPR": fpr(masks["Low"]),
        "Low_Total": int(masks["Low"].sum()),
        "Low_IoU": iou(masks["Low"]),
    }

