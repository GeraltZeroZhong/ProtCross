"""Inference APIs for applying ProtCross models."""

from .predictor import PredictionResult, PredictorAssets, ProtCrossPredictor, ResiduePrediction, predict_pdb

__all__ = [
    "PredictionResult",
    "PredictorAssets",
    "ProtCrossPredictor",
    "ResiduePrediction",
    "predict_pdb",
]
