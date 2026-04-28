import numpy as np

from evopoint_da.evaluation import calculate_plddt_metrics, compute_best_iou


def test_compute_best_iou_finds_reasonable_threshold():
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([0.1, 0.8, 0.7, 0.2])

    iou, threshold = compute_best_iou(y_true, y_probs)

    assert iou == 1.0
    assert 0.2 < threshold < 0.8


def test_calculate_plddt_metrics_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0])
    y_probs = np.array([0.1, 0.8, 0.7, 0.6])
    plddt = np.array([95, 92, 75, 50])

    metrics = calculate_plddt_metrics(y_true, y_probs, plddt, threshold=0.5)

    assert metrics["High_Total"] == 2
    assert metrics["Med_Total"] == 1
    assert metrics["Low_Total"] == 1

