import numpy as np
import pytest
from omegaconf import OmegaConf

from protcross.evaluation import calculate_plddt_metrics, compute_best_iou
from protcross.evaluation import adaptive


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


def test_evaluate_adaptive_does_not_fall_back_to_train_split_by_default(monkeypatch, tmp_path):
    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    requested_splits = []

    def fake_dataset(root, split):
        requested_splits.append(split)
        raise RuntimeError("corrupt test cache")

    monkeypatch.setattr(adaptive, "find_checkpoint", lambda *args, **kwargs: str(tmp_path / "model.ckpt"))
    monkeypatch.setattr(adaptive.EvoPointDALitModule, "load_from_checkpoint", lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(adaptive, "EvoPointDataset", fake_dataset)

    cfg = OmegaConf.create(
        {
            "ckpt_path": str(tmp_path / "model.ckpt"),
            "data": {"data_dir_af2": str(tmp_path), "batch_size": 1, "num_workers": 0},
        }
    )

    with pytest.raises(RuntimeError, match="Test split unavailable"):
        adaptive.evaluate_adaptive(cfg, project_root=tmp_path)

    assert requested_splits == ["test"]
