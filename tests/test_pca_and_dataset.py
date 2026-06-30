import torch
import time

from protcross.data import PCAReducer
from protcross.data.dataset import EvoPointDataset


def test_pca_reducer_round_trip(tmp_path):
    reducer = PCAReducer(n_components=2)
    reducer.fit([torch.randn(5, 4), torch.randn(5, 4)])
    path = tmp_path / "pca.pkl"
    reducer.save(path)

    loaded = PCAReducer(n_components=2)
    loaded.load(path)
    transformed = loaded.transform(torch.randn(3, 4))

    assert transformed.shape == (3, 2)


def test_pca_reducer_save_creates_parent_directory(tmp_path):
    reducer = PCAReducer(n_components=2)
    reducer.fit([torch.randn(5, 4), torch.randn(5, 4)])
    path = tmp_path / "artifacts" / "pca.pkl"

    reducer.save(path)

    assert path.exists()


def test_dataset_can_load_unlabeled_target_domain(tmp_path):
    torch.save(
        {
            "x": torch.randn(4, 128),
            "pos": torch.randn(4, 3),
            "plddt": torch.ones(4, 1),
            "y": torch.zeros(4),
        },
        tmp_path / "target.pt",
    )

    dataset = EvoPointDataset(
        root=tmp_path,
        split="train",
        require_labels=False,
        require_positive_labels=False,
    )

    assert len(dataset) == 1
    assert dataset[0].x.shape == (4, 128)


def test_dataset_cache_rebuilds_when_source_pt_changes(tmp_path):
    def write_sample(value: float) -> None:
        torch.save(
            {
                "x": torch.full((4, 128), value),
                "pos": torch.randn(4, 3),
                "plddt": torch.ones(4, 1),
                "y": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            tmp_path / "sample.pt",
        )

    write_sample(1.0)
    dataset = EvoPointDataset(root=tmp_path, split="all")
    assert float(dataset[0].x[0, 0]) == 1.0

    time.sleep(0.001)
    write_sample(9.0)
    dataset = EvoPointDataset(root=tmp_path, split="all")

    assert float(dataset[0].x[0, 0]) == 9.0
