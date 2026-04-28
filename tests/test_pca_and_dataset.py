import torch

from evopoint_da.data import PCAReducer
from evopoint_da.data.dataset import EvoPointDataset


def test_pca_reducer_round_trip(tmp_path):
    reducer = PCAReducer(n_components=2)
    reducer.fit([torch.randn(5, 4), torch.randn(5, 4)])
    path = tmp_path / "pca.pkl"
    reducer.save(path)

    loaded = PCAReducer(n_components=2)
    loaded.load(path)
    transformed = loaded.transform(torch.randn(3, 4))

    assert transformed.shape == (3, 2)


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

