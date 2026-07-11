import numpy as np
import pytest
import torch

from protcross.models.backbones import pointnet2


def test_manual_fps_matches_deterministic_gold_sampling():
    positions = torch.tensor(
        [[99.0], [0.0], [1.0], [2.0], [50.0], [51.0]],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 1, 1, 1, 1, 1], dtype=torch.long)

    sampled = pointnet2.manual_fps(positions, batch, ratio=0.5)

    # ceil(1 * .5) from batch 0 and ceil(5 * .5) from batch 1.  Each batch
    # starts at its first point when random_start=False.
    assert sampled.tolist() == [0, 1, 5, 3]


def test_manual_radius_caps_neighbors_per_target_in_source_order():
    source = torch.tensor([[0.9], [-0.2], [0.1], [-0.8], [1.0], [0.0]])
    target = torch.tensor([[0.0], [0.85]])
    source_batch = torch.zeros(len(source), dtype=torch.long)
    target_batch = torch.zeros(len(target), dtype=torch.long)

    row, col = pointnet2.manual_radius(
        source,
        target,
        1.0,
        source_batch,
        target_batch,
        max_num_neighbors=3,
    )

    # The radius comparison is strict (< r), and the cap belongs to each
    # target.  Source order is the fallback's deterministic policy for the
    # subset that torch_cluster documents as unspecified when over capacity.
    assert row.tolist() == [0, 0, 0, 1, 1, 1]
    assert col.tolist() == [0, 1, 2, 0, 2, 4]


def test_manual_knn_is_distance_sorted_and_stable_for_ties():
    source = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]])
    target = torch.tensor([[0.0]])
    source_batch = torch.zeros(len(source), dtype=torch.long)
    target_batch = torch.zeros(len(target), dtype=torch.long)

    row, col = pointnet2.manual_knn(source, target, 8, source_batch, target_batch)

    # k larger than the available source set is clipped, as in torch_cluster.
    assert row.tolist() == [0, 0, 0, 0, 0]
    assert col.tolist() == [4, 0, 1, 2, 3]


def test_chunked_distance_search_matches_single_matrix(monkeypatch):
    torch.manual_seed(19)
    source = torch.randn(17, 3)
    target = torch.randn(11, 3)
    source_batch = torch.tensor([0] * 9 + [1] * 8, dtype=torch.long)
    target_batch = torch.tensor([0] * 6 + [1] * 5, dtype=torch.long)

    monkeypatch.setattr(pointnet2, "DISTANCE_TARGET_CHUNK_SIZE", 10_000)
    radius_reference = pointnet2.manual_radius(
        source, target, 1.75, source_batch, target_batch, max_num_neighbors=5
    )
    knn_reference = pointnet2.manual_knn(source, target, 4, source_batch, target_batch)

    monkeypatch.setattr(pointnet2, "DISTANCE_TARGET_CHUNK_SIZE", 2)
    radius_chunked = pointnet2.manual_radius(
        source, target, 1.75, source_batch, target_batch, max_num_neighbors=5
    )
    knn_chunked = pointnet2.manual_knn(source, target, 4, source_batch, target_batch)

    assert all(torch.equal(actual, expected) for actual, expected in zip(radius_chunked, radius_reference))
    assert all(torch.equal(actual, expected) for actual, expected in zip(knn_chunked, knn_reference))


@pytest.mark.skipif(not pointnet2.HAS_CLUSTER, reason="torch_cluster is unavailable")
def test_manual_fallback_primitives_match_torch_cluster():
    positions = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [7.0, 0.0], [20.0, 0.0], [21.0, 0.0]],
        dtype=torch.float32,
    )
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)

    expected_fps = pointnet2.cluster_fps(positions, batch, ratio=0.5, random_start=False)
    actual_fps = pointnet2.manual_fps(positions, batch, ratio=0.5)
    assert torch.equal(actual_fps, expected_fps)

    targets = torch.tensor([[0.25, 0.0], [20.2, 0.0]], dtype=torch.float32)
    target_batch = torch.tensor([0, 1], dtype=torch.long)
    expected_radius = pointnet2.cluster_radius(
        positions,
        targets,
        6.5,
        batch,
        target_batch,
        max_num_neighbors=3,
    )
    actual_radius = pointnet2.manual_radius(
        positions,
        targets,
        6.5,
        batch,
        target_batch,
        max_num_neighbors=3,
    )
    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_radius, expected_radius))

    expected_knn = pointnet2.cluster_knn(positions, targets, 3, batch, target_batch)
    actual_knn = pointnet2.manual_knn(positions, targets, 3, batch, target_batch)
    assert all(torch.equal(actual, expected) for actual, expected in zip(actual_knn, expected_knn))


@pytest.mark.skipif(not pointnet2.HAS_CLUSTER, reason="torch_cluster is unavailable")
def test_backbone_output_matches_with_and_without_torch_cluster(monkeypatch):
    torch.manual_seed(7)
    backbone = pointnet2.PointNet2Backbone(in_channels=4, hidden_dim=8, out_channels=6).eval()
    features = torch.randn(11, 4)
    positions = torch.tensor(
        [[float(index), float((index * index) % 7), float(index % 3)] for index in range(11)],
        dtype=torch.float32,
    )
    batch = torch.zeros(11, dtype=torch.long)

    with torch.no_grad():
        monkeypatch.setattr(pointnet2, "HAS_CLUSTER", True)
        cluster_output, cluster_global = backbone(features, positions, batch)
        monkeypatch.setattr(pointnet2, "HAS_CLUSTER", False)
        fallback_output, fallback_global = backbone(features, positions, batch)

    np.testing.assert_allclose(fallback_output.numpy(), cluster_output.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(fallback_global.numpy(), cluster_global.numpy(), rtol=1e-5, atol=1e-6)
