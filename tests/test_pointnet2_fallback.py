import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from protcross.models.backbones import pointnet2


def _small_graph(node_count: int, *, seed: int, feature_scale: float = 1.0) -> Data:
    generator = torch.Generator().manual_seed(seed)
    # Keep every graph in the same coordinate region.  A missing/incorrect
    # batch vector would therefore create obvious cross-structure neighbors.
    return Data(
        x=torch.randn(node_count, 4, generator=generator, dtype=torch.float32) * feature_scale,
        pos=torch.randn(node_count, 3, generator=generator, dtype=torch.float32) * 2.0,
    )


def _batched_backbone_outputs(backbone, graphs: list[Data]):
    batch = Batch.from_data_list(graphs)
    node_features, global_features = backbone(batch.x, batch.pos, batch.batch)
    return list(node_features.split([graph.num_nodes for graph in graphs])), list(global_features.unbind())


def _scalar_radius_reference(x, y, r, batch_x, batch_y, max_num_neighbors):
    rows = []
    columns = []
    for batch_index in range(int(max(batch_x.max(), batch_y.max()).item()) + 1):
        source_indices = torch.where(batch_x == batch_index)[0]
        target_indices = torch.where(batch_y == batch_index)[0]
        if source_indices.numel() == 0 or target_indices.numel() == 0:
            continue
        within_radius = torch.cdist(y[target_indices], x[source_indices]) < r
        for target_offset in range(len(target_indices)):
            neighbors = torch.where(within_radius[target_offset])[0][:max_num_neighbors]
            if neighbors.numel() == 0:
                continue
            rows.append(target_indices[target_offset].expand(neighbors.numel()))
            columns.append(source_indices[neighbors])
    if not rows:
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        return empty, empty
    return torch.cat(rows), torch.cat(columns)


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


def test_vectorized_radius_matches_scalar_reference_for_multiple_graphs(monkeypatch):
    generator = torch.Generator().manual_seed(17)
    source = torch.randn(21, 3, generator=generator)
    target = torch.randn(13, 3, generator=generator)
    source_batch = torch.tensor([0] * 11 + [1] * 10, dtype=torch.long)
    target_batch = torch.tensor([0] * 7 + [1] * 6, dtype=torch.long)
    monkeypatch.setattr(pointnet2, "DISTANCE_TARGET_CHUNK_SIZE", 3)

    expected = _scalar_radius_reference(
        source,
        target,
        1.75,
        source_batch,
        target_batch,
        max_num_neighbors=4,
    )
    actual = pointnet2.manual_radius(
        source,
        target,
        1.75,
        source_batch,
        target_batch,
        max_num_neighbors=4,
    )

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


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


def test_backbone_fp32_batch_matches_individual_graphs_and_batch_order():
    torch.manual_seed(31)
    backbone = pointnet2.PointNet2Backbone(in_channels=4, hidden_dim=8, out_channels=6).eval()
    graphs = [_small_graph(13, seed=101), _small_graph(9, seed=202)]

    with torch.no_grad():
        individual = [
            backbone(
                graph.x,
                graph.pos,
                torch.zeros(graph.num_nodes, dtype=torch.long),
            )
            for graph in graphs
        ]
        batched_nodes, batched_global = _batched_backbone_outputs(backbone, graphs)
        reversed_nodes, reversed_global = _batched_backbone_outputs(backbone, list(reversed(graphs)))

    for index in range(len(graphs)):
        torch.testing.assert_close(batched_nodes[index], individual[index][0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(batched_global[index], individual[index][1][0], rtol=1e-6, atol=1e-7)

        reversed_index = len(graphs) - 1 - index
        torch.testing.assert_close(reversed_nodes[reversed_index], individual[index][0], rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(reversed_global[reversed_index], individual[index][1][0], rtol=1e-6, atol=1e-7)


def test_backbone_batch_graphs_are_isolated_even_when_coordinates_overlap():
    torch.manual_seed(37)
    backbone = pointnet2.PointNet2Backbone(in_channels=4, hidden_dim=8, out_channels=6).eval()
    reference = _small_graph(12, seed=303)
    companion = _small_graph(12, seed=404)
    changed_companion = companion.clone()
    changed_companion.x = companion.x * -1_000.0
    changed_companion.pos = companion.pos.flip(0)

    with torch.no_grad():
        original_nodes, original_global = _batched_backbone_outputs(backbone, [reference, companion])
        changed_nodes, changed_global = _batched_backbone_outputs(backbone, [reference, changed_companion])

    torch.testing.assert_close(changed_nodes[0], original_nodes[0], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(changed_global[0], original_global[0], rtol=1e-6, atol=1e-7)


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
