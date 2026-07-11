import math

import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool, MLP


try:
    from torch_cluster import fps as cluster_fps  # noqa: F401 - exported for parity tests
    from torch_cluster import radius as cluster_radius  # noqa: F401 - exported for parity tests
    from torch_cluster import knn as cluster_knn  # noqa: F401 - exported for parity tests
    HAS_CLUSTER = True
except (ImportError, OSError):
    HAS_CLUSTER = False


# ProtCross 0.2.1 deliberately uses one deterministic geometry implementation
# on every platform. torch_cluster's over-capacity radius subset is unspecified
# and produced materially different predictions between optional-dependency
# installs. The imported primitives remain available for parity tests only.
GEOMETRY_BACKEND = "deterministic-pytorch"
DISTANCE_TARGET_CHUNK_SIZE = 512


def manual_fps(x, batch, ratio=0.5):
    """Pure PyTorch deterministic FPS matching ``torch_cluster.fps``.

    ``safe_fps`` requests ``random_start=False`` from ``torch_cluster``.  In
    that mode each batch starts from its first point and samples
    ``ceil(num_points * ratio)`` points.  Keep those details identical here:
    PointNet++ checkpoints are sensitive to both the sampled count and the
    initial centroid.
    """
    if x.size(0) == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)

    new_indices = []

    batch_size = int(batch.max().item()) + 1
    for b_idx in range(batch_size):
        mask = batch == b_idx
        indices = torch.where(mask)[0]
        if indices.numel() == 0:
            continue
        pos_b = x[indices]

        num_points = indices.shape[0]
        num_samples = max(1, math.ceil(num_points * float(ratio)))

        dists = torch.full((num_points,), torch.inf, dtype=x.dtype, device=x.device)
        farthest = torch.zeros((), dtype=torch.long, device=x.device)

        sample_indices = []
        for _ in range(num_samples):
            sample_indices.append(indices[farthest])

            centroid = pos_b[farthest]
            dist = torch.sum((pos_b - centroid) ** 2, dim=-1)
            mask_dist = dist < dists
            dists[mask_dist] = dist[mask_dist]

            farthest = torch.max(dists, -1)[1]

        new_indices.append(torch.stack(sample_indices))

    return torch.cat(new_indices)

def manual_radius(x, y, r, batch_x, batch_y, max_num_neighbors=64):
    """Radius search fallback returning ``(row=target, col=source)``.

    ``torch_cluster.radius`` groups edges by target point and applies
    ``max_num_neighbors`` independently to each target.  Its choice is
    unspecified when a target has too many neighbors; this fallback makes that
    choice deterministic by retaining qualifying sources in input order.
    Applying the cap per source instead changes the PointNet++ graph (especially
    for the 20/40 Angstrom layers).
    """
    if x.size(0) == 0 or y.size(0) == 0:
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        return empty, empty

    row_list, col_list = [], []
    batch_size = int(max(batch_x.max(), batch_y.max()).item()) + 1

    for b_idx in range(batch_size):
        idx_x = torch.where(batch_x == b_idx)[0]
        idx_y = torch.where(batch_y == b_idx)[0]
        if idx_x.numel() == 0 or idx_y.numel() == 0:
            continue

        pos_x = x[idx_x]  # Source
        pos_y = y[idx_y]  # Target

        for start in range(0, pos_y.shape[0], DISTANCE_TARGET_CHUNK_SIZE):
            stop = min(start + DISTANCE_TARGET_CHUNK_SIZE, pos_y.shape[0])
            within_radius = torch.cdist(pos_y[start:stop], pos_x) < r

            for local_target_index in range(stop - start):
                target_index = start + local_target_index
                neighbors = torch.where(within_radius[local_target_index])[0]
                if len(neighbors) > max_num_neighbors:
                    neighbors = neighbors[:max_num_neighbors]

                if neighbors.numel() == 0:
                    continue
                row_list.append(idx_y[target_index].expand(neighbors.numel()))
                col_list.append(idx_x[neighbors])

    if len(row_list) == 0:
        return (torch.empty(0, dtype=torch.long, device=x.device),
                torch.empty(0, dtype=torch.long, device=x.device))

    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return row, col

def manual_knn(x, y, k, batch_x, batch_y):
    """KNN (Fallback): Finds for each element in y the k nearest points in x.
    Returns (row=y_idx, col=x_idx) to match torch_cluster.knn semantics.
    """
    if x.size(0) == 0 or y.size(0) == 0:
        empty = torch.empty(0, dtype=torch.long, device=x.device)
        return empty, empty

    row_list, col_list = [], []
    batch_size = int(max(batch_x.max(), batch_y.max()).item()) + 1

    for b_idx in range(batch_size):
        idx_x = torch.where(batch_x == b_idx)[0]
        idx_y = torch.where(batch_y == b_idx)[0]
        if idx_x.numel() == 0 or idx_y.numel() == 0:
            continue

        pos_x = x[idx_x]  # Source points
        pos_y = y[idx_y]  # Target points

        k_val = min(k, pos_x.size(0))
        for start in range(0, pos_y.shape[0], DISTANCE_TARGET_CHUNK_SIZE):
            stop = min(start + DISTANCE_TARGET_CHUNK_SIZE, pos_y.shape[0])
            # torch_cluster orders neighbors by increasing distance and
            # preserves source order for exact ties. ``topk`` is not stable.
            dist = torch.cdist(pos_y[start:stop], pos_x)
            nn_idx = torch.argsort(dist, dim=1, stable=True)[:, :k_val]

            row = idx_y[start:stop].repeat_interleave(k_val)
            col = idx_x[nn_idx.reshape(-1)]
            row_list.append(row)
            col_list.append(col)

    if len(row_list) == 0:
        dev = x.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return row, col

def safe_fps(x, batch, ratio):
    return manual_fps(x, batch, ratio)

def safe_radius(x, y, r, batch_x, batch_y, max_num_neighbors):
    return manual_radius(x, y, r, batch_x, batch_y, max_num_neighbors)

def safe_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
    """Simulate torch_geometric.nn.unpool.knn_interpolate.

    For each point in pos_y, find its k nearest neighbors in pos_x, then
    interpolate features from x (defined on pos_x) onto pos_y.
    """
    y_idx, x_idx = manual_knn(pos_x, pos_y, k, batch_x, batch_y)

    # Correct indexing: x_idx indexes pos_x/x, y_idx indexes pos_y/out
    diff = pos_x[x_idx] - pos_y[y_idx]
    dist = (diff * diff).sum(dim=-1).sqrt()
    weights = 1.0 / (dist + 1e-8)

    out = torch.zeros(pos_y.size(0), x.size(1), device=x.device)
    count = torch.zeros(pos_y.size(0), 1, device=x.device)

    weighted_x = x[x_idx] * weights.unsqueeze(1)
    out.index_add_(0, y_idx, weighted_x)
    count.index_add_(0, y_idx, weights.unsqueeze(1))

    return out / (count + 1e-8)

class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = safe_fps(pos, batch, ratio=self.ratio)

        row, col = safe_radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)

        edge_index = torch.stack([col, row], dim=0)

        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

class FPModule(nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def forward(self, x, pos, batch, x_skip, pos_skip, batch_skip):
        x = safe_knn_interpolate(x, pos, pos_skip, batch, batch_skip, k=self.k)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        x = self.nn(x)
        return x

class PointNet2Backbone(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, out_channels=128):
        super().__init__()
        self.sa1_mlp = MLP([3 + in_channels, hidden_dim, hidden_dim, hidden_dim*2])
        self.sa1 = SAModule(0.5, 10.0, self.sa1_mlp)
        self.sa2_mlp = MLP([3 + hidden_dim*2, hidden_dim*2, hidden_dim*2, hidden_dim*4])
        self.sa2 = SAModule(0.25, 20.0, self.sa2_mlp)
        self.sa3_mlp = MLP([3 + hidden_dim*4, hidden_dim*4, hidden_dim*8, hidden_dim*8])
        self.sa3 = SAModule(0.1, 40.0, self.sa3_mlp)

        self.fp3_mlp = MLP([hidden_dim*8 + hidden_dim*4, hidden_dim*4, hidden_dim*4])
        self.fp3 = FPModule(3, self.fp3_mlp)
        self.fp2_mlp = MLP([hidden_dim*4 + hidden_dim*2, hidden_dim*4, hidden_dim*2])
        self.fp2 = FPModule(3, self.fp2_mlp)
        self.fp1_mlp = MLP([hidden_dim*2 + in_channels, hidden_dim*2, out_channels])
        self.fp1 = FPModule(3, self.fp1_mlp)

    def forward(self, x, pos, batch):
        sa0_x, sa0_pos, sa0_batch = x, pos, batch

        sa1_x, sa1_pos, sa1_batch = self.sa1(sa0_x, sa0_pos, sa0_batch)
        sa2_x, sa2_pos, sa2_batch = self.sa2(sa1_x, sa1_pos, sa1_batch)
        sa3_x, sa3_pos, sa3_batch = self.sa3(sa2_x, sa2_pos, sa2_batch)

        global_feat = global_max_pool(sa3_x, sa3_batch)

        fp3_x = self.fp3(sa3_x, sa3_pos, sa3_batch, sa2_x, sa2_pos, sa2_batch)
        fp2_x = self.fp2(fp3_x, sa2_pos, sa2_batch, sa1_x, sa1_pos, sa1_batch)
        fp1_x = self.fp1(fp2_x, sa1_pos, sa1_batch, sa0_x, sa0_pos, sa0_batch)

        return fp1_x, global_feat
