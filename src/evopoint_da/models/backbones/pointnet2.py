import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool, MLP


try:
    from torch_cluster import fps as cluster_fps
    from torch_cluster import radius as cluster_radius
    from torch_cluster import knn as cluster_knn
    HAS_CLUSTER = True
except (ImportError, OSError) as e:
    HAS_CLUSTER = False
    print(f"[PointNet2] Warning: 'torch-cluster' load failed ({e}). Using pure PyTorch fallback.")



def manual_fps(x, batch, ratio=0.5):
    """最远点采样 (Farthest Point Sampling) 的纯 PyTorch 实现"""
    new_indices = []

    batch_size = batch.max().item() + 1
    for b_idx in range(batch_size):
        mask = batch == b_idx
        indices = torch.where(mask)[0]
        pos_b = x[indices]

        num_points = indices.shape[0]
        num_samples = int(num_points * ratio)
        if num_samples == 0:
            num_samples = 1

        dists = torch.ones(num_points, device=x.device) * 1e10
        farthest = torch.randint(0, num_points, (), dtype=torch.long, device=x.device)

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
    """半径搜索 - 返回 (row=Target, col=Source)"""
    # x: Source points (Neighbors)
    # y: Target points (Centers)
    row_list, col_list = [], []
    batch_size = batch_x.max().item() + 1

    for b_idx in range(batch_size):
        idx_x = torch.where(batch_x == b_idx)[0]
        idx_y = torch.where(batch_y == b_idx)[0]

        pos_x = x[idx_x]  # Source
        pos_y = y[idx_y]  # Target

        dist = torch.cdist(pos_x, pos_y)
        mask = dist < r

        for i in range(pos_x.shape[0]):
            neighbors = torch.where(mask[i])[0]
            if len(neighbors) > max_num_neighbors:
                neighbors = neighbors[:max_num_neighbors]

            row_list.append(idx_y[neighbors])
            col_list.append(torch.full_like(neighbors, idx_x[i]))

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
        # dist: [num_y, num_x] -> for each y find nearest in x
        dist = torch.cdist(pos_y, pos_x)
        _, nn_idx = dist.topk(k_val, dim=1, largest=False)

        row = idx_y.repeat_interleave(k_val)          # y indices (targets)
        col = idx_x[nn_idx.reshape(-1)]               # x indices (sources)

        row_list.append(row)
        col_list.append(col)

    if len(row_list) == 0:
        dev = x.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return row, col

# Wrapper functions
def safe_fps(x, batch, ratio):
    if HAS_CLUSTER:
        return cluster_fps(x, batch, ratio=ratio)
    return manual_fps(x, batch, ratio)

def safe_radius(x, y, r, batch_x, batch_y, max_num_neighbors):
    if HAS_CLUSTER:
        return cluster_radius(x, y, r, batch_x, batch_y, max_num_neighbors)
    return manual_radius(x, y, r, batch_x, batch_y, max_num_neighbors)

def safe_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
    """Simulate torch_geometric.nn.unpool.knn_interpolate.

    For each point in pos_y, find its k nearest neighbors in pos_x, then
    interpolate features from x (defined on pos_x) onto pos_y.
    """
    if HAS_CLUSTER:
        # torch_cluster.knn: for each y find k nearest in x -> returns (y_idx, x_idx)
        y_idx, x_idx = cluster_knn(pos_x, pos_y, k, batch_x, batch_y)
    else:
        # Fallback: keep the same semantics as torch_cluster.knn
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

# ==========================================
# Modules
# ==========================================

class SAModule(nn.Module):
    def __init__(self, ratio, r, nn):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, batch):
        idx = safe_fps(pos, batch, ratio=self.ratio)

        # radius 返回: row (Target), col (Source)
        row, col = safe_radius(pos, pos[idx], self.r, batch, batch[idx], max_num_neighbors=64)

        # FIX: PointNetConv 需要 edge_index 为 [source, target]
        # col 是 Source indices (0..N), row 是 Target indices (0..M)
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
        # SA1
        self.sa1_mlp = MLP([3 + in_channels, hidden_dim, hidden_dim, hidden_dim*2])
        self.sa1 = SAModule(0.5, 10.0, self.sa1_mlp)
        # SA2
        self.sa2_mlp = MLP([3 + hidden_dim*2, hidden_dim*2, hidden_dim*2, hidden_dim*4])
        self.sa2 = SAModule(0.25, 20.0, self.sa2_mlp)
        # SA3
        self.sa3_mlp = MLP([3 + hidden_dim*4, hidden_dim*4, hidden_dim*8, hidden_dim*8])
        self.sa3 = SAModule(0.1, 40.0, self.sa3_mlp)

        # FP3
        self.fp3_mlp = MLP([hidden_dim*8 + hidden_dim*4, hidden_dim*4, hidden_dim*4])
        self.fp3 = FPModule(3, self.fp3_mlp)
        # FP2
        self.fp2_mlp = MLP([hidden_dim*4 + hidden_dim*2, hidden_dim*4, hidden_dim*2])
        self.fp2 = FPModule(3, self.fp2_mlp)
        # FP1
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
