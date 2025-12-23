import torch
import torch.nn as nn
from torch_geometric.nn import PointNetConv, global_max_pool, MLP

# --- 尝试导入加速库，如果失败则使用纯 PyTorch 实现 ---
try:
    from torch_cluster import fps as cluster_fps
    from torch_cluster import radius as cluster_radius
    from torch_cluster import knn as cluster_knn
    HAS_CLUSTER = True
except (ImportError, OSError) as e:
    HAS_CLUSTER = False
    print(f"[PointNet2] Warning: 'torch-cluster' load failed ({e}). Using pure PyTorch fallback.")

# ==========================================
# Pure PyTorch Implementations (Fallbacks)
# ==========================================

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
        if num_samples == 0: num_samples = 1
        
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
        
        pos_x = x[idx_x] # Source
        pos_y = y[idx_y] # Target
        
        dist = torch.cdist(pos_x, pos_y)
        mask = dist < r
        
        # 遍历 Source 点，找附近的 Target 点 (逻辑等价于找 Source 的 Neighbors)
        # 这里 manual_radius 的实现稍微反直觉：它是找 "哪些 Target 在 Source 的半径内"
        # 结果：
        # row: indices in y (Target)
        # col: indices in x (Source)
        for i in range(pos_x.shape[0]):
            neighbors = torch.where(mask[i])[0]
            if len(neighbors) > max_num_neighbors:
                neighbors = neighbors[:max_num_neighbors]
            
            row_list.append(idx_y[neighbors])
            col_list.append(torch.full_like(neighbors, idx_x[i]))

    if len(row_list) == 0:
        return torch.empty(0, dtype=torch.long, device=x.device), torch.empty(0, dtype=torch.long, device=x.device)

    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return row, col

def manual_knn(x, y, k, batch_x, batch_y):
    """KNN"""
    row_list, col_list = [], []
    batch_size = batch_x.max().item() + 1
    
    for b_idx in range(batch_size):
        idx_x = torch.where(batch_x == b_idx)[0]
        idx_y = torch.where(batch_y == b_idx)[0]
        pos_x = x[idx_x]
        pos_y = y[idx_y]
        
        dist = torch.cdist(pos_x, pos_y)
        k_val = min(k, pos_y.shape[0])
        _, indices = dist.topk(k_val, dim=1, largest=False)
        
        for i in range(pos_x.shape[0]):
            neighbors = indices[i]
            row_list.append(idx_y[neighbors])
            col_list.append(torch.full_like(neighbors, idx_x[i]))
            
    row = torch.cat(row_list)
    col = torch.cat(col_list)
    return row, col

# Wrapper functions
def safe_fps(x, batch, ratio):
    if HAS_CLUSTER: return cluster_fps(x, batch, ratio=ratio)
    return manual_fps(x, batch, ratio)

def safe_radius(x, y, r, batch_x, batch_y, max_num_neighbors):
    if HAS_CLUSTER: return cluster_radius(x, y, r, batch_x, batch_y, max_num_neighbors)
    return manual_radius(x, y, r, batch_x, batch_y, max_num_neighbors)

def safe_knn_interpolate(x, pos_x, pos_y, batch_x, batch_y, k):
    """Simulate knn_interpolate"""
    if HAS_CLUSTER:
        assign_index = cluster_knn(pos_x, pos_y, k, batch_x, batch_y)
    else:
        # manual_knn returns (row=Source, col=Target)
        # We need (Source, Target)
        assign_index_row, assign_index_col = manual_knn(pos_y, pos_x, k, batch_y, batch_x)
        assign_index = torch.stack([assign_index_row, assign_index_col], dim=0)
        
    y_idx, x_idx = assign_index[0], assign_index[1]
    
    dist = (pos_x[y_idx] - pos_y[x_idx]).pow(2).sum(dim=-1).sqrt()
    weights = 1.0 / (dist + 1e-8)
    
    out = torch.zeros(pos_y.shape[0], x.shape[1], device=x.device)
    count = torch.zeros(pos_y.shape[0], 1, device=x.device)
    
    weighted_x = x[y_idx] * weights.unsqueeze(1)
    out.index_add_(0, x_idx, weighted_x)
    count.index_add_(0, x_idx, weights.unsqueeze(1))
    
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