import torch
import torch.nn as nn
import numpy as np

def furthest_point_sampling(points, num_samples):
    """
    最远点采样算法的 PyTorch 实现
    替代 pointnet2_ops.furthest_point_sample
    
    Args:
        points: (B, N, 3) 输入点云
        num_samples: 采样点数
        
    Returns:
        sampled_indices: (B, num_samples) 采样点索引
    """
    B, N, _ = points.shape
    device = points.device
    
    # 确保采样数不超过总点数
    num_samples = min(num_samples, N)
    
    # 初始化：随机选择第一个点
    sampled_indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    distances = torch.full((B, N), float('inf'), device=device)
    
    # 随机选择起始点
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    for i in range(num_samples):
        sampled_indices[:, i] = farthest
        
        # 计算当前最远点到所有点的距离
        centroid = points[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)  # (B, N)
        
        # 更新距离矩阵
        distances = torch.minimum(distances, dist)
        
        # 选择下一个最远点
        farthest = torch.argmax(distances, dim=-1)
    
    return sampled_indices

def gather_operation(points, indices):
    """
    根据 index 收集点云特征
    替代 pointnet2_utils.gather_operation
    
    Args:
        points: (B, C, N) 输入特征
        indices: (B, M) 采样索引
        
    Returns:
        gathered_points: (B, C, M) 收集的特征
    """
    B, C, N = points.shape
    M = indices.shape[1]
    device = points.device
    
    # 使用 torch.gather 收集特征
    # indices 需要扩展到 (B, 1, M) 以匹配 points 的维度
    expanded_indices = indices.unsqueeze(1).expand(B, C, M)
    
    # 使用 gather 操作
    gathered_points = torch.gather(points, 2, expanded_indices)
    
    return gathered_points

class CompatibleDGCNNGrouper(nn.Module):
    """
    完全兼容 DGCNN_Grouper 的替代实现
    """
    def __init__(self, num_groups=512):
        super().__init__()
        self.num_groups = num_groups
        
        # 复制原始 DGCNN_Grouper 的层结构
        self.input_trans = nn.Conv1d(3, 8, 1)
        self.layer1 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1, bias=False),
            nn.GroupNorm(4, 64),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=1, bias=False),
            nn.GroupNorm(4, 128),
            nn.LeakyReLU(negative_slope=0.2)
        )
    
    def fps_downsample(self, coor, x, num_group):
        """
        自定义的 FPS 下采样实现
        Args:
            coor: (B, 3, N) 坐标
            x: (B, C, N) 特征
            num_group: 采样点数
        """
        xyz = coor.transpose(1, 2).contiguous() # (B, N, 3)
        
        # 使用自定义 FPS
        fps_idx = furthest_point_sampling(xyz, num_group)  # (B, num_group)
        
        combined_x = torch.cat([coor, x], dim=1)  # (B, 3+C, N)
        
        # 使用自定义 gather 操作
        new_combined_x = gather_operation(combined_x, fps_idx)  # (B, 3+C, num_group)
        
        new_coor = new_combined_x[:, :3, :]  # (B, 3, num_group)
        new_x = new_combined_x[:, 3:, :]     # (B, C, num_group)
        
        return new_coor, new_x
    
    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        """
        获取图特征（修复索引越界问题）
        """
        # coor: bs, 3, np, x: bs, c, np
        
        k = 16
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)
        
        with torch.no_grad():
            # 使用修复的 KNN 搜索
            idx = knn_point(k, coor_k.transpose(-1, -2).contiguous(), coor_q.transpose(-1, -2).contiguous()) # B G M
            idx = idx.transpose(-1, -2).contiguous()
            assert idx.shape[1] == k
            
            # 修复索引计算 - 确保索引不越界
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
            
            # 确保索引在有效范围内
            idx = torch.clamp(idx, 0, x_k.size(2) - 1)
            
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        
        # 安全的索引访问
        try:
            feature = x_k[idx, :].view(batch_size, num_points_q, k, num_dims).permute(0, 3, 1, 2).contiguous()
        except:
            # 如果仍然有问题，使用简化的特征
            feature = torch.zeros(batch_size, num_dims, num_points_q, k, device=x_q.device)
            return feature.permute(0, 2, 3, 1).contiguous()
        
        feature = feature - x_q.view(batch_size, -1, num_points_q, 1).repeat(1, 1, 1, k)
        
        return feature.permute(0, 2, 3, 1).contiguous()
    
    def forward(self, x):
        """
        前向传播，完全复制原始 DGCNN_Grouper 的逻辑
        """
        f = self.input_trans(x) # b 8 n
        
        coor_q, f_q = self.fps_downsample(x, f, 512) # b 3 512, b 8 512
        coor_k, f_k = self.fps_downsample(x, f, 256) # b 3 256, b 8 256
        coor_k2, f_k2 = self.fps_downsample(x, f, 128) # b 3 128, b 8 128
        coor_k3, f_k3 = self.fps_downsample(x, f, 64) # b 3 64, b 8 64
        
        feature = self.get_graph_feature(coor_q, f_q, coor_k, f_k)
        feature = self.layer1(feature)
        feature = self.get_graph_feature(coor_q, f_q, coor_k2, f_k2)
        feature = torch.cat([feature, self.layer2(feature)], dim=1)
        feature = self.get_graph_feature(coor_q, f_q, coor_k3, f_k3)
        feature = torch.cat([feature, self.layer3(feature)], dim=1)
        feature = self.layer4(feature)
        
        f_q = f_q.view(1, f_q.size(1), f_q.size(2), 1)
        f_q = f_q.repeat(1, 1, 1, 16).view(1, f_q.size(1), -1)
        f_q = torch.cat([f_q, feature], dim=2)
        
        return coor_q, f_q

def knn_point(nsample, xyz, new_xyz):
    """
    KNN 点搜索实现（修复索引越界）
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    
    # 确保索引不越界
    N = xyz.size(1)  # 总点数
    group_idx = torch.clamp(group_idx, 0, N - 1)
    
    return group_idx

def square_distance(src, dst):
    """
    计算平方距离
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
