import numpy as np
import open3d as o3d
import sys
import os

# 添加GeoTransformer路径
sys.path.append('/home/mlx/mujoco/YOLO_World-SAM-GraspNet/GeoTransformer')
sys.path.append('/home/mlx/mujoco/YOLO_World-SAM-GraspNet/GeoTransformer/experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn')

import torch
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from config import make_cfg
from model import create_model

def load_geotransformer_model():
    """加载GeoTransformer模型"""
    # 配置参数
    weights_path = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/GeoTransformer/weight/geotransformer-3dmatch.pth.tar'
    neighbor_limits = [38, 36, 36, 38]
    
    # 初始化模型
    cfg = make_cfg()
    model = create_model(cfg).cuda()
    state_dict = torch.load(weights_path)
    model.load_state_dict(state_dict["model"])
    model.eval()
    
    return model, cfg, neighbor_limits

def register_two_pointclouds(model, cfg, neighbor_limits, src_points, ref_points, max_points=50000):
    """使用GeoTransformer配准两个点云"""
    # 确保点云格式正确
    if len(src_points.shape) > 2:
        src_points = src_points.reshape(-1, 3)
    if len(ref_points.shape) > 2:
        ref_points = ref_points.reshape(-1, 3)
    
    # 移除无效点
    src_valid = ~np.isnan(src_points).any(axis=1) & ~np.isinf(src_points).any(axis=1)
    ref_valid = ~np.isnan(ref_points).any(axis=1) & ~np.isinf(ref_points).any(axis=1)
    src_points = src_points[src_valid]
    ref_points = ref_points[ref_valid]
    
    # 下采样到合理数量
    if len(src_points) > max_points:
        indices = np.random.choice(len(src_points), max_points, replace=False)
        src_points = src_points[indices]
    if len(ref_points) > max_points:
        indices = np.random.choice(len(ref_points), max_points, replace=False)
        ref_points = ref_points[indices]
    
    # 创建简单特征
    src_feats = np.ones_like(src_points[:, :1])
    ref_feats = np.ones_like(ref_points[:, :1])
    
    # 准备数据
    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
        "transform": np.eye(4, dtype=np.float32),  # 添加虚拟变换矩阵
    }
    
    # 数据预处理
    data_dict = registration_collate_fn_stack_mode(
        [data_dict], 
        cfg.backbone.num_stages, 
        cfg.backbone.init_voxel_size, 
        cfg.backbone.init_radius, 
        neighbor_limits
    )
    
    # 模型推理
    with torch.no_grad():
        data_dict = to_cuda(data_dict)
        output_dict = model(data_dict)
        output_dict = release_cuda(output_dict)
    
    # 获取变换矩阵
    transform = output_dict["estimated_transform"].cpu().numpy()
    return transform

def main():
    """主函数 - 简单的两点云配准示例"""
    print("正在加载GeoTransformer模型...")
    model, cfg, neighbor_limits = load_geotransformer_model()
    print("模型加载完成")
    
    # 配置你的点云文件路径
    src_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/apple1_final.ply'  # 源点云
    ref_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/apple.ply'        # 参考点云
    
    # 检查文件是否存在
    if not os.path.exists(src_file):
        print(f"源文件不存在: {src_file}")
        return
    if not os.path.exists(ref_file):
        print(f"参考文件不存在: {ref_file}")
        return
    
    # 加载点云
    print("正在加载点云...")
    src_pcd = o3d.io.read_point_cloud(src_file)
    ref_pcd = o3d.io.read_point_cloud(ref_file)
    
    print(f"源点云点数: {len(src_pcd.points)}")
    print(f"参考点云点数: {len(ref_pcd.points)}")
    
    # 转换为numpy数组
    src_points = np.asarray(src_pcd.points)
    ref_points = np.asarray(ref_pcd.points)
    
    # 执行配准
    print("正在进行点云配准...")
    transform = register_two_pointclouds(model, cfg, neighbor_limits, src_points, ref_points)
    
    print("配准完成，变换矩阵:")
    print(transform)
    
    # 应用变换
    src_pcd_transformed = src_pcd.transform(transform)
    
    # 可视化结果
    ref_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 参考点云
    src_pcd_transformed.paint_uniform_color([0, 1, 0])  # 绿色 - 配准后的源点云
    
    print("显示配准结果...")
    o3d.visualization.draw_geometries([ref_pcd, src_pcd_transformed])
    
    # 保存配准后的点云
    output_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/geotransformer_aligned.ply'
    o3d.io.write_point_cloud(output_file, src_pcd_transformed)
    print(f"配准结果已保存到: {output_file}")
    
    # 合并点云
    fused_pcd = ref_pcd + src_pcd_transformed
    fused_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/geotransformer_fused.ply'
    o3d.io.write_point_cloud(fused_file, fused_pcd)
    print(f"融合结果已保存到: {fused_file}")

if __name__ == "__main__":
    main()
