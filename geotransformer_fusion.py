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
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
from config import make_cfg
from model import create_model

class GeoTransformerFusion:
    def __init__(self):
        # 配置参数
        self.weights_path = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/GeoTransformer/weight/geotransformer-3dmatch.pth.tar'
        self.voxel_size = 0.025  # 3DMatch标准体素大小2.5cm
        self.neighbor_limits = [38, 36, 36, 38]
        
        # 初始化模型
        self.cfg = make_cfg()
        self.model = create_model(self.cfg).cuda()
        state_dict = torch.load(self.weights_path)
        self.model.load_state_dict(state_dict["model"])
        self.model.eval()
        
    def preprocess_pointcloud(self, points, max_points=50000):
        """预处理点云数据"""
        # 确保点云是Nx3格式
        if len(points.shape) > 2:
            points = points.reshape(-1, 3)
        
        # 移除无效点
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        points = points[valid_mask]
        
        # 下采样到合理数量
        if len(points) > max_points:
            indices = np.random.choice(len(points), max_points, replace=False)
            points = points[indices]
        
        # 归一化到3DMatch尺度
        if self.voxel_size > 0:
            points = points / self.voxel_size
        
        return points.astype(np.float32)
    
    def register_pointclouds(self, src_points, ref_points):
        """使用GeoTransformer进行点云配准"""
        # 预处理点云
        src_points = self.preprocess_pointcloud(src_points)
        ref_points = self.preprocess_pointcloud(ref_points)
        
        # 创建特征（简单特征）
        src_feats = np.ones_like(src_points[:, :1])
        ref_feats = np.ones_like(ref_points[:, :1])
        
        # 准备数据字典
        data_dict = {
            "ref_points": ref_points,
            "src_points": src_points,
            "ref_feats": ref_feats,
            "src_feats": src_feats,
            "transform": np.eye(4, dtype=np.float32),  # 添加虚拟变换矩阵
        }
        
        # 数据预处理
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], 
            self.cfg.backbone.num_stages, 
            self.cfg.backbone.init_voxel_size, 
            self.cfg.backbone.init_radius, 
            self.neighbor_limits
        )
        
        # 模型推理
        with torch.no_grad():
            data_dict = to_cuda(data_dict)
            output_dict = self.model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)
        
        # 获取变换矩阵
        estimated_transform = output_dict["estimated_transform"].cpu().numpy()
        
        # 恢复原始尺度
        if self.voxel_size > 0:
            scale_matrix = np.eye(4)
            scale_matrix[:3, :3] *= self.voxel_size
            estimated_transform = scale_matrix @ estimated_transform @ np.linalg.inv(scale_matrix)
        
        return estimated_transform
    
    def fuse_pointclouds(self, pcd_list):
        """融合多个点云"""
        if len(pcd_list) < 2:
            return pcd_list[0] if pcd_list else None
        
        # 以第一个点云为参考
        ref_pcd = pcd_list[0]
        fused_pcd = ref_pcd
        
        print(f"开始融合 {len(pcd_list)} 个点云...")
        
        for i in range(1, len(pcd_list)):
            print(f"正在融合第 {i+1} 个点云...")
            
            # 转换为numpy数组
            src_points = np.asarray(pcd_list[i].points)
            ref_points = np.asarray(ref_pcd.points)
            
            # 进行配准
            transform = self.register_pointclouds(src_points, ref_points)
            
            # 应用变换
            src_pcd_transformed = pcd_list[i].transform(transform)
            
            # 合并点云
            fused_pcd += src_pcd_transformed
            
            print(f"第 {i+1} 个点云融合完成")
        
        return fused_pcd

def main():
    """主函数示例"""
    # 初始化融合器
    fusion = GeoTransformerFusion()
    
    # 示例：加载你的点云文件
    pcd_files = [
        '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/right1_background_no_background.ply',
        '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/left1_background_no_background.ply',
        # 添加更多点云文件路径
    ]
    
    # 加载点云
    pcd_list = []
    for file_path in pcd_files:
        if os.path.exists(file_path):
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) > 0:
                pcd_list.append(pcd)
                print(f"加载点云: {file_path}, 点数: {len(pcd.points)}")
    
    if len(pcd_list) < 2:
        print("需要至少2个点云文件进行融合")
        return
    
    # 执行融合
    fused_pcd = fusion.fuse_pointclouds(pcd_list)
    
    # 保存结果
    output_path = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/geotransformer_fused.ply'
    o3d.io.write_point_cloud(output_path, fused_pcd)
    print(f"融合结果已保存到: {output_path}")
    print(f"融合后点云点数: {len(fused_pcd.points)}")
    
    # 可视化
    fused_pcd.paint_uniform_color([0.5, 0.5, 0.5])
    o3d.visualization.draw_geometries([fused_pcd])

if __name__ == "__main__":
    main()
