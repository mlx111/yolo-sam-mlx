import numpy as np
import open3d as o3d
import sys
import os

# 添加GeoTransformer ModelNet配置路径
sys.path.append('GeoTransformer')
sys.path.append('GeoTransformer/experiments/geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn')

import torch
from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from config import make_cfg
from model import create_model

class GeoTransformerObjectFusion:
    def __init__(self):
        # 配置参数 - 使用ModelNet预训练模型
        self.weights_path = 'GeoTransformer/weight/geotransformer-modelnet.pth.tar'
        self.neighbor_limits = [38, 36, 36]  # ModelNet配置：3个阶段
        self.max_points = 717  # ModelNet标准点数
        
        # 初始化ModelNet配置的模型
        self.cfg = make_cfg()
        self.model = create_model(self.cfg).cuda()
        state_dict = torch.load(self.weights_path)
        self.model.load_state_dict(state_dict["model"])
        self.model.eval()
        
    def preprocess_object_pointcloud(self, points):
        """预处理物体级点云数据"""
        # 确保点云是Nx3格式
        if len(points.shape) > 2:
            points = points.reshape(-1, 3)
        
        # 移除无效点
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        points = points[valid_mask]
        
        # 归一化物体到单位球内（ModelNet标准）
        if len(points) > 0:
            # 计算中心点
            centroid = np.mean(points, axis=0)
            points = points - centroid
            
            # 计算最大距离并缩放到[-0.5, 0.5]
            max_dist = np.max(np.linalg.norm(points, axis=1))
            if max_dist > 0:
                points = points / max_dist * 0.5
        
        # 下采样到ModelNet标准点数
        if len(points) > self.max_points:
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
        elif len(points) < self.max_points:
            # 如果点数不够，进行重复采样
            indices = np.random.choice(len(points), self.max_points, replace=True)
            points = points[indices]
        
        return points.astype(np.float32)
    
    def register_objects(self, src_points, ref_points):
        """使用GeoTransformer进行物体级点云配准"""
        # 预处理点云
        src_points = self.preprocess_object_pointcloud(src_points)
        ref_points = self.preprocess_object_pointcloud(ref_points)
        
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
        estimated_transform = output_dict["estimated_transform"]
        if isinstance(estimated_transform, torch.Tensor):
            estimated_transform = estimated_transform.cpu().numpy()
        
        return estimated_transform
    
    def fuse_objects(self, pcd_list):
        """融合多个物体点云"""
        if len(pcd_list) < 2:
            return pcd_list[0] if pcd_list else None
        
        # 以第一个点云为参考
        ref_pcd = pcd_list[0]
        fused_pcd = ref_pcd
        
        print(f"开始融合 {len(pcd_list)} 个物体点云...")
        
        for i in range(1, len(pcd_list)):
            print(f"正在融合第 {i+1} 个物体点云...")
            
            # 转换为numpy数组
            src_points = np.asarray(pcd_list[i].points)
            ref_points = np.asarray(ref_pcd.points)
            
            # 进行配准
            transform = self.register_objects(src_points, ref_points)
            
            # 应用变换
            src_pcd_transformed = pcd_list[i].transform(transform)
            
            # 合并点云
            fused_pcd += src_pcd_transformed
            
            print(f"第 {i+1} 个物体点云融合完成")
        
        return fused_pcd

def main():
    """主函数示例"""
    # 初始化物体级融合器
    fusion = GeoTransformerObjectFusion()
    obj='apple'
    # 配置你的物体点云文件路径
    pcd_files = [
        "mirror/inputs/left_apple_centered.ply", 
        "mirror/inputs/right_apple_centered.ply"
        # 添加更多物体点云文件路径
    ]
    
    # 加载点云
    pcd_list = []
    for file_path in pcd_files:
        if os.path.exists(file_path):
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) > 0:
                pcd_list.append(pcd)
                print(f"加载物体点云: {file_path}, 点数: {len(pcd.points)}")
    
    if len(pcd_list) < 2:
        print("需要至少2个物体点云文件进行融合")
        return
    
    # 执行融合
    fused_pcd = fusion.fuse_objects(pcd_list)
    
    # 保存结果
    output_path = f'outputs/geotransformer_{obj}_fused.ply'
    o3d.io.write_point_cloud(output_path, fused_pcd)
    print(f"物体融合结果已保存到: {output_path}")
    print(f"融合后点云点数: {len(fused_pcd.points)}")
    
    # 可视化
    fused_pcd.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([fused_pcd])

def simple_object_registration():
    """简单的两物体配准示例"""
    print("正在加载GeoTransformer物体级模型...")
    
    # 初始化融合器
    fusion = GeoTransformerObjectFusion()
    print("物体级模型加载完成")
    
    # 配置物体点云文件路径
    src_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/apple1_final.ply'  # 源物体
    ref_file = '/home/mlx/mujoco/YOLO_World-SAM-GraspNet/outputs/apple.ply'        # 参考物体
    
    # 检查文件是否存在
    if not os.path.exists(src_file):
        print(f"源文件不存在: {src_file}")
        return
    if not os.path.exists(ref_file):
        print(f"参考文件不存在: {ref_file}")
        return
    
    # 加载点云
    print("正在加载物体点云...")
    src_pcd = o3d.io.read_point_cloud(src_file)
    ref_pcd = o3d.io.read_point_cloud(ref_file)
    
    print(f"源物体点云点数: {len(src_pcd.points)}")
    print(f"参考物体点云点数: {len(ref_pcd.points)}")
    
    # 转换为numpy数组
    src_points = np.asarray(src_pcd.points)
    ref_points = np.asarray(ref_pcd.points)
    
    # 执行配准
    print("正在进行物体点云配准...")
    transform = fusion.register_objects(src_points, ref_points)
    
    print("物体配准完成，变换矩阵:")
    print(transform)
    
    # 应用变换
    src_pcd_transformed = src_pcd.transform(transform)
    
    # 可视化结果
    ref_pcd.paint_uniform_color([1, 0, 0])  # 红色 - 参考物体
    src_pcd_transformed.paint_uniform_color([0, 1, 0])  # 绿色 - 配准后的源物体
    
    print("显示物体配准结果...")
    o3d.visualization.draw_geometries([ref_pcd, src_pcd_transformed])
    
    # 保存配准后的点云
    output_file = 'outputs/geotransformer_object_aligned.ply'
    o3d.io.write_point_cloud(output_file, src_pcd_transformed)
    print(f"物体配准结果已保存到: {output_file}")
    
    # 合并物体点云
    fused_pcd = ref_pcd + src_pcd_transformed
    fused_file = 'outputs/geotransformer_object_fused.ply'
    o3d.io.write_point_cloud(fused_file, fused_pcd)
    print(f"物体融合结果已保存到: {fused_file}")

if __name__ == "__main__":
    # 选择运行模式
    print("选择运行模式:")
    print("1. 简单两物体配准")
    print("2. 多物体融合")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        simple_object_registration()
    elif choice == "2":
        main()
    else:
        print("无效选择，运行简单两物体配准...")
        simple_object_registration()
