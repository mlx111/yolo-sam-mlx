#!/usr/bin/env python3
"""
RGB-D点云融合完整指南和优化版本
基于TEASER++ + ICP的双视角点云融合
"""

import numpy as np
import open3d as o3d
import teaserpp_python
import time
import argparse
import os

class PointCloudFusion:
    def __init__(self, voxel_size=0.005):
        """
        初始化点云融合器
        
        Args:
            voxel_size: 下采样体素大小(m)，建议0.002-0.01
        """
        self.voxel_size = voxel_size
        
    def preprocess_point_cloud(self, pcd):
        """
        预处理点云：下采样、法向量估计、FPFH特征
        """
        print(f"原始点云点数: {len(pcd.points)}")
        
        # 1. 下采样
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        print(f"下采样后点数: {len(pcd_down.points)}")
        
        # 2. 估计法向量
        radius_normal = self.voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        # 3. 计算FPFH特征
        radius_feature = self.voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        
        return pcd_down, pcd_fpfh
    
    def teaser_registration(self, source_down, target_down, source_fpfh, target_fpfh):
        """
        使用TEASER++进行粗配准
        """
        print("正在进行TEASER++粗配准...")
        
        # 基于特征匹配建立初始对应关系
        distance_threshold = self.voxel_size * 1.5
        result_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        
        # 提取对应点
        corrs = np.asarray(result_feature_matching.correspondence_set)
        print(f"特征匹配找到 {len(corrs)} 对对应点")
        
        if len(corrs) < 10:
            print("警告：对应点太少，可能影响配准质量")
        
        source_points = np.asarray(source_down.points)[corrs[:, 0]].T
        target_points = np.asarray(target_down.points)[corrs[:, 1]].T
        
        # 配置TEASER++参数
        solver_params = teaserpp_python.RobustRegistrationSolver.Params()
        solver_params.noise_bound = self.voxel_size
        solver_params.cbar2 = 1.0
        solver_params.estimate_scaling = False
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 100
        solver_params.rotation_cost_threshold = 1e-12
        
        solver = teaserpp_python.RobustRegistrationSolver(solver_params)
        
        start_time = time.time()
        solver.solve(source_points, target_points)
        end_time = time.time()
        
        print(f"TEASER++求解耗时: {end_time - start_time:.4f}s")
        
        solution = solver.getSolution()
        
        # 构建变换矩阵
        transformation = np.eye(4)
        transformation[:3, :3] = solution.rotation
        transformation[:3, 3] = solution.translation
        
        return transformation
    
    def refine_with_icp(self, source, target, initial_transform):
        """
        使用ICP进行精配准
        """
        print("正在进行ICP精配准...")
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, self.voxel_size * 2, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        print(f"ICP收敛时间: {icp_result.inlier_rmse:.6f}")
        print(f"ICP对应点数: {icp_result.correspondence_set_size}")
        
        return icp_result.transformation
    
    def fuse_point_clouds(self, source_path, target_path, output_path=None, visualize=True):
        """
        完整的点云融合流程
        
        Args:
            source_path: 源点云路径
            target_path: 目标点云路径  
            output_path: 输出路径，默认为None不保存
            visualize: 是否可视化结果
        """
        # 1. 加载点云
        source = self.load_point_cloud(source_path)
        target = self.load_point_cloud(target_path)
        
        # 给点云上色便于区分
        source.paint_uniform_color([1, 0.706, 0])   # 黄色
        target.paint_uniform_color([0, 0.651, 0.929]) # 蓝色
        
        # 2. 预处理
        print("预处理源点云...")
        source_down, source_fpfh = self.preprocess_point_cloud(source)
        print("预处理目标点云...")
        target_down, target_fpfh = self.preprocess_point_cloud(target)
        
        # 3. 粗配准
        teaser_trans = self.teaser_registration(source_down, target_down, source_fpfh, target_fpfh)
        print("TEASER++变换矩阵:\n", teaser_trans)
        
        # 4. 精配准
        final_trans = self.refine_with_icp(source, target, teaser_trans)
        print("最终变换矩阵:\n", final_trans)
        
        # 5. 融合
        source.transform(final_trans)
        combined_pcd = source + target
        
        # 体素融合去重
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=self.voxel_size * 0.4)
        
        print(f"融合后点云点数: {len(fused_pcd.points)}")
        
        # 6. 可视化
        if visualize:
            print("显示融合结果...")
            o3d.visualization.draw_geometries([fused_pcd], window_name="点云融合结果")
        
        # 7. 保存
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            o3d.io.write_point_cloud(output_path, fused_pcd)
            print(f"融合结果已保存至: {output_path}")
        
        return fused_pcd, final_trans
    
    def load_point_cloud(self, file_path):
        """
        加载点云文件，支持.ply和.npy格式
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.endswith('.npy'):
            points = np.load(file_path)
            if points.ndim != 2 or points.shape[1] != 3:
                raise ValueError(f"numpy数组格式错误，期望(N,3)，实际{points.shape}")
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                raise ValueError(f"点云文件为空: {file_path}")
        
        return pcd

def main():
    parser = argparse.ArgumentParser(description="RGB-D点云融合工具")
    parser.add_argument("--source", required=True, help="源点云路径")
    parser.add_argument("--target", required=True, help="目标点云路径")
    parser.add_argument("--output", default="outputs/fused_result.ply", help="输出路径")
    parser.add_argument("--voxel_size", type=float, default=0.005, help="体素大小(m)")
    parser.add_argument("--no_visualize", action="store_true", help="不显示可视化结果")
    
    args = parser.parse_args()
    
    # 创建融合器
    fusion = PointCloudFusion(voxel_size=args.voxel_size)
    
    # 执行融合
    try:
        fused_pcd, transform = fusion.fuse_point_clouds(
            args.source, args.target, args.output, not args.no_visualize
        )
        print("融合完成！")
        
    except Exception as e:
        print(f"融合失败: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    # 使用示例
    print("=== 点云融合工具使用指南 ===")
    print("\n1. 命令行使用:")
    print("   python pointcloud_fusion_guide.py --source left.ply --target right.ply")
    print("   python pointcloud_fusion_guide.py --source left.npy --target right.npy --output result.ply")
    
    print("\n2. 参数说明:")
    print("   --voxel_size: 体素大小，建议0.002-0.01，越小精度越高但速度越慢")
    print("   --no_visualize: 添加此参数不显示可视化窗口")
    
    print("\n3. 代码中使用:")
    print("   fusion = PointCloudFusion(voxel_size=0.005)")
    print("   fused_pcd, transform = fusion.fuse_point_clouds('left.ply', 'right.ply')")
    
    print("\n=== 运行示例 ===")
    
    # 如果直接运行，使用默认的示例文件
    if len(os.sys.argv) == 1:
        print("使用默认示例文件进行融合...")
        fusion = PointCloudFusion(voxel_size=0.005)
        try:
            fused_pcd, transform = fusion.fuse_point_clouds(
                "outputs/left1_background.ply", 
                "outputs/right1_background.ply",
                "outputs/fused_background.ply"
            )
        except Exception as e:
            print(f"示例运行失败: {e}")
            print("请检查点云文件是否存在，或使用命令行指定自己的文件")
    else:
        exit(main())
