#!/usr/bin/env python3
"""
改进的全局点云融合方案
结合相机外参先验和配准优化
"""

import numpy as np
import open3d as o3d
import teaserpp_python
import time
import json

class ImprovedGlobalFusion:
    def __init__(self):
        """初始化融合器"""
        # 相机外参（右相机相对于左相机）
        self.rotation_matrix = np.array([
            [0.64348702, -0.24681324, 0.72457414],
            [0.17130231, 0.96901536, 0.17794592],
            [-0.74604288, 0.00961534, 0.66582848]
        ])
        
        self.translation_vector = np.array([-890.21218409, -121.98498635, 871.98401914])
        
        # 构建完整变换矩阵
        self.camera_extrinsics = np.eye(4)
        self.camera_extrinsics[:3, :3] = self.rotation_matrix
        self.camera_extrinsics[:3, 3] = self.translation_vector
        
        print("相机外参初始化完成")
        print("旋转矩阵:")
        print(self.rotation_matrix)
        print("平移向量 (mm):", self.translation_vector)
    
    def analyze_point_clouds(self, source_path, target_path):
        """分析点云特征"""
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # 计算点云尺寸
        source_size = np.max(source_points, axis=0) - np.min(source_points, axis=0)
        target_size = np.max(target_points, axis=0) - np.min(target_points, axis=0)
        
        # 计算点云中心
        source_center = np.mean(source_points, axis=0)
        target_center = np.mean(target_points, axis=0)
        
        print(f"\n=== 点云分析 ===")
        print(f"源点云尺寸: {source_size}")
        print(f"目标点云尺寸: {target_size}")
        print(f"源点云中心: {source_center}")
        print(f"目标点云中心: {target_center}")
        print(f"源点云点数: {len(source.points)}")
        print(f"目标点云点数: {len(target.points)}")
        
        return {
            'source_size': source_size,
            'target_size': target_size,
            'source_center': source_center,
            'target_center': target_center,
            'source_points': len(source.points),
            'target_points': len(target.points)
        }
    
    def auto_scale_detection(self, analysis):
        """自动检测缩放因子"""
        estimated_object_size = np.max([np.mean(analysis['source_size']), 
                                      np.mean(analysis['target_size'])])
        camera_distance = np.linalg.norm(self.translation_vector)
        
        print(f"\n=== 自动缩放检测 ===")
        print(f"估计物体尺寸: {estimated_object_size:.2f}")
        print(f"相机距离: {camera_distance:.2f} mm")
        
        if camera_distance > estimated_object_size * 10:
            scale_factor = estimated_object_size / 100
            print(f"检测到单位不匹配，建议缩放因子: {scale_factor:.6f}")
            return scale_factor
        else:
            print("单位匹配，使用缩放因子 1.0")
            return 1.0
    
    def method1_extrinsics_only(self, source_path, target_path, scale_factor=1.0):
        """方法1：仅使用相机外参"""
        print("\n=== 方法1：仅使用相机外参 ===")
        
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        # 给点云上色
        source.paint_uniform_color([1, 0.706, 0])  # 黄色
        target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
        
        # 应用缩放因子
        scaled_extrinsics = self.camera_extrinsics.copy()
        scaled_extrinsics[:3, 3] *= scale_factor
        
        print(f"使用缩放因子: {scale_factor}")
        print("变换矩阵:")
        print(scaled_extrinsics)
        
        # 应用变换
        target.transform(scaled_extrinsics)
        
        # 融合
        combined_pcd = source + target
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)
        
        print(f"融合后点数: {len(fused_pcd.points)}")
        
        return fused_pcd, scaled_extrinsics
    
    def method2_registration_only(self, source_path, target_path, voxel_size=0.5):
        """方法2：仅使用配准算法"""
        print("\n=== 方法2：仅使用配准算法 ===")
        
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        # 给点云上色
        source.paint_uniform_color([1, 0.706, 0])  # 黄色
        target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
        
        # 预处理
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 计算法向量
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # 计算FPFH特征
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        # 特征匹配
        distance_threshold = voxel_size * 1.5
        result_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        
        corrs = np.asarray(result_feature_matching.correspondence_set)
        print(f"特征匹配对应点数: {len(corrs)}")
        
        if len(corrs) < 10:
            print("警告：对应点太少，使用更宽松参数")
            distance_threshold = voxel_size * 3
            result_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                source_down, target_down, source_fpfh, target_fpfh, True,
                distance_threshold,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                3, [
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.5),
                    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
            
            corrs = np.asarray(result_feature_matching.correspondence_set)
            print(f"重新匹配后对应点数: {len(corrs)}")
        
        if len(corrs) >= 5:
            # TEASER++配准
            source_points = np.asarray(source_down.points)[corrs[:, 0]].T
            target_points = np.asarray(target_down.points)[corrs[:, 1]].T
            
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.noise_bound = voxel_size * 2
            solver_params.cbar2 = 1.0
            solver_params.estimate_scaling = False
            solver_params.rotation_gnc_factor = 1.4
            solver_params.rotation_max_iterations = 100
            
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(source_points, target_points)
            
            solution = solver.getSolution()
            
            teaser_trans = np.eye(4)
            teaser_trans[:3, :3] = solution.rotation
            teaser_trans[:3, 3] = solution.translation
            
            print("TEASER++变换矩阵:")
            print(teaser_trans)
        else:
            print("对应点太少，使用单位矩阵")
            teaser_trans = np.eye(4)
        
        # ICP精配准
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 2, teaser_trans,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        final_trans = icp_result.transformation
        print("ICP最终变换矩阵:")
        print(final_trans)
        
        # 应用变换并融合
        source.transform(final_trans)
        combined_pcd = source + target
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)
        
        print(f"融合后点数: {len(fused_pcd.points)}")
        
        return fused_pcd, final_trans
    
    def method3_extrinsics_plus_registration(self, source_path, target_path, scale_factor=1.0, voxel_size=0.5):
        """方法3：外参+配准（推荐）"""
        print("\n=== 方法3：外参+配准 ===")
        
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        # 给点云上色
        source.paint_uniform_color([1, 0.706, 0])  # 黄色
        target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
        
        # 应用缩放因子到外参
        scaled_extrinsics = self.camera_extrinsics.copy()
        scaled_extrinsics[:3, 3] *= scale_factor
        
        print(f"使用缩放因子: {scale_factor}")
        print("初始变换矩阵（基于外参）:")
        print(scaled_extrinsics)
        
        # 应用初始变换
        target.transform(scaled_extrinsics)
        
        # 可视化初始对齐
        print("显示初始对齐结果...")
        o3d.visualization.draw_geometries([source, target], 
                                       window_name="方法3：初始对齐")
        
        # 精细配准
        print("进行精细配准...")
        
        # 下采样
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
        
        # 计算法向量
        source_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        # 计算FPFH特征
        source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100))
        
        # 特征匹配
        distance_threshold = voxel_size * 1.5
        result_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        
        corrs = np.asarray(result_feature_matching.correspondence_set)
        print(f"特征匹配对应点数: {len(corrs)}")
        
        if len(corrs) >= 5:
            # TEASER++精细配准
            source_points = np.asarray(source_down.points)[corrs[:, 0]].T
            target_points = np.asarray(target_down.points)[corrs[:, 1]].T
            
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.noise_bound = voxel_size * 3  # 增大噪声容忍度
            solver_params.cbar2 = 1.0
            solver_params.estimate_scaling = False
            solver_params.rotation_gnc_factor = 1.1
            solver_params.rotation_max_iterations = 50
            
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(source_points, target_points)
            
            solution = solver.getSolution()
            
            refinement_transform = np.eye(4)
            refinement_transform[:3, :3] = solution.rotation
            refinement_transform[:3, 3] = solution.translation
            
            print("精细变换增量:")
            print(refinement_transform)
            
            # 应用精细变换
            target.transform(refinement_transform)
            current_transform = refinement_transform
        else:
            print("对应点太少，跳过精细配准")
            current_transform = np.eye(4)
        
        # ICP最终优化
        source.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        target.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
        
        icp_result = o3d.pipelines.registration.registration_icp(
            source, target, voxel_size * 2, current_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        final_transform = icp_result.transformation
        print("ICP最终变换增量:")
        print(final_transform)
        
        # 计算总体变换
        overall_transform = final_transform @ scaled_extrinsics
        print("总体变换矩阵:")
        print(overall_transform)
        
        # 应用最终变换并融合
        target.transform(final_transform)
        combined_pcd = source + target
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)
        
        print(f"融合后点数: {len(fused_pcd.points)}")
        
        return fused_pcd, overall_transform
    
    def compare_all_methods(self, source_path, target_path):
        """比较所有融合方法"""
        print("=== 全局点云融合方法对比 ===")
        
        # 分析点云
        analysis = self.analyze_point_clouds(source_path, target_path)
        
        # 自动检测缩放因子
        scale_factor = self.auto_scale_detection(analysis)
        
        results = {}
        
        # 方法1：仅使用外参
        results['extrinsics_only'], trans1 = self.method1_extrinsics_only(
            source_path, target_path, scale_factor)
        o3d.visualization.draw_geometries([results['extrinsics_only']], 
                                       window_name="方法1：仅外参")
        
        # 方法2：仅使用配准
        results['registration_only'], trans2 = self.method2_registration_only(
            source_path, target_path)
        o3d.visualization.draw_geometries([results['registration_only']], 
                                       window_name="方法2：仅配准")
        
        # 方法3：外参+配准
        results['extrinsics_plus_registration'], trans3 = self.method3_extrinsics_plus_registration(
            source_path, target_path, scale_factor)
        o3d.visualization.draw_geometries([results['extrinsics_plus_registration']], 
                                       window_name="方法3：外参+配准")
        
        # 保存所有结果
        for method, fused_pcd in results.items():
            output_path = f"outputs/global_fused_{method}.ply"
            o3d.io.write_point_cloud(output_path, fused_pcd)
            print(f"{method} 结果已保存至: {output_path}")
        
        return results, [trans1, trans2, trans3]

if __name__ == "__main__":
    print("=== 改进的全局点云融合系统 ===")
    
    fusion = ImprovedGlobalFusion()
    
    # 使用背景点云进行融合
    source_path = "outputs/left1_background.ply"
    target_path = "outputs/right1_background.ply"
    
    # 比较所有方法
    results, transforms = fusion.compare_all_methods(source_path, target_path)
    
    print("\n=== 完成 ===")
    print("已生成三种融合方法的结果，请查看并选择最佳效果")
    print("推荐方法：extrinsics_plus_registration（外参+配准）")
