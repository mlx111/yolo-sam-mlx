#!/usr/bin/env python3
"""
点云融合诊断工具
用于分析为什么融合后的苹果点云只有一个面
"""

import numpy as np
import open3d as o3d
import teaserpp_python
import time

def analyze_point_cloud(pcd, name):
    """分析单个点云的特征"""
    print(f"\n=== {name} 分析 ===")
    print(f"点数: {len(pcd.points)}")
    
    if len(pcd.points) == 0:
        print("错误：点云为空")
        return
    
    points = np.asarray(pcd.points)
    
    # 计算点云的边界框
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    
    print(f"中心点: [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]")
    print(f"尺寸: [{size[0]:.2f}, {size[1]:.2f}, {size[2]:.2f}]")
    
    # 分析点云的分布
    # 计算每个坐标轴的方差
    variance = np.var(points, axis=0)
    print(f"各轴方差: X={variance[0]:.2f}, Y={variance[1]:.2f}, Z={variance[2]:.2f}")
    
    # 判断点云的主要分布方向
    main_axis = np.argmax(variance)
    axis_names = ['X', 'Y', 'Z']
    print(f"主要分布方向: {axis_names[main_axis]}")
    
    # 计算点到中心的距离分布
    distances = np.linalg.norm(points - center, axis=1)
    print(f"平均距离中心: {distances.mean():.2f}")
    print(f"最大距离中心: {distances.max():.2f}")
    
    return center, size, variance

def visualize_alignment(source, target, transformation=None):
    """可视化两个点云的对齐情况"""
    # 给点云上色
    source_vis = source.paint_uniform_color([1, 0.706, 0])  # 黄色
    target_vis = target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
    
    if transformation is not None:
        source_vis.transform(transformation)
    
    print("\n显示点云对齐情况（黄色=源点云，蓝色=目标点云）")
    o3d.visualization.draw_geometries([source_vis, target_vis], 
                                    window_name="点云对齐诊断")

def check_overlap(source, target, transformation=None, threshold=0.1):
    """检查两个点云的重叠程度"""
    if transformation is not None:
        source_transformed = source.transform(transformation)
    else:
        source_transformed = source
    
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    
    # 使用KDTree查找邻近点
    target_tree = o3d.geometry.KDTreeFlann(target)
    
    overlap_count = 0
    for point in source_points:
        _, idx, dist = target_tree.search_knn_vector_3d(point, 1)
        if dist[0] < threshold:
            overlap_count += 1
    
    overlap_ratio = overlap_count / len(source_points)
    print(f"\n重叠分析:")
    print(f"源点云点数: {len(source_points)}")
    print(f"重叠点数: {overlap_count}")
    print(f"重叠比例: {overlap_ratio:.2%}")
    
    return overlap_ratio

def diagnose_fusion(source_path, target_path):
    """诊断点云融合问题"""
    print("=== 点云融合诊断开始 ===")
    
    # 加载点云
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    # 分析单个点云
    source_center, source_size, source_var = analyze_point_cloud(source, "源点云")
    target_center, target_size, target_var = analyze_point_cloud(target, "目标点云")
    
    # 计算两个点云中心的距离
    center_distance = np.linalg.norm(source_center - target_center)
    print(f"\n两个点云中心距离: {center_distance:.2f}")
    
    # 检查初始对齐情况
    print("\n=== 初始对齐情况 ===")
    initial_overlap = check_overlap(source, target, threshold=2.0)
    visualize_alignment(source, target)
    
    # 尝试简单的配准
    print("\n=== 尝试简单配准 ===")
    
    # 下采样
    voxel_size = 0.5
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    print(f"下采样后 - 源点云: {len(source_down.points)}点, 目标点云: {len(target_down.points)}点")
    
    # 计算法向量
    source_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    target_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # 粗配准
    try:
        # 使用RANSAC进行粗配准
        distance_threshold = voxel_size * 1.5
        ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, 
            o3d.pipelines.registration.compute_fpfh_feature(
                source_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)),
            o3d.pipelines.registration.compute_fpfh_feature(
                target_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5, max_nn=100)),
            True, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
        
        print(f"RANSAC找到 {len(ransac_result.correspondence_set)} 个对应点")
        
        if len(ransac_result.correspondence_set) > 10:
            # 使用TEASER++
            corrs = np.asarray(ransac_result.correspondence_set)
            source_points = np.asarray(source_down.points)[corrs[:, 0]].T
            target_points = np.asarray(target_down.points)[corrs[:, 1]].T
            
            solver_params = teaserpp_python.RobustRegistrationSolver.Params()
            solver_params.noise_bound = voxel_size * 2
            solver_params.cbar2 = 1.0
            solver_params.estimate_scaling = False
            
            solver = teaserpp_python.RobustRegistrationSolver(solver_params)
            solver.solve(source_points, target_points)
            
            solution = solver.getSolution()
            transformation = np.eye(4)
            transformation[:3, :3] = solution.rotation
            transformation[:3, 3] = solution.translation
            
            print("TEASER++配准成功")
            print("变换矩阵:\n", transformation)
            
            # 检查配准后的重叠
            final_overlap = check_overlap(source, target, transformation, threshold=2.0)
            visualize_alignment(source, target, transformation)
            
        else:
            print("警告：对应点太少，配准可能失败")
            transformation = np.eye(4)
            
    except Exception as e:
        print(f"配准失败: {e}")
        transformation = np.eye(4)
    
    # 给出诊断建议
    print("\n=== 诊断建议 ===")
    
    if center_distance > 20:
        print("⚠️  两个点云中心距离过大，可能来自完全不同的视角")
    
    if initial_overlap < 0.1:
        print("⚠️  初始重叠度过低，两个点云可能来自相似视角")
    
    if max(source_size) < 10 or max(target_size) < 10:
        print("⚠️  点云尺寸过小，可能只包含物体的局部")
    
    # 检查是否需要手动调整
    print("\n💡 建议:")
    print("1. 检查两个点云是否真的来自苹果的不同视角")
    print("2. 如果视角相似，尝试从更大角度差异的视角获取点云")
    print("3. 考虑手动调整一个点云的初始位置")
    print("4. 尝试不同的体素大小参数")

if __name__ == "__main__":
    # 诊断苹果点云融合
    diagnose_fusion("mirror/inputs/left_apple_centered.ply", 
                   "mirror/inputs/right_apple_centered.ply")
