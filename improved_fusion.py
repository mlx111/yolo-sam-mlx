#!/usr/bin/env python3
"""
改进的点云融合方案
针对苹果点云只有一个面的问题提供多种解决方案
"""

import numpy as np
import open3d as o3d
import teaserpp_python
import time

def create_mirror_view(pcd, mirror_axis='x'):
    """
    创建点云的镜像视图，模拟从对面视角观察
    """
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # 创建镜像变换矩阵
    if mirror_axis == 'x':
        transform = np.array([[-1, 0, 0, 0],
                           [0, 1, 0, 0], 
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
    elif mirror_axis == 'y':
        transform = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, 1, 0], 
                           [0, 0, 0, 1]])
    else:  # z axis
        transform = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
    
    # 应用变换
    mirrored_pcd = pcd.transform(transform)
    
    # 添加一些随机扰动，模拟不同视角
    points = np.asarray(mirrored_pcd.points)
    
    # 轻微旋转，模拟视角变化
    angle = np.random.uniform(-0.1, 0.1)  # 小角度旋转
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle])
    mirrored_pcd.rotate(rotation_matrix, center=(0, 0, 0))
    
    return mirrored_pcd

def create_rotated_view(pcd, angle_degrees=180):
    """
    创建旋转视图，模拟从不同角度观察
    """
    angle_rad = np.radians(angle_degrees)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, angle_rad])
    
    rotated_pcd = pcd.rotate(rotation_matrix, center=(0, 0, 0))
    return rotated_pcd

def enhance_point_cloud_depth(pcd, depth_factor=0.3):
    """
    增强点云的深度信息，通过轻微的Z轴偏移模拟厚度
    """
    points = np.asarray(pcd.points)
    
    # 为每个点添加轻微的Z轴偏移
    z_offset = np.random.normal(0, depth_factor, len(points))
    points[:, 2] += z_offset
    
    enhanced_pcd = o3d.geometry.PointCloud()
    enhanced_pcd.points = o3d.utility.Vector3dVector(points)
    
    if pcd.has_colors():
        enhanced_pcd.colors = pcd.colors
    
    return enhanced_pcd

def multi_view_fusion(source_path, target_path, method='mirror'):
    """
    多视角融合方案
    """
    print(f"=== 使用 {method} 方法进行多视角融合 ===")
    
    # 加载原始点云
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    print(f"原始点云 - 源: {len(source.points)}点, 目标: {len(target.points)}点")
    
    # 创建额外的视角
    if method == 'mirror':
        # 创建镜像视图
        source_mirror = create_mirror_view(source, mirror_axis='x')
        target_mirror = create_mirror_view(target, mirror_axis='y')
        
        # 组合多个视角
        multi_view_source = source + source_mirror
        multi_view_target = target + target_mirror
        
    elif method == 'rotate':
        # 创建旋转视图
        source_rotated = create_rotated_view(source, 120)
        target_rotated = create_rotated_view(target, -120)
        
        # 组合多个视角
        multi_view_source = source + source_rotated
        multi_view_target = target + target_rotated
        
    elif method == 'depth':
        # 增强深度信息
        multi_view_source = enhance_point_cloud_depth(source)
        multi_view_target = enhance_point_cloud_depth(target)
    
    print(f"多视角点云 - 源: {len(multi_view_source.points)}点, 目标: {len(multi_view_target.points)}点")
    
    # 给点云上色
    multi_view_source.paint_uniform_color([1, 0.706, 0])  # 黄色
    multi_view_target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
    
    # 进行配准
    voxel_size = 0.5
    
    # 下采样
    source_down = multi_view_source.voxel_down_sample(voxel_size)
    target_down = multi_view_target.voxel_down_sample(voxel_size)
    
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
    
    # TEASER++配准
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
    
    if len(corrs) > 10:
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
    else:
        print("对应点太少，使用单位矩阵")
        transformation = np.eye(4)
    
    # ICP精配准
    multi_view_source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    multi_view_target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    icp_result = o3d.pipelines.registration.registration_icp(
        multi_view_source, multi_view_target, voxel_size * 2, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    
    final_trans = icp_result.transformation
    
    # 融合
    multi_view_source.transform(final_trans)
    combined_pcd = multi_view_source + multi_view_target
    fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.2)
    
    print(f"融合后点云点数: {len(fused_pcd.points)}")
    
    # 可视化
    o3d.visualization.draw_geometries([fused_pcd], window_name=f"多视角融合结果 ({method})")
    
    # 保存
    output_path = f"outputs/apple_multi_view_fused_{method}.ply"
    o3d.io.write_point_cloud(output_path, fused_pcd)
    print(f"结果已保存至: {output_path}")
    
    return fused_pcd

def create_synthetic_apple():
    """
    创建一个合成的完整苹果点云作为参考
    """
    print("=== 创建合成苹果点云 ===")
    
    # 创建球体作为苹果的基本形状
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=30)
    mesh.compute_vertex_normals()
    
    # 添加一些变形使其更像苹果
    vertices = np.asarray(mesh.vertices)
    
    # 苹果通常不是完美的球体，底部稍平，顶部稍尖
    for i, v in enumerate(vertices):
        # 底部压扁
        if v[2] < -10:
            vertices[i][2] *= 0.7
        # 顶部拉长
        elif v[2] > 10:
            vertices[i][2] *= 1.2
            vertices[i][0] *= 0.9
            vertices[i][1] *= 0.9
    
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    
    # 转换为点云
    pcd = mesh.sample_points_poisson_disk(number_of_points=5000)
    
    # 添加一些噪声使其更真实
    points = np.asarray(pcd.points)
    noise = np.random.normal(0, 0.5, points.shape)
    points += noise
    pcd.points = o3d.utility.Vector3dVector(points)
    
    pcd.paint_uniform_color([0.8, 0.2, 0.1])  # 苹果红色
    
    o3d.visualization.draw_geometries([pcd], window_name="合成苹果点云")
    o3d.io.write_point_cloud("outputs/synthetic_apple.ply", pcd)
    
    return pcd

if __name__ == "__main__":
    print("=== 苹果点云多视角融合方案 ===")
    
    # 方案1：镜像融合
    print("\n1. 尝试镜像融合...")
    multi_view_fusion("mirror/inputs/left_apple_centered.ply", 
                     "mirror/inputs/right_apple_centered.ply", 
                     method='mirror')
    
    # 方案2：旋转融合
    print("\n2. 尝试旋转融合...")
    multi_view_fusion("mirror/inputs/left_apple_centered.ply", 
                     "mirror/inputs/right_apple_centered.ply", 
                     method='rotate')
    
    # 方案3：深度增强融合
    print("\n3. 尝试深度增强融合...")
    multi_view_fusion("mirror/inputs/left_apple_centered.ply", 
                     "mirror/inputs/right_apple_centered.ply", 
                     method='depth')
    
    # 方案4：创建合成苹果作为参考
    print("\n4. 创建合成苹果作为参考...")
    synthetic_apple = create_synthetic_apple()
    
    print("\n=== 完成 ===")
    print("请检查生成的融合结果，选择效果最好的方案")
