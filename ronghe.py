'''import numpy as np
import open3d as o3d
import teaserpp_python
import time

def preprocess_point_cloud(pcd, voxel_size):
    """
    预处理：下采样、估计法向量、计算FPFH特征
    """
    # 1. 下采样
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # 2. 估计法向量（ICP和特征计算需要）
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # 3. 计算FPFH特征
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def teaser_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """
    使用 TEASER++ 进行粗配准
    """
    print(f"源点云下采样后点数: {len(source_down.points)}")
    print(f"目标点云下采样后点数: {len(target_down.points)}")
    
    # 建立初步对应关系（基于特征空间距离）
    distance_threshold = voxel_size * 1.5
    result_feature_matching = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    # 提取对应点的坐标
    corrs = np.asarray(result_feature_matching.correspondence_set)
    print(f"特征匹配对应点数: {len(corrs)}")
    
    if len(corrs) < 10:
        print("警告：对应点太少，尝试使用更宽松的参数...")
        # 使用更宽松的参数重新匹配
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
        
        if len(corrs) < 5:
            print("错误：对应点太少，无法进行配准")
            return np.eye(4)  # 返回单位矩阵
    
    source_points = np.asarray(source_down.points)[corrs[:, 0]].T
    target_points = np.asarray(target_down.points)[corrs[:, 1]].T

    # 配置 TEASER++ 参数
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.noise_bound = voxel_size * 2  # 增加噪声容忍度
    solver_params.cbar2 = 1.0
    solver_params.estimate_scaling = False  # RGB-D尺度已知，设为False
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    
    start = time.time()
    solver.solve(source_points, target_points)
    end = time.time()
    
    print(f"TEASER++ 求解耗时: {end - start:.4f} s")
    
    solution = solver.getSolution()
    
    # 构建 4x4 变换矩阵
    transformation = np.eye(4)
    transformation[:3, :3] = solution.rotation
    transformation[:3, 3] = solution.translation
    
    return transformation

def main_fusion(source_path, target_path):
    # 1. 加载点云
    # 注意：这里直接读取你 PointCloudGenerator 生成的 .npy 或 .ply
    if source_path.endswith('.npy'):
        src_np = np.load(source_path)
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(src_np)
    else:
        source = o3d.io.read_point_cloud(source_path)

    if target_path.endswith('.npy'):
        tgt_np = np.load(target_path)
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(tgt_np)
    else:
        target = o3d.io.read_point_cloud(target_path)
    print("加载点云完成")
    # 给点云上色以便区分
    source.paint_uniform_color([1, 0, 0])   # 黄色
    target.paint_uniform_color([0, 0, 0.929]) # 蓝色

    # 2. 预处理
    voxel_size = 0.05  # 根据点云间距调整，约为点间距的1/4到1/2
    
    # 为ICP计算法向量（Point-to-Plane需要）
    print("计算法向量...")
    radius_normal = voxel_size * 2
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    print("预处理完成")
    # 3. 粗配准 (TEASER++)
    teaser_trans = teaser_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print("TEASER++ 粗配准矩阵:\n", teaser_trans)

    # 4. 精配准 (Point-to-Plane ICP)
    print("正在进行 ICP 精配准...")
    icp_result = o3d.pipelines.registration.registration_icp(
        source, target, voxel_size * 2, teaser_trans,
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    final_trans = icp_result.transformation
    print("ICP 精配准矩阵:\n", final_trans)

    # 5. 融合与去重
    source.transform(final_trans)
    combined_pcd = source + target
    
    # 通过再次下采样进行体素融合，去除重复点并使密度均匀
    fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02) # 融合时使用更小的体素

    # 6. 可视化结果
    print("显示融合结果...")
    o3d.visualization.draw_geometries([fused_pcd], window_name="点云融合结果")
    
    # 7. 保存结果
    o3d.io.write_point_cloud("outputs/background_fused_scene.ply", fused_pcd)
    print("融合点云已保存至 outputs/background_fused_scene.ply")

if __name__ == "__main__":
    # 替换为你实际生成的文件路径
    #main_fusion("mirror/inputs/left_apple_centered.ply", "mirror/inputs/right_apple_centered.ply")
    main_fusion("outputs/right1_background_no_background.ply", "outputs/left1_background_no_background.ply")'''


import numpy as np
import open3d as o3d

def refine_fusion_with_extrinsics():
    # 1. 填入你提供的外参数据
    R = np.array([
        [0.64348702, -0.24681324, 0.72457414],
        [0.17130231, 0.96901536, 0.17794592],
        [-0.74604288, 0.00961534, 0.66582848]
    ])
    
    # 平移向量从 mm 转换为 m
    t = np.array([-890.21218409, -121.98498635, 871.98401914]) / 1000.0
    
    # 构造 4x4 变换矩阵 (右相对于左)
    T_extrinsic = np.eye(4)
    T_extrinsic[:3, :3] = R
    T_extrinsic[:3, 3] = t

    # 2. 加载点云 (请确保使用原始坐标的点云，不要用 _centered 后缀的文件)
    # 因为外参是基于相机原始坐标系的
    src_pcd = o3d.io.read_point_cloud("outputs/right_apple.ply")
    ref_pcd = o3d.io.read_point_cloud("outputs/left_apple.ply")
    
    if len(src_pcd.points) == 0 or len(ref_pcd.points) == 0:
        print("错误：无法读取点云文件，请检查路径。")
        return

    # 3. 第一步：应用初始外参
    # 这步执行后，苹果的两个面应该已经靠得很近了，但还没重合
    src_pcd.transform(T_extrinsic)
    
    # 4. 第二步：ICP 精修 (Local Refinement)
    # 这里的 threshold 是关键！
    # 如果外参误差在 1cm 左右，我们就设为 0.015
    # 这样它只会微调边缘，而不会把苹果的正面吸到背面去
    print("正在进行局部 ICP 精修...")
    threshold = 0.015 
    
    # 建议使用 Point-to-Plane ICP（需要法向量），如果没法向量就用 Point-to-Point
    src_pcd.estimate_normals()
    ref_pcd.estimate_normals()
    
    reg_p2p = o3d.pipelines.registration.registration_icp(
        src_pcd, ref_pcd, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50)
    )
    
    # 应用微调矩阵
    src_pcd.transform(reg_p2p.transformation)
    print("精修完成。")

    # 5. 合并与保存
    # 染色观察：左视图红色，右视图绿色
    ref_pcd.paint_uniform_color([1, 0, 0])
    src_pcd.paint_uniform_color([0, 1, 0])
    
    fused_pcd = ref_pcd + src_pcd
    
    # 最终去重采样：1mm 步长
    fused_pcd = fused_pcd.voxel_down_sample(voxel_size=0.001)
    
    o3d.io.write_point_cloud("outputs/apple_fixed_fusion.ply", fused_pcd)
    print(f"融合结果已保存，总点数: {len(fused_pcd.points)}")
    
    # 可视化窗口
    o3d.visualization.draw_geometries([fused_pcd], window_name="Extrinsic + ICP Refined")

if __name__ == "__main__":
    refine_fusion_with_extrinsics()