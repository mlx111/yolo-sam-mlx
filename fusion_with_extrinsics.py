#!/usr/bin/env python3
"""
利用相机外参进行增强的点云融合
即使外参不够精确，也可以作为初始估计提升融合效果
"""

import numpy as np
import open3d as o3d
import teaserpp_python
import time
import json
from zhongxin import normalize_to_origin
class ExtrinsicsEnhancedFusion:
    def __init__(self, extrinsics_left=None, extrinsics_right=None):
        """
        初始化融合器
        
        Args:
            extrinsics_left: 左相机外参矩阵 (4x4)
            extrinsics_right: 右相机外参矩阵 (4x4)
        """
        self.extrinsics_left = extrinsics_left
        self.extrinsics_right = extrinsics_right
        
        # 初始化相机外参
        self.rotation_matrix = None
        self.translation_vector = None
        self.camera_extrinsics = None
        
    def load_extrinsics_from_file(self, file_path):
        """从文件加载外参"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # 假设外参格式为 {"left": 4x4_matrix, "right": 4x4_matrix}
            if 'left' in data:
                self.extrinsics_left = np.array(data['left'])
            if 'right' in data:
                self.extrinsics_right = np.array(data['right'])
                
            print(f"成功加载外参文件: {file_path}")
            return True
            
        except Exception as e:
            print(f"加载外参文件失败: {e}")
            return False
    
    def set_camera_extrinsics(self, rotation_matrix, translation_vector):
        """
        设置相机间的相对外参（右相机相对于左相机）
        
        Args:
            rotation_matrix: 3x3 旋转矩阵
            translation_vector: 3x1 平移向量（mm）
        """
        # 构建4x4变换矩阵
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation_matrix
        extrinsics[:3, 3] = translation_vector
        
        # 保存到实例变量
        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.camera_extrinsics = extrinsics
        
        print("相机外参设置成功:")
        print("旋转矩阵:")
        print(rotation_matrix)
        print("平移向量 (mm):", translation_vector)
        print("完整变换矩阵:")
        print(extrinsics)
        
        return extrinsics
    
    def create_example_extrinsics(self):
        """使用你提供的实际外参数据"""
        # 你提供的实际外参数据
        rotation_matrix = np.array([
            [0.64348702, -0.24681324, 0.72457414],
            [0.17130231, 0.96901536, 0.17794592],
            [-0.74604288, 0.00961534, 0.66582848]
        ])
        
        translation_vector = np.array([-890.21218409, -121.98498635, 871.98401914])
        #translation_vector = np.array([0,0,0])
        self.set_camera_extrinsics(rotation_matrix, translation_vector)
    
    def transform_to_common_frame(self, pcd, extrinsics):
        """将点云转换到公共坐标系"""
        # 外参是世界坐标系到相机坐标系的变换
        # 我们需要其逆变换来将点云转换到世界坐标系
        world_transform = np.linalg.inv(extrinsics)
        pcd.transform(world_transform)
        return pcd
    
    def auto_scale_detection(self, source_path, target_path):
        """
        自动检测合适的缩放因子
        """
        print("\n=== 自动缩放检测 ===")
        
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)
        
        # 计算点云的尺寸
        source_size = np.max(source_points, axis=0) - np.min(source_points, axis=0)
        target_size = np.max(target_points, axis=0) - np.min(target_points, axis=0)
        
        print(f"源点云尺寸: {source_size}")
        print(f"目标点云尺寸: {target_size}")
        
        # 估计物体大小
        estimated_object_size = np.max([np.mean(source_size), np.mean(target_size)])
        print(f"估计物体尺寸: {estimated_object_size:.2f}")
        
        # 计算相机距离
        camera_distance = np.linalg.norm(self.translation_vector)
        print(f"相机距离: {camera_distance:.2f} mm")
        
        # 自动判断缩放因子
        if camera_distance > estimated_object_size * 10:
            # 相机距离远大于物体尺寸，需要缩放
            scale_factor = estimated_object_size / 100  # 假设物体应该在100mm左右
            print(f"检测到单位不匹配，建议缩放因子: {scale_factor:.6f}")
            return scale_factor
        elif camera_distance < estimated_object_size * 0.1:
            # 相机距离远小于物体尺寸，可能需要反向缩放
            scale_factor = estimated_object_size / camera_distance
            print(f"检测到比例过小，建议缩放因子: {scale_factor:.6f}")
            return scale_factor
        else:
            print("单位匹配，使用缩放因子 1.0")
            return 1.0
    
    def enhanced_fusion(self, source_path, target_path, use_extrinsics=True, 
                     refine_with_registration=True, voxel_size=0.5, auto_scale=True):
        """
        使用相机外参进行增强的点云融合
        
        Args:
            source_path: 源点云路径（左相机）
            target_path: 目标点云路径（右相机）
            use_extrinsics: 是否使用外参进行初始对齐
            refine_with_registration: 是否使用配准算法进行精细调整
            voxel_size: 体素大小
            auto_scale: 是否自动检测缩放因子
        """
        print("=== 相机外参增强点云融合 ===")
        
        # 自动检测缩放因子
        scale_factor = 1.0
        if auto_scale and use_extrinsics:
            scale_factor = self.auto_scale_detection(source_path, target_path)
        
        # 加载点云
        source = o3d.io.read_point_cloud(source_path)
        target = o3d.io.read_point_cloud(target_path)
        
        print(f"原始点云 - 源(左): {len(source.points)}点, 目标(右): {len(target.points)}点")
        
        # 给点云上色
        source.paint_uniform_color([1, 0.706, 0])  # 黄色
        target.paint_uniform_color([0, 0.651, 0.929])  # 蓝色
        
        initial_transform = np.eye(4)
        
        if use_extrinsics and hasattr(self, 'camera_extrinsics'):
            print("使用相机外参进行初始对齐...")
            
            # 应用缩放因子到平移向量
            scaled_extrinsics = self.camera_extrinsics.copy()
            scaled_extrinsics[:3, 3] *= scale_factor
            
            print(f"使用缩放因子: {scale_factor}")
            print("缩放后的变换矩阵:")
            print(scaled_extrinsics)
            
            # 相机外参是右相机相对于左相机的变换
            # 我们需要将右相机点云变换到左相机坐标系
            initial_transform = scaled_extrinsics
            
            print("初始变换矩阵（基于相机外参）:")
            print(initial_transform)
            
            # 应用初始变换：将右相机点云变换到左相机坐标系
            target.transform(initial_transform)
            
            # 可视化初始对齐结果
            print("显示基于相机外参的初始对齐结果...")
            o3d.visualization.draw_geometries([source, target], 
                                           window_name="基于相机外参的初始对齐")
        
        if refine_with_registration:
            print("使用配准算法进行精细调整...")
            
            # 下采样
            source_down = source.voxel_down_sample(voxel_size)
            target_down = target.voxel_down_sample(voxel_size)
            
            print(f"下采样后 - 源: {len(source_down.points)}点, 目标: {len(target_down.points)}点")
            
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
            
            # TEASER++配准（使用相机外参作为初始估计）
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
                
                # 配置TEASER++，使用更保守的参数因为我们有好的初始估计
                solver_params = teaserpp_python.RobustRegistrationSolver.Params()
                solver_params.noise_bound = voxel_size * 2  # 可以适当增大，因为外参已有一定精度
                solver_params.cbar2 = 1.0
                solver_params.estimate_scaling = False
                solver_params.rotation_gnc_factor = 1.1  # 更保守的因子
                solver_params.rotation_max_iterations = 50  # 减少迭代次数
                
                solver = teaserpp_python.RobustRegistrationSolver(solver_params)
                solver.solve(source_points, target_points)
                
                solution = solver.getSolution()
                
                # 构建精细变换矩阵（相对于初始变换的增量）
                refinement_transform = np.eye(4)
                refinement_transform[:3, :3] = solution.rotation
                refinement_transform[:3, 3] = solution.translation
                
                print("精细变换增量:")
                print(refinement_transform)
                
                # 组合初始变换和精细变换
                # 注意：target已经被初始变换过了，所以这里只需要应用增量
                final_transform = refinement_transform
                
            else:
                print("对应点太少，使用初始变换")
                final_transform = np.eye(4)
            
            # ICP精配准
            source.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            target.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
            
            icp_result = o3d.pipelines.registration.registration_icp(
                source, target, voxel_size * 2, final_transform,
                o3d.pipelines.registration.TransformationEstimationPointToPlane()
            )
            
            final_transform = icp_result.transformation
            print("ICP最终变换增量:")
            print(final_transform)
            
            # 计算总体变换矩阵
            total_transform = final_transform @ initial_transform
            print("总体变换矩阵:")
            print(total_transform)
        
        else:
            total_transform = initial_transform
        
        # 应用最终变换到源点云（如果需要的话）
        # 注意：target已经被变换了，所以我们只需要变换source来对齐
        if refine_with_registration:
            source.transform(final_transform)
        
        # 融合
        combined_pcd = source + target
        fused_pcd = combined_pcd.voxel_down_sample(voxel_size=0.2)
        
        print(f"融合后点云点数: {len(fused_pcd.points)}")
        
        # 可视化最终结果
        o3d.visualization.draw_geometries([fused_pcd], window_name="相机外参增强融合结果")
        
        # 保存结果
        output_path = "outputs/apple_camera_extrinsics_fused.ply"
        o3d.io.write_point_cloud(output_path, fused_pcd)
        print(f"结果已保存至: {output_path}")
        
        return fused_pcd, total_transform
    
    def compare_methods(self, source_path, target_path):
        """比较不同融合方法的效果"""
        print("=== 融合方法对比 ===")
        
        results = {}
        
        # 方法1：仅使用外参
        print("\n1. 仅使用外参...")
        fused1, trans1 = self.enhanced_fusion(source_path, target_path, 
                                           use_extrinsics=True, 
                                           refine_with_registration=False)
        results['extrinsics_only'] = (fused1, trans1)
        
        # 方法2：仅使用配准
        print("\n2. 仅使用配准...")
        fused2, trans2 = self.enhanced_fusion(source_path, target_path, 
                                           use_extrinsics=False, 
                                           refine_with_registration=True)
        results['registration_only'] = (fused2, trans2)
        
        # 方法3：外参+配准
        print("\n3. 外参+配准...")
        fused3, trans3 = self.enhanced_fusion(source_path, target_path, 
                                           use_extrinsics=True, 
                                           refine_with_registration=True)
        results['extrinsics_plus_registration'] = (fused3, trans3)
        
        # 保存所有结果
        for method, (fused_pcd, transform) in results.items():
            output_path = f"outputs/apple_fused_{method}.ply"

            o3d.io.write_point_cloud(output_path, fused_pcd)
            print(f"{method} 结果已保存至: {output_path}")
            #normalize_to_origin(output_path,output_path)
        
        return results

def create_extrinsics_template():
    """创建外参文件模板"""
    template = {
        "left": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "right": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    }
    
    with open("camera_extrinsics_template.json", "w") as f:
        json.dump(template, f, indent=2)
    
    print("已创建外参模板文件: camera_extrinsics_template.json")
    print("请编辑此文件，填入你的实际外参矩阵")

if __name__ == "__main__":
    print("=== 外参增强点云融合系统 ===")
    
    # 创建外参模板
    create_extrinsics_template()
    
    # 初始化融合器
    fusion = ExtrinsicsEnhancedFusion()
    
    # 方案1：使用示例外参
    print("\n=== 使用示例外参 ===")
    fusion.create_example_extrinsics()
    ply1="outputs/left_apple_centered.ply"
    ply2="outputs/right_apple_centered.ply"
    #ply1="outputs/left1_background.ply"
    #ply2="outputs/right1_background.ply"
    # 进行融合
    fused_pcd, final_transform = fusion.enhanced_fusion(
        ply1,
        ply2,
        use_extrinsics=True,
        refine_with_registration=True
    )
    
    # 方案2：比较不同方法
    print("\n=== 比较不同融合方法 ===")
    fusion.compare_methods(ply1,ply2)
    
    print("\n=== 完成 ===")
    print("请检查生成的融合结果，选择效果最好的方法")
    print("如果要使用你的实际外参，请编辑 camera_extrinsics_template.json 文件")
