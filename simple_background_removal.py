#!/usr/bin/env python3
"""
简化版点云背景去除系统
高效去除墙壁、地面等背景，保留机械臂、传送带和物体
"""

import numpy as np
import open3d as o3d

class SimpleBackgroundRemoval:
    def __init__(self):
        """初始化背景去除器"""
        self.plane_threshold = 0.02  # 平面检测阈值
        
    def load_point_cloud(self, file_path):
        """加载点云文件"""
        pcd = o3d.io.read_point_cloud(file_path)
        print(f"加载点云: {file_path}")
        print(f"点云点数: {len(pcd.points)}")
        return pcd
    
    def analyze_point_cloud(self, pcd):
        """分析点云特征"""
        points = np.asarray(pcd.points)
        
        # 计算点云边界
        min_bounds = np.min(points, axis=0)
        max_bounds = np.max(points, axis=0)
        
        print(f"\n=== 点云分析 ===")
        print(f"最小边界: {min_bounds}")
        print(f"最大边界: {max_bounds}")
        print(f"Z轴范围: [{min_bounds[2]:.3f}, {max_bounds[2]:.3f}]")
        
        return min_bounds, max_bounds
    
    def remove_ground_and_ceiling(self, pcd, min_percentile=5, max_percentile=99):
        """去除地面和天花板"""
        points = np.asarray(pcd.points)
        z_coords = points[:, 2]
        
        # 计算高度阈值
        z_min = np.percentile(z_coords, min_percentile)
        z_max = np.percentile(z_coords, max_percentile)
        
        print(f"\n=== 去除地面和天花板 ===")
        print(f"Z轴范围: [{z_min:.3f}, {z_max:.3f}]")
        
        # 筛选高度范围内的点
        mask = (z_coords >= z_min) & (z_coords <= z_max)
        filtered_points = points[mask]
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        print(f"过滤前点数: {len(points)}")
        print(f"过滤后点数: {len(filtered_points)}")
        
        return filtered_pcd
    
    def remove_large_planes(self, pcd, threshold=None, min_points=1000):
        """去除大平面（墙壁）"""
        if threshold is None:
            threshold = self.plane_threshold
            
        print(f"\n=== 去除大平面 ===")
        print(f"平面检测阈值: {threshold}")
        
        remaining_pcd = pcd
        total_removed = 0
        
        # 迭代去除多个平面
        for i in range(5):  # 最多检测5个平面
            if len(remaining_pcd.points) < min_points:
                break
                
            # 平面分割
            plane_model, inliers = remaining_pcd.segment_plane(
                distance_threshold=threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            # 如果平面点数足够多，认为是墙壁/地面
            if len(inliers) >= min_points:
                # 提取平面点
                plane_cloud = remaining_pcd.select_by_index(inliers)
                remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)
                
                # 平面方程: ax + by + cz + d = 0
                a, b, c, d = plane_model
                print(f"平面 {i+1}: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
                print(f"平面点数: {len(inliers)}")
                
                total_removed += len(inliers)
            else:
                break
        
        print(f"总共去除平面点数: {total_removed}")
        print(f"剩余点数: {len(remaining_pcd.points)}")
        
        return remaining_pcd
    
    def remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """去除离群点"""
        print(f"\n=== 去除离群点 ===")
        
        # 统计离群点去除
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        filtered_pcd = pcd.select_by_index(ind)
        
        print(f"统计过滤前点数: {len(pcd.points)}")
        print(f"统计过滤后点数: {len(filtered_pcd.points)}")
        
        return filtered_pcd
    
    def filter_by_distance_from_center(self, pcd, max_distance=None):
        """根据距离中心的距离过滤"""
        points = np.asarray(pcd.points)
        
        # 计算点云中心
        center = np.mean(points, axis=0)
        
        # 计算每个点到中心的距离
        distances = np.linalg.norm(points - center, axis=1)
        
        if max_distance is None:
            # 自动计算合理距离阈值
            max_distance = np.percentile(distances, 90)
        
        print(f"\n=== 根据距离中心过滤 ===")
        print(f"中心位置: {center}")
        print(f"最大距离: {max_distance:.3f}")
        
        # 筛选距离范围内的点
        mask = distances <= max_distance
        filtered_points = points[mask]
        
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
        
        print(f"距离过滤前点数: {len(points)}")
        print(f"距离过滤后点数: {len(filtered_points)}")
        
        return filtered_pcd
    
    def comprehensive_removal(self, input_path, output_path=None):
        """综合背景去除流程"""
        print("=== 简化版背景去除流程 ===")
        
        # 加载点云
        pcd = self.load_point_cloud(input_path)
        
        # 分析点云
        min_bounds, max_bounds = self.analyze_point_cloud(pcd)
        
        # 步骤1: 去除地面和天花板
        pcd_filtered = self.remove_ground_and_ceiling(pcd)
        
        # 步骤2: 去除大平面（墙壁）
        pcd_no_planes = self.remove_large_planes(pcd_filtered, min_points=3000)
        
        # 步骤3: 根据距离中心过滤（去除远处的背景）
        pcd_distance_filtered = self.filter_by_distance_from_center(pcd_no_planes)
        
        # 步骤4: 去除离群点
        pcd_final = self.remove_outliers(pcd_distance_filtered)
        
        # 最终统计
        print(f"\n=== 最终结果 ===")
        print(f"原始点数: {len(pcd.points)}")
        print(f"最终点数: {len(pcd_final.points)}")
        print(f"保留比例: {len(pcd_final.points) / len(pcd.points) * 100:.1f}%")
        
        # 可视化最终结果
        pcd_final.paint_uniform_color([0, 1, 0])  # 绿色
        o3d.visualization.draw_geometries([pcd_final], window_name="背景去除结果")
        
        # 保存结果
        if output_path:
            o3d.io.write_point_cloud(output_path, pcd_final)
            print(f"结果已保存至: {output_path}")
        
        return pcd_final
    
    def process_multiple_files(self, input_files, output_dir="outputs/"):
        """批量处理多个文件"""
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for input_file in input_files:
            filename = os.path.basename(input_file)
            name, ext = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{name}_no_background{ext}")
            
            print(f"\n{'='*50}")
            print(f"处理文件: {filename}")
            print(f"{'='*50}")
            
            self.comprehensive_removal(input_file, output_file)

def main():
    """主函数"""
    print("=== 简化版点云背景去除系统 ===")
    
    # 创建背景去除器
    remover = SimpleBackgroundRemoval()
    
    # 处理融合后的全局点云
    input_files = [
        "outputs/right1_background.ply",
        "outputs/left1_background.ply"
    ]
    
    # 批量处理
    remover.process_multiple_files(input_files)
    
    print("\n=== 完成 ===")
    print("背景去除完成，请查看outputs目录中的结果文件")

if __name__ == "__main__":
    main()
