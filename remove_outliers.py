#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
点云离群点去除Demo
使用统计学离群点移除（SOR）和半径离群点移除（ROR）算法
"""

import numpy as np
import open3d as o3d
import os
import sys

def remove_outliers_statistical(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    使用统计学离群点移除（SOR）算法
    
    Args:
        pcd: 输入点云
        nb_neighbors: 考虑的邻居数量
        std_ratio: 标准差比率阈值
    
    Returns:
        cl_pcd: 清理后的点云
        ind: 内点索引
    """
    print(f"[*] 使用统计学离群点移除 (SOR)")
    print(f"    - 邻居数量: {nb_neighbors}")
    print(f"    - 标准差比率: {std_ratio}")
    
    cl_pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    print(f"[*] SOR处理结果:")
    print(f"    - 原始点数: {len(pcd.points)}")
    print(f"    - 保留点数: {len(cl_pcd.points)}")
    print(f"    - 移除点数: {len(pcd.points) - len(cl_pcd.points)}")
    print(f"    - 保留率: {len(cl_pcd.points) / len(pcd.points) * 100:.2f}%")
    
    return cl_pcd, ind

def remove_outliers_radius(pcd, nb_points=16, radius=0.05):
    """
    使用半径离群点移除（ROR）算法
    
    Args:
        pcd: 输入点云
        nb_points: 半径内最少点数
        radius: 搜索半径
    
    Returns:
        cl_pcd: 清理后的点云
        ind: 内点索引
    """
    print(f"[*] 使用半径离群点移除 (ROR)")
    print(f"    - 半径内最少点数: {nb_points}")
    print(f"    - 搜索半径: {radius}")
    
    cl_pcd, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
    
    print(f"[*] ROR处理结果:")
    print(f"    - 原始点数: {len(pcd.points)}")
    print(f"    - 保留点数: {len(cl_pcd.points)}")
    print(f"    - 移除点数: {len(pcd.points) - len(cl_pcd.points)}")
    print(f"    - 保留率: {len(cl_pcd.points) / len(pcd.points) * 100:.2f}%")
    
    return cl_pcd, ind

def visualize_before_after(original_pcd, cleaned_pcd):
    """
    可视化处理前后的点云对比
    """
    print(f"[*] 正在生成对比可视化...")
    
    # 为原始点云设置颜色（红色表示离群点）
    original_colors = np.ones((len(original_pcd.points), 3)) * [0.8, 0.8, 0.8]  # 灰色
    
    # 为清理后的点云设置颜色（蓝色）
    cleaned_colors = np.ones((len(cleaned_pcd.points), 3)) * [0, 0, 1]  # 蓝色
    
    original_pcd.colors = o3d.utility.Vector3dVector(original_colors)
    cleaned_pcd.colors = o3d.utility.Vector3dVector(cleaned_colors)
    
    # 创建可视化窗口
    o3d.visualization.draw_geometries([original_pcd], 
                                     window_name="原始点云（含离群点）",
                                     width=1200, height=800)
    
    o3d.visualization.draw_geometries([cleaned_pcd], 
                                     window_name="清理后点云",
                                     width=1200, height=800)

def main():
    """主函数"""
    
    # ==================== 配置参数 ====================
    # 在这里修改所有输入参数
    input_file = "outputs/merged_apple_final.ply"  # 输入点云文件
    output_file = "outputs/merged_apple_final_cleaned.ply"  # 输出文件
    
    # 离群点移除参数
    use_sor = True  # 是否使用统计学离群点移除
    sor_nb_neighbors = 20  # SOR邻居数量
    sor_std_ratio = 2.0  # SOR标准差比率
    
    use_ror = True  # 是否使用半径离群点移除
    ror_nb_points = 16  # ROR半径内最少点数
    ror_radius = 0.05  # ROR搜索半径
    
    # 可视化设置
    show_visualization = False  # 是否显示可视化
    
    # 获取当前脚本所在的绝对目录
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(BASE_DIR, input_file)
    output_path = os.path.join(BASE_DIR, output_file)
    
    # 检查文件是否存在
    if not os.path.exists(input_path):
        print(f"[!] 错误：输入文件不存在: {input_path}")
        return
    
    print("=" * 60)
    print("点云离群点去除 Demo")
    print("=" * 60)
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"使用SOR: {use_sor}")
    print(f"使用ROR: {use_ror}")
    print("-" * 60)
    
    try:
        # ==================== 读取点云 ====================
        print(f"[*] 正在读取点云文件...")
        original_pcd = o3d.io.read_point_cloud(input_path)
        original_points = np.asarray(original_pcd.points)
        
        print(f"[*] 原始点云信息:")
        print(f"    - 点数: {len(original_points)}")
        
        # 计算点云边界
        if len(original_points) > 0:
            min_bounds = np.min(original_points, axis=0)
            max_bounds = np.max(original_points, axis=0)
            dimensions = max_bounds - min_bounds
            print(f"    - 边界: [{min_bounds}, {max_bounds}]")
            print(f"    - 尺寸: {dimensions}")
        
        current_pcd = original_pcd
        
        # ==================== 统计学离群点移除 ====================
        if use_sor:
            print(f"\n[*] 开始统计学离群点移除...")
            current_pcd, sor_ind = remove_outliers_statistical(
                current_pcd, sor_nb_neighbors, sor_std_ratio
            )
        
        # ==================== 半径离群点移除 ====================
        if use_ror:
            print(f"\n[*] 开始半径离群点移除...")
            current_pcd, ror_ind = remove_outliers_radius(
                current_pcd, ror_nb_points, ror_radius
            )
        
        # ==================== 保存结果 ====================
        print(f"\n[*] 正在保存清理后的点云...")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存清理后的点云
        o3d.io.write_point_cloud(output_path, current_pcd)
        print(f"[*] 清理后的点云已保存至: {output_path}")
        
        # 保存为不同格式（可选）
        output_path_obj = output_path.replace('.ply', '.obj')
        o3d.io.write_point_cloud(output_path_obj, current_pcd)
        print(f"[*] OBJ格式文件已保存至: {output_path_obj}")
        
        # ==================== 可视化 ====================
        if show_visualization:
            visualize_before_after(original_pcd, current_pcd)
        
        # ==================== 最终统计 ====================
        print(f"\n[*] 最终统计:")
        print(f"    - 原始点数: {len(original_points)}")
        print(f"    - 最终点数: {len(current_pcd.points)}")
        print(f"    - 总移除点数: {len(original_points) - len(current_pcd.points)}")
        print(f"    - 最终保留率: {len(current_pcd.points) / len(original_points) * 100:.2f}%")
        
        # 计算清理后的点云边界
        cleaned_points = np.asarray(current_pcd.points)
        if len(cleaned_points) > 0:
            min_bounds_clean = np.min(cleaned_points, axis=0)
            max_bounds_clean = np.max(cleaned_points, axis=0)
            dimensions_clean = max_bounds_clean - min_bounds_clean
            print(f"    - 清理后边界: [{min_bounds_clean}, {max_bounds_clean}]")
            print(f"    - 清理后尺寸: {dimensions_clean}")
        
        print(f"\n[*] 点云离群点去除Demo运行完成！")
        
    except Exception as e:
        print(f"[!] 运行出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
