#!/usr/bin/env python3
"""
点云查看工具
用于查看保存的点云文件
"""

import numpy as np
import open3d as o3d
import os
import sys

def load_and_visualize(file_path, color=[0, 1, 0]):
    """
    加载并可视化点云文件
    """
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return None
    
    # 根据文件扩展名选择加载方式
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    elif file_path.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
    else:
        print(f"不支持的文件格式: {file_path}")
        return None
    
    print(f"加载点云: {file_path}")
    print(f"点云数量: {len(points)}")
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    
    return pcd

def main():
    """主函数"""
    print("📊 点云查看工具")
    print("="*40)
    
    # 检查文件
    files_to_check = [
        "outputs/right_apple.npy",      # 原始输入
        "outputs/complete_apple.npy",   # 补全结果 (NPY)
        "outputs/complete_apple.ply"    # 补全结果 (PLY)
    ]
    
    point_clouds = []
    labels = []
    
    # 加载输入点云（红色）
    input_pcd = load_and_visualize("outputs/right_apple.npy", color=[1, 0, 0])
    if input_pcd:
        point_clouds.append(input_pcd)
        labels.append("输入 (红色)")
    
    # 加载补全结果（绿色）
    complete_pcd = load_and_visualize("outputs/complete_apple.npy", color=[0, 1, 0])
    if complete_pcd:
        point_clouds.append(complete_pcd)
        labels.append("补全 (绿色)")
    
    if not point_clouds:
        print("没有找到可用的点云文件！")
        print("请先运行 safe_pointcloud_completion.py 生成点云文件。")
        return
    
    print("\n🎨 可视化说明:")
    for label in labels:
        print(f"   - {label}")
    print("\n控制说明:")
    print("   - 鼠标左键拖拽: 旋转")
    print("   - 鼠标右键拖拽: 平移")
    print("   - 鼠标滚轮: 缩放")
    print("   - 关闭窗口退出")
    
    # 可视化
    o3d.visualization.draw_geometries(
        point_clouds, 
        window_name="点云查看器",
        width=1200,
        height=800
    )
    
    print("\n✅ 查看完成！")

if __name__ == "__main__":
    main()
