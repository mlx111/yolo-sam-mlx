#!/usr/bin/env python3
import numpy as np
import open3d as o3d
import argparse
import os

def npy_ply(npy_path: str, ply_path: str = None, color: bool = False):
    """
    将 .npy 格式的点云转换为 .ply 格式
    
    Args:
        npy_path: 输入的 .npy 文件路径
        ply_path: 输出的 .ply 文件路径（可选，默认与输入同名）
        color: 是否包含颜色信息（如果 npy 是 Nx6 格式）
    """
    # 加载点云
    points = np.load(npy_path)
    print(f"加载点云: {points.shape}")
    
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    
    if points.shape[1] == 3:
        # 只有坐标 (N, 3)
        pcd.points = o3d.utility.Vector3dVector(points)
        print("检测到纯坐标点云 (Nx3)")
    elif points.shape[1] == 6:
        # 坐标 + 颜色 (N, 6)
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if color:
            # 假设颜色是 0-255 范围，归一化到 0-1
            colors = points[:, 3:6] / 255.0
            colors = np.clip(colors, 0, 1)  # 确保在有效范围内
            pcd.colors = o3d.utility.Vector3dVector(colors)
            print("检测到带颜色点云 (Nx6)，已添加颜色信息")
        else:
            print("检测到带颜色点云 (Nx6)，但未启用颜色保存")
    else:
        raise ValueError(f"不支持的点云格式: {points.shape}，期望 (N, 3) 或 (N, 6)")
    
    # 设置输出路径
    if ply_path is None:
        ply_path = os.path.splitext(npy_path)[0] + ".ply"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(ply_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存为 PLY
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"✅ 已保存到: {ply_path}")
    return ply_path

def main():
    parser = argparse.ArgumentParser(description="将 .npy 点云转换为 .ply 格式")
    parser.add_argument("input", help="输入的 .npy 文件路径")
    parser.add_argument("-o", "--output", help="输出的 .ply 文件路径（可选）")
    parser.add_argument("--color", action="store_true", help="保存颜色信息（如果存在）")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        return
    
    npy_ply(args.input, args.output, args.color)

if __name__ == "__main__":
    main()
