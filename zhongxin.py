import open3d as o3d
import numpy as np

def normalize_to_origin(input_path, output_path):
    # 1. 读取点云
    pcd = o3d.io.read_point_cloud(input_path)
    if pcd.is_empty():
        print("读取失败，文件可能为空")
        return

    # 2. 计算中心 (这里推荐使用 Axis Aligned Bounding Box 的中心)
    # 这样可以确保物体的几何形状在各轴向是对称分布的
    aabb = pcd.get_axis_aligned_bounding_box()
    center = aabb.get_center()
    
    print(f"原始中心坐标: {center}")

    # 3. 平移点云: 将中心点移动到 (0, 0, 0)
    # 核心公式: p_new = p_old - center
    pcd.translate(-center)

    # 4. 验证新中心
    new_center = pcd.get_axis_aligned_bounding_box().get_center()
    print(f"归一化后中心坐标: {new_center}")

    # 5. 保存
    o3d.io.write_point_cloud(output_path, pcd)
    print(f"✅ 归一化完成，已保存至: {output_path}")

if __name__=='__main__':
    '''objs=['pear','apple','bowl']
    flags=['right','left','all']
    for obj in objs:
        for flag in flags:
            normalize_to_origin(f"outputs/{flag}_{obj}.ply", f"outputs/{flag}_{obj}.ply")'''
    normalize_to_origin("outputs/merged_apple_final_cleaned.ply","outputs/merged_apple_final_cleaned.ply")