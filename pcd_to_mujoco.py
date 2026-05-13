'''import argparse
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import open3d as o3d


@dataclass
class MeshBuildConfig:
    voxel_size: float = 0.005  # 增大体素尺寸，减少过度下采样
    normal_radius: float = 0.01
    normal_max_nn: int = 30
    poisson_depth: int = 9
    poisson_scale: float = 1.1
    poisson_linear_fit: bool = False
    crop_std_ratio: float = 2.0
    simplify_target_triangles: int = 5000
    remove_degenerate: bool = True
    translate_to_origin: bool = True
    scale_to_meters: float = 1.0


def load_point_cloud(path: str, visualize: bool = False) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        points = np.load(path)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(".npy point cloud must be Nx3")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        if visualize:
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
        return pcd
    if ext in {".ply", ".pcd"}:
        pcd = o3d.io.read_point_cloud(path)
        if pcd.is_empty():
            raise ValueError(f"Empty point cloud: {path}")
        print(f"PLY文件加载: {len(pcd.points)} 个点")
        print(f"是否有法向量: {pcd.has_normals()}")
        if pcd.has_colors():
            print(f"是否有颜色: {pcd.has_colors()}")
        if visualize:
            o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
        return pcd
    raise ValueError("Unsupported point cloud format. Use .npy/.ply/.pcd")


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, cfg: MeshBuildConfig) -> o3d.geometry.PointCloud:
    print(f"原始点云: {len(pcd.points)} 个点")
    
    if cfg.scale_to_meters != 1.0:
        pcd.scale(cfg.scale_to_meters, center=(0, 0, 0))

    # 智能下采样：根据点云大小调整体素尺寸
    original_points = len(pcd.points)
    if original_points > 100000:
        # 大点云：适度下采样
        voxel_size = 0.01
    elif original_points > 50000:
        # 中等点云：轻度下采样
        voxel_size = 0.008
    elif original_points > 10000:
        # 小点云：轻微下采样
        voxel_size = 0.005
    else:
        # 很小的点云：不下采样
        voxel_size = 0
    
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
        print(f"下采样后(体素={voxel_size}): {len(pcd.points)} 个点")

    # 智能离群点去除：根据点云大小调整参数
    if len(pcd.points) > 10000:
        nb_neighbors = 50
        std_ratio = 2.5
    elif len(pcd.points) > 1000:
        nb_neighbors = 30
        std_ratio = 2.0
    else:
        nb_neighbors = 20
        std_ratio = 3.0  # 更宽松的阈值
    
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"离群点去除后: {len(pcd.points)} 个点")
    
    # 检查点云是否为空或过少
    if len(pcd.points) < 10:
        raise ValueError(f"点云在预处理后点数过少({len(pcd.points)})，请检查原始数据")

    # 计算法向量（如果还没有的话）
    if not pcd.has_normals():
        print("计算法向量...")
        # 根据点云密度调整搜索参数
        if len(pcd.points) > 50000:
            radius = 0.02
            max_nn = 30
        elif len(pcd.points) > 10000:
            radius = 0.03
            max_nn = 50
        else:
            radius = 0.05
            max_nn = 100
        
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
        )
        pcd.normalize_normals()
        print(f"法向量计算完成: {pcd.has_normals()}")
    else:
        print("点云已有法向量，跳过计算")
    
    return pcd


def reconstruct_mesh_poisson(pcd: o3d.geometry.PointCloud, cfg: MeshBuildConfig) -> o3d.geometry.TriangleMesh:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, scale=cfg.poisson_scale, linear_fit=cfg.poisson_linear_fit
    )
    densities = np.asarray(densities)
    if densities.size > 0:
        density_threshold = np.quantile(densities, 0.05)
        keep = densities >= density_threshold
        mesh = mesh.select_by_index(np.where(keep)[0])
    return mesh


def postprocess_mesh(mesh: o3d.geometry.TriangleMesh, cfg: MeshBuildConfig) -> o3d.geometry.TriangleMesh:
    if cfg.remove_degenerate:
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

    if cfg.simplify_target_triangles > 0 and len(mesh.triangles) > cfg.simplify_target_triangles:
        mesh = mesh.simplify_quadric_decimation(cfg.simplify_target_triangles)

    mesh.compute_vertex_normals()

    if cfg.translate_to_origin:
        center = mesh.get_center()
        mesh.translate(-center)

    return mesh


def save_mesh(mesh: o3d.geometry.TriangleMesh, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not o3d.io.write_triangle_mesh(output_path, mesh):
        raise IOError(f"Failed to write mesh: {output_path}")


def build_mjcf(mesh_file: str, body_name: str, geom_density: float = 500.0) -> str:
    mesh_file = mesh_file.replace("\\", "/")
    mjcf = f"""<mujoco model=\"{body_name}\">
  <compiler angle=\"radian\" meshdir=\".\"/>
  <asset>
    <mesh name=\"{body_name}_mesh\" file=\"{os.path.basename(mesh_file)}\"/>
    <material name=\"{body_name}_mat\" rgba=\"0.8 0.8 0.8 1\"/>
  </asset>
  <worldbody>
  
  <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>
    <body name=\"{body_name}\" pos=\"0 0 0\">
      <freejoint/>
      <geom type=\"mesh\" mesh=\"{body_name}_mesh\" material=\"{body_name}_mat\" density=\"{geom_density}\"/>
    </body>
  </worldbody>
</mujoco>"""
    return mjcf


def write_mjcf(mjcf_text: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(mjcf_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert point cloud to MuJoCo mesh + MJCF")
    parser.add_argument("--pcd", required=True, help="Input point cloud (.npy/.ply/.pcd)")
    parser.add_argument("--out-mesh", default="outputs/object.stl", help="Output mesh path")
    parser.add_argument("--out-mjcf", default="outputs/object.xml", help="Output MJCF path")
    parser.add_argument("--name", default="object", help="Body/mesh name in MJCF")
    parser.add_argument("--scale", type=float, default=0.001, help="Scale to meters (e.g., 0.001 for mm)")
    parser.add_argument("--voxel", type=float, default=0.002, help="Voxel size in meters")
    parser.add_argument("--poisson-depth", type=int, default=9, help="Poisson reconstruction depth")
    parser.add_argument("--triangles", type=int, default=5000, help="Target triangle count for simplification")
    return parser.parse_args()


def main() -> None:
    obj='pear3'
    pcd =load_point_cloud(f"outputs/{obj}.ply",True)
    
    pcd = preprocess_point_cloud(pcd, MeshBuildConfig())
    
    # 确保法向量存在
    if not pcd.has_normals():
        print("警告: 预处理后仍无法向量，使用备用方法计算...")
        # 使用更保守的参数
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=100)
        )
        pcd.normalize_normals()
        
        if not pcd.has_normals():
            raise RuntimeError("无法计算点云法向量，请检查点云质量")
    
    print(f"最终点云: {len(pcd.points)} 个点，法向量: {pcd.has_normals()}")
    
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud")
    mesh = reconstruct_mesh_poisson(pcd, MeshBuildConfig())
    mesh = postprocess_mesh(mesh, MeshBuildConfig())
    save_mesh(mesh,f"outputs/{obj}.stl")
    mjcf_text = build_mjcf(f"outputs/{obj}.stl", obj)
    write_mjcf(mjcf_text, f"outputs/{obj}.xml")


if __name__ == "__main__":
    main()
'''

import argparse
import os
from dataclasses import dataclass
import numpy as np
import open3d as o3d

@dataclass
class MeshBuildConfig:
    voxel_size: float = 0.001       # 1mm 的体素下采样
    poisson_depth: int = 9          # 泊松重建深度
    poisson_scale: float = 1.1
    simplify_target_triangles: int = 5000
    scale_to_meters: float = 1.0    # 关键：根据你的数据，这里必须是 1.0
    translate_to_origin: bool = True

def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    print(f"正在加载点云: {path}")
    pcd = o3d.io.read_point_cloud(path)
    if pcd.is_empty():
        raise ValueError(f"无法读取点云或点云为空: {path}")
    return pcd

def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, cfg: MeshBuildConfig) -> o3d.geometry.PointCloud:
    print(f"原始点数: {len(pcd.points)}")
    
    # 1. 尺度处理
    if cfg.scale_to_meters != 1.0:
        pcd.scale(cfg.scale_to_meters, center=(0, 0, 0))
    
    # 打印包围盒，确认尺寸是否正确
    bbox = pcd.get_axis_aligned_bounding_box()
    print(f"点云当前包围盒尺寸: {bbox.get_extent()} (如果是米，0.1代表10cm)")

    # 2. 下采样
    if cfg.voxel_size > 0:
        pcd = pcd.voxel_down_sample(cfg.voxel_size)
        print(f"下采样后点数: {len(pcd.points)}")

    # 3. 核心：重估法向量 (解决 Found bad data 问题)
    print("正在估算法向量并统一朝向...")
    # 使用较大的搜索半径以获得平滑的法向量
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30)
    )
    
    # 统一法向量朝向：假设相机在上方或从中心向外定向
    # 这步对泊松重建是否能形成闭合曲面至关重要
    pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # 4. 去除离群点
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    return pcd

def reconstruct_mesh(pcd: o3d.geometry.PointCloud, cfg: MeshBuildConfig):
    '''print(f"正在进行泊松重建 (Depth={cfg.poisson_depth})...")
    
    # 执行重建
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=cfg.poisson_depth, scale=cfg.poisson_scale
    )
    
    # 清理低密度区域（移除重建产生的气泡/噪点）
    densities = np.asarray(densities)
    if densities.size > 0:
        density_threshold = np.quantile(densities, 0.1) # 剔除 10% 低密度部分
        mesh = mesh.select_by_index(np.where(densities > density_threshold)[0])

    return mesh'''
    print("正在使用滚球法重建...")
    # 自动计算半径（通常为体素大小的几倍）
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist
    
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd,
        o3d.utility.DoubleVector([radius, radius * 2])
    )
    return mesh
    '''print("正在尝试 Alpha Shapes 重建 (最适合薄壳结构)...")
    
    # Alpha 值决定了网格的紧密程度。值越小越精细，值越大越圆润
    # 建议尝试 0.01 到 0.05 之间的值
    alpha = 0.05
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    
    # 如果 Alpha Shapes 结果太破碎，回退到改进版滚球法
    if len(mesh.triangles) < 100:
        print("Alpha Shapes 失败，回退到增强版滚球法...")
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2, avg_dist * 4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
            
    return mesh'''

def postprocess_mesh(mesh: o3d.geometry.TriangleMesh, cfg: MeshBuildConfig) -> o3d.geometry.TriangleMesh:
    print("正在进行后期处理与简化...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=10)
    # 移除无效几何
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    # 简化面数
    if cfg.simplify_target_triangles > 0 and len(mesh.triangles) > cfg.simplify_target_triangles:
        mesh = mesh.simplify_quadric_decimation(cfg.simplify_target_triangles)

    # 居中处理
    if cfg.translate_to_origin:
        center = mesh.get_center()
        mesh.translate(-center)
        print(f"模型已平移至原点，原始中心: {center}")

    # 修改这里：
    print("计算法向量...")
    mesh.compute_vertex_normals()
    mesh.orient_triangles()
    # 如果你一定要确保面法向量存在，且上面那行没报错，
    # 可以尝试下面的通用调用方式（如果仍然报错，直接删掉这一行即可）：
    try:
        mesh.compute_face_normals()
    except:
        print("提示: 当前环境无需手动调用 compute_face_normals")
    
    return mesh

def build_mjcf(mesh_file: str, body_name: str) -> str:
    # 生成基础的 MuJoCo XML
    mjcf = f"""<mujoco model="{body_name}">
  <compiler angle="radian" meshdir="."/>
  <worldbody>
    <light directional="true" pos="-0.5 0.5 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" pos="0 0 0" rgba="0.8 0.8 0.8 1"/>
    <body name="{body_name}" pos="0 0 0.2">
      <freejoint/>
      <geom type="mesh" mesh="{body_name}_mesh" density="500" rgba="0.7 0.3 0.3 1"/>
    </body>
  </worldbody>
  <asset>
    <mesh name="{body_name}_mesh" file="{os.path.basename(mesh_file)}"/>
  </asset>
</mujoco>"""
    return mjcf

def main():
    # 参数配置
    obj_name = 'left_apple_mirror_combined_2_completed1'
    input_ply = f"outputs/{obj_name}.ply"
    output_stl = f"outputs/{obj_name}.stl"
    output_xml = f"outputs/{obj_name}.xml"

    # 初始化配置
    cfg = MeshBuildConfig()
    
    # 加载
    try:
        pcd = load_point_cloud(input_ply)
        
        # 预处理
        pcd = preprocess_point_cloud(pcd, cfg)
        
        # 重建
        mesh = reconstruct_mesh(pcd, cfg)
        
        # 后处理
        mesh = postprocess_mesh(mesh, cfg)
        
        # 保存 STL
        print(f"正在保存 STL: {output_stl}")
        o3d.io.write_triangle_mesh(output_stl, mesh)
        
        # 生成并保存 MJCF
        mjcf_text = build_mjcf(output_stl, obj_name)
        with open(output_xml, "w") as f:
            f.write(mjcf_text)
            
        print("\n" + "="*30)
        print(f"🎉 转换成功！")
        print(f"1. 模型文件: {output_stl}")
        print(f"2. 仿真配置文件: {output_xml}")
        print("="*30)
        
        # 最后可视化预览
        o3d.visualization.draw_geometries([mesh], window_name="Final Mesh Preview")

    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")

if __name__ == "__main__":
    main()