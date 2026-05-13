# file: rgbd_to_mujoco_mesh.py
import open3d as o3d
import numpy as np
import os

# ---------- 配置区 ----------
color_path = "color.png"       # 彩色图路径
depth_path = "depth.png"       # 深度图路径（单位：与下面 depth_scale 对应）
depth_scale = 1000.0           # 深度单位缩放（如果深度以毫米存储，scale=1000）
depth_trunc = 6.0              # 截断深度（米）
intrinsics = {
    "width": 640,
    "height": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 319.5,
    "cy": 239.5
}

out_mesh_obj = "mujoco_model.obj"
# -----------------------------

def load_rgbd(color_path, depth_path):
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)
    return rgbd

def build_intrinsic(intrinsics):
    return o3d.camera.PinholeCameraIntrinsic(
        intrinsics["width"], intrinsics["height"],
        intrinsics["fx"], intrinsics["fy"],
        intrinsics["cx"], intrinsics["cy"])

def rgbd_to_pcd(rgbd, intrinsic):
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd, intrinsic)
    # Open3D 默认相机坐标系：需要按你的坐标习惯转换
    # 例如将 z 轴朝向相机前方，y 向上：下面示例会把 y 轴朝上（视你数据格式而定）
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    return pcd

def preprocess_pcd(pcd, voxel_size=0.005):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down, ind = pcd_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(100)
    return pcd_down

def poisson_reconstruct(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit)
    densities = np.asarray(densities)
    # 删除低密度顶点（三选一策略：按密度阈值或按分位数）
    density_threshold = np.quantile(densities, 0.01)
    vertices_to_keep = densities > density_threshold
    # Crop mesh using vertex mask
    verts = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # keep only faces with all 3 vertices in the mask
    valid_tri_mask = vertices_to_keep[triangles].all(axis=1)
    new_triangles = triangles[valid_tri_mask]
    # build new mesh
    import trimesh as tm  # pip install trimesh
    tm_mesh = tm.Trimesh(vertices=verts, faces=new_triangles, process=False)
    # convert back to open3d
    mesh_o3d = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tm_mesh.vertices),
                                        o3d.utility.Vector3iVector(tm_mesh.faces))
    mesh_o3d.compute_vertex_normals()
    return mesh_o3d

def postprocess_mesh(mesh, target_triangles=20000):
    # 简化
    if np.asarray(mesh.triangles).shape[0] > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh

def main():
    rgbd = load_rgbd(color_path, depth_path)
    intrinsic = build_intrinsic(intrinsics)
    pcd = rgbd_to_pcd(rgbd, intrinsic)
    print("原始点数:", np.asarray(pcd.points).shape[0])
    pcd_clean = preprocess_pcd(pcd, voxel_size=0.004)
    print("预处理后点数:", np.asarray(pcd_clean.points).shape[0])
    mesh = poisson_reconstruct(pcd_clean, depth=9)
    mesh = postprocess_mesh(mesh, target_triangles=25000)
    print("三角面数:", np.asarray(mesh.triangles).shape[0])
    # 尝试把点云颜色投到顶点色（简单方式）
    # 将每个顶点的颜色设为最近邻点云颜色平均
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_clean)
    vert_colors = []
    for v in mesh.vertices:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(v, 3)
        colors = np.asarray(pcd_clean.colors)[idx]
        vert_colors.append(colors.mean(axis=0))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(vert_colors))
    # 导出 OBJ （包含顶点色）
    o3d.io.write_triangle_mesh(out_mesh_obj, mesh, write_vertex_colors=True)
    print("已导出:", out_mesh_obj)
    # 提示：若需要更精细纹理（UV/贴图），建议导入 Blender 做烘焙 UV
    print("完成。接下来在 MuJoCo XML 中添加 asset->mesh 和 geom->type='mesh' 即可使用。")

if __name__ == "__main__":
    main()
