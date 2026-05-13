import os
import shutil
import sys

import open3d as o3d

from pc3 import run_merge
from remove_outliers import remove_outliers_statistical, remove_outliers_radius
from mirror.zhongxin import normalize_to_origin
from buquan.mirror_demo import main as mirror_main
from buquan.complete_pointcloud_from_symmetric import symmetric
from buquan.complete_partial_to_mujoco import to_stl_from_completed_ply


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_all(obj: str):
    raw_path = run_merge(obj=obj, refine=True, visualize=False)

    raw_pcd = o3d.io.read_point_cloud(raw_path)
    if raw_pcd.is_empty():
        raise RuntimeError(f"融合点云为空: {raw_path}")

    cleaned_pcd, _ = remove_outliers_statistical(raw_pcd, nb_neighbors=20, std_ratio=2.0)
    cleaned_pcd, _ = remove_outliers_radius(cleaned_pcd, nb_points=16, radius=0.05)
    if cleaned_pcd.is_empty():
        raise RuntimeError("离群点去除后点云为空。")

    cleaned_path = os.path.join(ROOT_DIR, "outputs", f"merged_{obj}_final_cleaned.ply")
    os.makedirs(os.path.dirname(cleaned_path), exist_ok=True)
    o3d.io.write_point_cloud(cleaned_path, cleaned_pcd)
    print(f"✅ 清理后点云: {cleaned_path}")

    centered_path = os.path.join(ROOT_DIR, "outputs", f"merged_{obj}_final_centered.ply")
    normalize_to_origin(cleaned_path, centered_path)
    if not os.path.exists(centered_path):
        raise RuntimeError(f"中心化输出不存在: {centered_path}")

    buquan_input = os.path.join(ROOT_DIR, "buquan", f"pointcloud_{obj}_fused_refined.ply")
    shutil.copyfile(centered_path, buquan_input)
    print(f"✅ 传入补全点云: {buquan_input}")

    mirror_main(obj)
    symmetric(obj)

    completed_ply = os.path.join(ROOT_DIR, "buquan", f"{obj}_completed_pc_only.ply")
    report = to_stl_from_completed_ply(
        input_ply=completed_ply,
        object_name=f"{obj}",
        output_dir=os.path.join(ROOT_DIR, "manipulator_grasp", "assets", "fruit", "stl"),
    )
    
    stl_path = report["outputs"]["stl"]
    print(f"✅ 补全点云: {completed_ply}")
    print(f"✅ STL 导出: {stl_path}")


if __name__ == "__main__":
    obj = sys.argv[1] if len(sys.argv) > 1 else "apple"
    run_all(obj)
