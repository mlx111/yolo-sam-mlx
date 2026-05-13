'''import cv2
import numpy as np
import os
from datetime import datetime
import open3d as o3d

class OrbbecMultiCameraClient:
    def __init__(self, save_dir=f'output/orbbec_multi_2'):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        
        # --- 相机内参 ---
        self.K_left = np.array([[1129.8136, 0., 961.0022], [0., 1128.6075, 546.8298], [0., 0., 1.]], dtype=np.float32)
        self.K_right = np.array([[1126.8856, 0., 954.9412], [0., 1126.4037, 536.3848], [0., 0., 1.]], dtype=np.float32)
        
        # 标定外参 (右相对于左)
        # 注意：这里的 flip_mat 逻辑如果和 Open3D 默认坐标系冲突会导致裂开
        self.flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        
        original_R_phys = np.array([
            [0.64348702, -0.24681324, 0.72457414],
            [0.17130231, 0.96901536, 0.17794592],
            [-0.74604288, 0.00961534, 0.66582848]
        ], dtype=np.float32)
        original_T_phys = np.array([[-890.21218409], [-121.98498635], [871.98401914]], dtype=np.float32)

        # 构造变换矩阵
        self.R_point = self.flip_mat @ original_R_phys @ self.flip_mat.T
        self.T_point = self.flip_mat @ (original_T_phys / 1000.0)

    def undistort_image(self, img, dev_idx):
        h, w = img.shape[:2]
        K = self.K_left if dev_idx == 0 else self.K_right
        # 这里假设畸变为0，如果有畸变请补充 D 参数
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, np.zeros(5), (w, h), 1, (w, h))
        return cv2.undistort(img, K, np.zeros(5), None, new_K), new_K.astype(np.float32)

    def depth_to_point_cloud(self, depth_img, color_img, dev_idx, mask_img=None):
        """生成点云并同步校正掩码"""
        undist_color, new_K = self.undistort_image(color_img, dev_idx)
        undist_depth, _ = self.undistort_image(depth_img, dev_idx)

        if mask_img is not None:
            # 掩码也需要畸变校正，否则边缘对不上
            K = self.K_left if dev_idx == 0 else self.K_right
            undist_mask = cv2.undistort(mask_img, K, np.zeros(5), None, new_K)
            if undist_mask.shape[:2] != undist_depth.shape[:2]:
                undist_mask = cv2.resize(undist_mask, (undist_depth.shape[1], undist_depth.shape[0]), interpolation=cv2.INTER_NEAREST)
            undist_depth = cv2.bitwise_and(undist_depth, undist_depth, mask=undist_mask)

        o3d_depth = o3d.geometry.Image(undist_depth.astype(np.uint16))
        o3d_color = o3d.geometry.Image(cv2.cvtColor(undist_color, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(undist_depth.shape[1], undist_depth.shape[0], new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        
        # 翻转坐标系
        pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return pcd

    def merge_point_clouds(self, pcd_left, pcd_right, refine=True):
        """合并点云并修复断层"""
        if pcd_left is None or pcd_right is None: return None, None, None

        # 初始外参变换矩阵
        initial_T = np.eye(4, dtype=np.float32)
        initial_T[:3, :3] = self.R_point
        initial_T[:3, 3] = self.T_point.flatten()

        # 应用初始变换（如果是裂开的，尝试把 initial_T 换成 np.linalg.inv(initial_T)）
        pcd_left_transformed = pcd_left.transform(initial_T)

        if refine:
            print("正在进行 ICP 精修...")
            pcd_left_transformed.estimate_normals()
            pcd_right.estimate_normals()
            # 针对裂开的情况，搜索半径放宽到 5cm
            reg = o3d.pipelines.registration.registration_icp(
                pcd_left_transformed, pcd_right, 0.05, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            pcd_left_transformed.transform(reg.transformation)

        merged_pcd = pcd_left_transformed + pcd_right
        return merged_pcd.voxel_down_sample(0.001), pcd_left_transformed, pcd_right

    def visualize(self, pcd, title="Point Cloud"):
        """修复 AttributeError 的方法名"""
        if pcd:
            o3d.visualization.draw_geometries([pcd], window_name=title, width=1280, height=720)

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    client = OrbbecMultiCameraClient()
    obj = "apple"
    
    # 模拟加载
    color_0 = cv2.imread("inputs/cleft001.png")
    depth_0 = cv2.imread("inputs/dleft001.png", -1)
    mask_0  = cv2.imread(f"inputs/left_mask_{obj}.png", 0)
    
    color_1 = cv2.imread("inputs/cright001.png")
    depth_1 = cv2.imread("inputs/dright001.png", -1)
    mask_1  = cv2.imread(f"inputs/right_mask_{obj}.png", 0)

    if color_0 is not None:
        pcd_l = client.depth_to_point_cloud(depth_0, color_0, 0, mask_img=mask_0)
        pcd_r = client.depth_to_point_cloud(depth_1, color_1, 1, mask_img=mask_1)

        # 尝试合并
        merged, _, _ = client.merge_point_clouds(pcd_l, pcd_r, refine=True)
        
        # 现在 client.visualize 可以正常工作了
        client.visualize(merged, f"Merged {obj}")'''

import cv2
import numpy as np
import os
import open3d as o3d

class OrbbecMultiCameraClient:
    def __init__(self):
        # --- 相机内参 (请保持你之前的参数) ---
        self.K_left = np.array([[1129.8136, 0., 961.0022], [0., 1128.6075, 546.8298], [0., 0., 1.]], dtype=np.float32)
        self.K_right = np.array([[1126.8856, 0., 954.9412], [0., 1126.4037, 536.3848], [0., 0., 1.]], dtype=np.float32)
        
        # 你的标定外参 (右相对于左)
        self.flip_mat = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float32)
        original_R_phys = np.array([[0.64348702, -0.24681324, 0.72457414], [0.17130231, 0.96901536, 0.17794592], [-0.74604288, 0.00961534, 0.66582848]], dtype=np.float32)
        original_T_phys = np.array([[-890.21218409], [-121.98498635], [871.98401914]], dtype=np.float32)

        self.R_point = self.flip_mat @ original_R_phys @ self.flip_mat.T
        self.T_point = self.flip_mat @ (original_T_phys / 1000.0)

    def depth_to_point_cloud(self, depth_img, color_img, dev_idx, mask_img=None):
        if depth_img is None or color_img is None: return None
        
        # 1. 畸变校正 (假设 D=0)
        K = self.K_left if dev_idx == 0 else self.K_right
        h, w = depth_img.shape[:2]
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, np.zeros(5), (w, h), 1, (w, h))
        undist_depth = cv2.undistort(depth_img, K, np.zeros(5), None, new_K)
        undist_color = cv2.undistort(color_img, K, np.zeros(5), None, new_K)

        # 2. 只有当掩码真实存在时才过滤
        if mask_img is not None and mask_img.size > 0:
            undist_mask = cv2.undistort(mask_img, K, np.zeros(5), None, new_K)
            if undist_mask.shape[:2] != undist_depth.shape[:2]:
                undist_mask = cv2.resize(undist_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            # 过滤深度
            undist_depth = cv2.bitwise_and(undist_depth, undist_depth, mask=undist_mask)
        else:
            print(f"⚠️ 警告: 相机 {dev_idx} 未检测到有效掩码，将生成全局点云")

        # 3. 生成 Open3D 点云
        o3d_depth = o3d.geometry.Image(undist_depth.astype(np.uint16))
        o3d_color = o3d.geometry.Image(cv2.cvtColor(undist_color, cv2.COLOR_BGR2RGB))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_color, o3d_depth, depth_scale=1000.0, convert_rgb_to_intensity=False)
        
        intrinsic = o3d.camera.PinholeCameraIntrinsic(w, h, new_K[0,0], new_K[1,1], new_K[0,2], new_K[1,2])
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
        pcd.transform([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        return pcd

    def merge_point_clouds(self, pcd_left, pcd_right, refine=True):
        if pcd_left is None or pcd_right is None: return None
        
        # 初始变换
        initial_T = np.eye(4)
        initial_T[:3, :3] = self.R_point
        initial_T[:3, 3] = self.T_point.flatten()

        # 尝试变换左点云
        pcd_l_trans = pcd_left.transform(initial_T)

        if refine and len(pcd_l_trans.points) > 10:
            pcd_l_trans.estimate_normals()
            pcd_right.estimate_normals()
            reg = o3d.pipelines.registration.registration_icp(
                pcd_l_trans, pcd_right, 0.03, np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint()
            )
            pcd_l_trans.transform(reg.transformation)

        return pcd_l_trans + pcd_right

    def visualize(self, pcd, title="Result"):
        if pcd and not pcd.is_empty():
            o3d.visualization.draw_geometries([pcd], window_name=title)
        else:
            print("❌ 错误: 点云为空，无法显示！请检查掩码文件是否读取成功。")


def run_merge(obj: str, refine: bool = True, save_path: str = None, visualize: bool = False) -> str:
    """
    读取双目图像并生成融合点云，返回保存文件的绝对路径。
    """
    if obj is None or str(obj).strip() == "":
        raise ValueError("obj 不能为空，例如 apple / bowl / pear。")

    client = OrbbecMultiCameraClient()

    m0_path = f"inputs/left_mask_{obj}.png"
    m1_path = f"inputs/right_mask_{obj}.png"
    m0 = cv2.imread(m0_path, 0)
    m1 = cv2.imread(m1_path, 0)
    if m0 is None:
        raise FileNotFoundError(f"找不到左相机掩码: {m0_path}")
    if m1 is None:
        raise FileNotFoundError(f"找不到右相机掩码: {m1_path}")

    c0 = cv2.imread("inputs/cleft001.png")
    d0 = cv2.imread("inputs/dleft001.png", -1)
    c1 = cv2.imread("inputs/cright001.png")
    d1 = cv2.imread("inputs/dright001.png", -1)
    if c0 is None or d0 is None or c1 is None or d1 is None:
        raise FileNotFoundError("输入图像缺失，请检查 inputs/cleft001.png, dleft001.png, cright001.png, dright001.png")

    p_l = client.depth_to_point_cloud(d0, c0, 0, mask_img=m0)
    p_r = client.depth_to_point_cloud(d1, c1, 1, mask_img=m1)
    merged = client.merge_point_clouds(p_l, p_r, refine=refine)
    if merged is None or merged.is_empty():
        raise RuntimeError("点云融合失败，结果为空。")

    if save_path is None:
        save_path = f"outputs/merged_{obj}_final.ply"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not o3d.io.write_point_cloud(save_path, merged):
        raise RuntimeError(f"点云保存失败: {save_path}")

    if visualize:
        client.visualize(merged, f"Merged {obj}")

    abs_path = os.path.abspath(save_path)
    print(f"✅ 点云已成功保存至: {abs_path}")
    return abs_path

# ---------------------- 主逻辑 ----------------------
if __name__ == "__main__":
    obj = "apple"
    run_merge(obj=obj, refine=True, visualize=True)
