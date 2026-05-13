import numpy as np
import cv2
import os
from math import pi
import threading
import open3d as o3d
import spatialmath as sm
from npy_ply import npy_ply
from camera_pose_mujoco import convert_raw_rotation_to_mujoco, rotation_matrix_from_euler_xyz_deg


CAMERA_EULER_DEG = {
    "left": [-17.08, -40.16, 96.59],
    "right": [-97.480497, -1.078239, -0.070489],
}

class PointCloudGenerator:
    """
    彩色图像与深度图像转点云的工具类
    使用相机相对于TCP的6D位姿（[x, y, z, rx, ry, rz]）替代变换矩阵
    仅可视化功能依赖open3d，其他核心逻辑完全基于numpy实现
    支持自动检测掩码图像路径，灵活生成局部/全局点云
    """

    def __init__(
            self,
            fx: float,
            fy: float,
            cx: float,
            cy: float,
            tcp_pose: list = None,
            camera_to_tcp_pose: list = None,  # 相机相对于TCP的6D位姿
            visualize: bool = False,
            save_point_cloud: bool = True,
            save_path: str = "point_cloud.npy"
    ):
        """
        类初始化：配置相机内参、机械臂参数、可视化与保存选项

        Args:
            fx, fy, cx, cy: 相机内参
            tcp_pose: 机械臂TCP在基坐标系下的6D位姿 [x, y, z, rx, ry, rz]
            camera_to_tcp_pose: 相机相对于TCP的6D位姿 [x, y, z, rx, ry, rz]
                               单位：平移(mm)，旋转(弧度)
        """
        # 相机内参（核心参数）
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # 机械臂TCP位姿（默认值）
        self.tcp_pose = tcp_pose if tcp_pose is not None else [
            0, 0, 450, 0 / 180 * pi, 60 / 180 * pi, 0 / 180 * pi
        ]
        if len(self.tcp_pose) != 6:
            raise ValueError("tcp_pose必须是长度为6的列表：[x, y, z, rx, ry, rz]")

        # 相机相对于TCP的6D位姿（默认值：相机在TCP前方100mm处，无旋转）
        self.camera_to_tcp_pose = camera_to_tcp_pose if camera_to_tcp_pose is not None else [
            0, 0, 0, -90 / 180 * pi, 0 / 180 * pi, -90 / 180 * pi  # x, y, z, rx, ry, rz
        ]
        if len(self.camera_to_tcp_pose) != 6:
            raise ValueError("camera_to_tcp_pose必须是长度为6的列表：[x, y, z, rx, ry, rz]")

        # 可视化与保存配置
        self.visualize = visualize
        self.save_point_cloud = save_point_cloud
        self.save_path = save_path

        # 可视化窗口相关（仅可视化时初始化）
        self.vis = None
        self.visualization_thread = None

        # 缓存生成的点云数据
        self.generated_point_cloud = None

    @staticmethod
    def rotation_matrix(r_x: float, r_y: float, r_z: float) -> np.ndarray:
        """生成绕X、Y、Z轴旋转的组合旋转矩阵（Z→Y→X顺序）"""
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(r_x), -np.sin(r_x)],
                       [0, np.sin(r_x), np.cos(r_x)]])
        Ry = np.array([[np.cos(r_y), 0, np.sin(r_y)],
                       [0, 1, 0],
                       [-np.sin(r_y), 0, np.cos(r_y)]])
        Rz = np.array([[np.cos(r_z), -np.sin(r_z), 0],
                       [np.sin(r_z), np.cos(r_z), 0],
                       [0, 0, 1]])
        
        mapping_matrix = np.array([
        [0, 0, 1],    # 现实z轴 → MuJoCo x轴
        [-1, 0, 0],   # 现实x轴 → MuJoCo -y轴
        [0, -1, 0]    # 现实y轴 → MuJoCo -z轴
        ])
        '''Rz=np.linalg.inv(Rz)
        Ry=np.linalg.inv(Ry)
        Rx=np.linalg.inv(Rx)'''
        R_camera_to_world = np.dot(Rz,np.dot(Ry,Rx))
        R_mujoco = mapping_matrix @ R_camera_to_world @ mapping_matrix.T
        #R_camera_to_world = np.linalg.inv(R_camera_to_world)
        return R_camera_to_world

    @staticmethod
    def pose_to_matrix(pose: list) -> np.ndarray:
        """将6D位姿 [x,y,z,rx,ry,rz] 转换为4x4变换矩阵"""
        if len(pose) != 6:
            raise ValueError("位姿必须是长度为6的列表：[x, y, z, rx, ry, rz]")
        x, y, z, rx, ry, rz = pose

        R = PointCloudGenerator.rotation_matrix(rx, ry, rz)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]

        return T

    def get_point_cloud_bounds(self, point_cloud: np.ndarray = None) -> dict:
        """获取点云X/Y/Z轴的最大值和最小值（纯numpy实现）"""
        target_pcd = point_cloud if point_cloud is not None else self.generated_point_cloud

        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        x_min, y_min, z_min = np.min(target_pcd, axis=0)
        x_max, y_max, z_max = np.max(target_pcd, axis=0)

        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max
        }

    def get_point_cloud_center(
            self,
            point_cloud: np.ndarray = None,
            use_bounds: bool = True,
            use_base_z: bool = False,
    ) -> np.ndarray:
        """计算点云的中心点坐标（纯numpy实现）"""
        target_pcd = point_cloud if point_cloud is not None else self.generated_point_cloud

        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        if use_bounds:
            bounds = self.get_point_cloud_bounds(target_pcd)
            x_center = (bounds["x_min"] + bounds["x_max"]) / 2
            y_center = (bounds["y_min"] + bounds["y_max"]) / 2
            if use_base_z:
                z_center = bounds["z_min"]
            else:
                z_center = (bounds["z_min"] + bounds["z_max"]) / 2
        else:
            x_center, y_center, z_center = np.mean(target_pcd, axis=0)

        return np.array([x_center, y_center, z_center])

    @staticmethod
    def downsample_image(
            color_img: np.ndarray,
            depth_img: np.ndarray,
            scale_percent: float = 1.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """对彩色图和深度图进行均匀下采样"""
        if scale_percent <= 0 or scale_percent > 1:
            raise ValueError("下采样比例必须在(0, 1]范围内")
        if scale_percent == 1.0:
            return color_img.copy(), depth_img.copy()

        interval = int(1 / np.sqrt(scale_percent))
        # 确保深度图是单通道
        if len(depth_img.shape) == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        resized_color = np.zeros_like(color_img)
        resized_depth = np.zeros_like(depth_img)

        for i in range(0, color_img.shape[0], interval):
            for j in range(0, color_img.shape[1], interval):
                resized_color[i, j] = color_img[i, j]
                resized_depth[i, j] = depth_img[i, j]

        return resized_color, resized_depth

    def process_images(
            self,
            color_img: np.ndarray,
            depth_img: np.ndarray,
            color_scale: float = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """图像预处理（彩色图缩放 + 深度图中心裁剪）"""
        if color_img is None or depth_img is None:
            raise ValueError("彩色图或深度图为空，请检查输入")
        if color_img.shape[:2] != depth_img.shape[:2]:
            raise ValueError("原始彩色图与深度图的尺寸必须一致（H×W）")

        # 确保深度图是单通道
        if len(depth_img.shape) == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

        h_orig, w_orig = color_img.shape[:2]
        new_w = int(w_orig * color_scale)
        new_h = int(h_orig * color_scale)
        resized_color = cv2.resize(color_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        start_x = (w_orig - new_w) // 2
        start_y = (h_orig - new_h) // 2
        cropped_depth = depth_img[start_y:start_y + new_h, start_x:start_x + new_w]

        return resized_color, cropped_depth

    def matrix_to_pose(self, matrix: np.ndarray) -> list:
        """将4x4变换矩阵转换为6D位姿 [x,y,z,rx,ry,rz]（纯numpy实现）"""
        if matrix.shape != (4, 4):
            raise ValueError("输入必须是4x4的变换矩阵")

        x, y, z = matrix[:3, 3]
        R = matrix[:3, :3]

        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6

        if not singular:
            rx = np.arctan2(R[2, 1], R[2, 2])
            ry = np.arctan2(-R[2, 0], sy)
            rz = np.arctan2(R[1, 0], R[0, 0])
        else:
            rx = np.arctan2(-R[1, 2], R[1, 1])
            ry = np.arctan2(-R[2, 0], sy)
            rz = 0.0

        return [x, y, z, rx, ry, rz]

    def calculate_camera_pose(self) -> tuple[list, np.ndarray]:
        """
        计算相机在基坐标系下的位姿（纯numpy实现）
        转换关系：相机在基坐标系位姿 = TCP在基坐标系位姿 × 相机相对于TCP的位姿
        """
        # 将TCP位姿转换为变换矩阵
        tcp_matrix = self.pose_to_matrix(self.tcp_pose)
        # 将相机相对于TCP的位姿转换为变换矩阵
        camera_tcp_matrix = self.pose_to_matrix(self.camera_to_tcp_pose)
        # 计算相机在基坐标系下的变换矩阵
        camera_matrix = np.dot(tcp_matrix, camera_tcp_matrix)
        # 转换为6D位姿
        camera_pose = self.matrix_to_pose(camera_matrix)

        return camera_pose, camera_matrix

    def apply_mask(
            self,
            color_img: np.ndarray,
            depth_img: np.ndarray,
            mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """应用掩码到彩色图和深度图"""
        # 确保深度图和掩码是单通道
        if len(depth_img.shape) == 3:
            depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if mask.shape[:2] != color_img.shape[:2]:
            mask = cv2.resize(mask, (color_img.shape[1], color_img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        mask = (mask > 0).astype(np.uint8)
        color_masked = color_img * np.stack([mask] * 3, axis=-1)
        depth_masked = depth_img * mask

        return color_masked, depth_masked

    def statistical_outlier_removal(pcd, radius=0.05, min_points=5):
        #cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        cl, ind = pcd.remove_radius_outlier(nb_points=min_points, radius=radius)

        pcd = pcd.select_by_index(ind)
        return pcd
    def _visualization_loop(self, points_world, valid_color, window_name):
        """可视化循环（仅此处依赖open3d，单独线程运行）"""
        try:
            # 仅在可视化时导入open3d，非可视化场景无需安装


            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(window_name=window_name)

            # 构建open3d点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            colors = valid_color / 255.0
            colors = colors[:, [2, 1, 0]]  # BGR转RGB
            pcd.colors = o3d.utility.Vector3dVector(colors)
            axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=200)
            cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=5)
            pcd = pcd.select_by_index(ind)
            '''cl, ind = pcd.remove_statistical_outlier(nb_neighbors=35, std_ratio=1)
            pcd = pcd.select_by_index(ind)'''
            #clean_pcd = self.statistical_outlier_removal(pcd)
            self.vis.add_geometry(pcd)
            self.vis.add_geometry(axis)
            self.vis.run()
            self.vis.destroy_window()
            self.vis = None
        except ImportError:
            print("警告：未安装open3d，无法进行可视化！请执行 `pip install open3d` 安装")
            self.vis = None
        except Exception as e:
            print(f"可视化线程出错: {str(e)}")
            if self.vis:
                self.vis.destroy_window()
                self.vis = None

    def _safe_visualize(self, points_world, valid_color, window_name):
        """安全的可视化方法（使用线程避免阻塞主进程）"""
        if self.visualization_thread and self.visualization_thread.is_alive():
            print("等待现有可视化窗口关闭...")
            self.visualization_thread.join(timeout=5.0)

            if self.visualization_thread.is_alive():
                print("强制终止现有可视化线程")
                if self.vis:
                    self.vis.destroy_window()
                self.visualization_thread = None

        self.visualization_thread = threading.Thread(
            target=self._visualization_loop,
            args=(points_world, valid_color, window_name),
            daemon=True
        )
        self.visualization_thread.start()

    @staticmethod
    def read_image_safely(path: str, is_depth: bool = False) -> np.ndarray:
        """
        安全读取图像的工具函数
        Args:
            path: 图像文件路径
            is_depth: 是否为深度图（深度图使用IMREAD_UNCHANGED读取）
        Returns:
            读取的图像数组
        Raises:
            FileNotFoundError: 文件不存在时抛出
            ValueError: 图像读取失败时抛出
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"图像文件不存在：{path}")
        if is_depth:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            # 强制将深度图转为单通道（解决JPG伪彩色深度图3通道问题）
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.imread(path)
        if img is None:
            raise ValueError(f"图像读取失败：{path}")
        return img
    def generate_point_cloud(
            self,
            color_image_ori: np.ndarray,
            depth_image_ori: np.ndarray,
            mask_path: str = None,  # 新增：掩码图像路径
            mask: np.ndarray = None,  # 保留：手动传入掩码数组
            use_mask_auto: bool = True,  # 新增：自动判断掩码路径是否存在
            downsample_scale: float = 1.0,
            image_alignment_scale: float = 1.0,
            objects = None,
            flag=None,
            type1=None
    ) -> dict:
        """
        核心方法：生成点云并转换到基坐标系（纯numpy实现，仅可视化依赖open3d）
        新增特性：
            1. 自动检测mask_path是否存在，存在则读取掩码生成局部点云
            2. use_mask_auto=False时，仅使用手动传入的mask数组（兼容原有逻辑）
        """
        if color_image_ori is None:
            return {"state": "fail", "info": "原始彩色图为空", "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}
        if depth_image_ori is None:
            return {"state": "fail", "info": "原始深度图为空", "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}

        # ===== 新增：自动处理掩码逻辑 =====
        final_mask = None
        processing_mode = "完整"

        # 自动模式：优先读取掩码路径
        if use_mask_auto and mask_path is not None and mask_path.strip() != "":
            try:
                # 安全读取掩码图像
                final_mask = self.read_image_safely(mask_path, is_depth=True)
                processing_mode = "带掩码区域（自动读取）"
                #print(f"成功读取掩码图像：{mask_path}")
            except FileNotFoundError:
                print(f"掩码路径不存在：{mask_path}，将生成全局点云")
            except ValueError as e:
                print(f"掩码图像读取失败：{e}，将生成全局点云")
        # 非自动模式：使用手动传入的mask
        elif not use_mask_auto and mask is not None:
            final_mask = mask
            processing_mode = "带掩码区域（手动传入）"

        try:
            # 1. 图像预处理（对齐）
            color_img, depth_img = self.process_images(
                color_image_ori, depth_image_ori, color_scale=image_alignment_scale
            )

            # 2. 应用掩码（如果有有效掩码）
            if final_mask is not None:
                color_img, depth_img = self.apply_mask(color_img, depth_img, final_mask)
                if not np.any(depth_img > 0):
                    return {"state": "fail", "info": "掩码区域内无有效深度数据",
                            "x": None, "y": None, "z": None,
                            "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}

            # 3. 图像下采样
            color_down, depth_down = self.downsample_image(
                color_img, depth_img, scale_percent=downsample_scale
            )

            # 4. 计算调整后的相机内参
            scale = downsample_scale * image_alignment_scale
            fx_f = self.fx * scale
            fy_f = self.fy * scale
            cx_f = self.cx * scale
            cy_f = self.cy * scale
            # 5. 生成相机坐标系下的点云（纯numpy实现）
            # 强制确保depth_down是单通道（最终防护）
            if len(depth_down.shape) == 3:
                depth_down = cv2.cvtColor(depth_down, cv2.COLOR_BGR2GRAY)
            valid_mask = depth_down > 0
            # 确保valid_mask是二维（防止极端情况）
            if len(valid_mask.shape) > 2:
                valid_mask = valid_mask[:, :, 0]
            if not np.any(valid_mask):
                return {"state": "fail", "info": f"{processing_mode}图像无有效深度数据",
                        "x": None, "y": None, "z": None,
                        "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0}
            valid_depth = depth_down[valid_mask]
            valid_color = color_down[valid_mask]
            # 修复核心解包问题：确保np.where返回二维索引
            coords = np.where(valid_mask)
            v, u = coords[0], coords[1]  # 显式取前两个维度
            # 像素坐标转相机坐标 (深度值从mm转换为m，与全局点云保持一致)
            valid_depth_meters = valid_depth / 1000.0  # mm → m
            x_cam = (u - cx_f) * valid_depth_meters / fx_f
            y_cam = (v - cy_f) * valid_depth_meters / fy_f
            z_cam = valid_depth_meters
            points_cam = np.stack((x_cam, y_cam, z_cam), axis=1)
            if type1=='normal':
                camera_to_tcp_matrix = self.pose_to_matrix(self.camera_to_tcp_pose)
                #print("旋转矩阵:",camera_to_tcp_matrix)
                R_cam_tcp = camera_to_tcp_matrix[:3, :3]
                T_cam_tcp = camera_to_tcp_matrix[:3, 3]
                points_tcp = np.dot(points_cam,R_cam_tcp.T) + T_cam_tcp
                # 6.2 TCP→基坐标系
                tcp_to_base_matrix = self.pose_to_matrix(self.tcp_pose)
                R_tcp_base = tcp_to_base_matrix[:3, :3]
                T_tcp_base = tcp_to_base_matrix[:3, 3]
                points_world = np.dot(points_tcp, R_tcp_base.T) + T_tcp_base
            elif type1=='merge':
                # 应用与全局点云相同的坐标系变换（X轴+Y轴翻转）
                flip_transform = np.array([[-1, 0, 0],
                                          [0, -1, 0],
                                          [0, 0, 1]])
                points_world = np.dot(points_cam, flip_transform.T)
                points_tcp = points_world  # 为merge模式设置points_tcp，避免可视化时出错
            # 7. 缓存生成的点云
            #point=points_cam@martix
            #self.generated_point_cloud = points_world
                        # 构建open3d点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world)
            '''cl, ind = pcd.remove_radius_outlier(nb_points=10, radius=4)
            pcd = pcd.select_by_index(ind)
            cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=1000.0)
            pcd = pcd.select_by_index(ind)
            
            labels = np.array(pcd.cluster_dbscan(eps=2, min_points=10))
            if len(labels[labels >= 0]) > 0:
                largest_cluster = labels == np.argmax(np.bincount(labels[labels >= 0]))
                pcd = pcd.select_by_index(np.where(largest_cluster)[0])'''

            points_world=np.asarray(pcd.points)
            self.generated_point_cloud=points_world
            # 8. 计算点云信息（纯numpy）
            bounds = self.get_point_cloud_bounds()
            center = self.get_point_cloud_center(use_base_z=(objects == 'roboticarm'))
            hx, hy, hz = center[0], center[1], center[2]
            x_min, y_min, z_min = bounds["x_min"], bounds["y_min"], bounds["z_min"]
            x_max, y_max, z_max = bounds["x_max"], bounds["y_max"], bounds["z_max"]

            # 9. 保存点云
            if self.save_point_cloud:
                if objects==None:
                    save_path=f"outputs/{flag}1_background.npy"
                    save_path2=f"outputs/{flag}1_background.ply"
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, points_world)
                    print(f"点云已保存至: {save_path}")
                    npy_ply(save_path,save_path2)
                elif objects !='roboticarm':
                    save_path=f"outputs/{flag}_{objects}.npy"
                    save_path2=f"outputs/{flag}_{objects}.ply"
                    save_dir = os.path.dirname(save_path)
                    if save_dir:
                        os.makedirs(save_dir, exist_ok=True)
                    np.save(save_path, points_world)
                    print(f"点云已保存至: {save_path}")
                    npy_ply(save_path,save_path2)
            # 10. 可视化点云（仅此处可能调用open3d）
            if self.visualize and len(points_world) > 0:
                self._safe_visualize(points_world, valid_color, f"{processing_mode}点云可视化")
                print("可视化窗口已打开（若未显示请检查open3d安装），关闭窗口后程序将继续执行...")
            # 11. 返回结果
            return {
                "state": "success",
                "info": f"{processing_mode}点云生成成功，包含{len(points_world)}个点",
                "x": hx, "y": hy, "z": hz,
                "x_min": x_min, "y_min": y_min, "z_min": z_min,
                "x_max": x_max, "y_max": y_max, "z_max": z_max,
                "point_count": len(points_world),
                "point_cloud_cam": points_cam,
                "point_cloud": points_world,
                "processing_mode": processing_mode  # 新增：返回处理模式
            }

        except Exception as e:
            import traceback
            self.generated_point_cloud = None
            if self.vis:
                self.vis.destroy_window()
                self.vis = None
            if self.visualization_thread and self.visualization_thread.is_alive():
                self.visualization_thread.join(timeout=2.0)
            # 打印详细错误栈，便于调试
            error_detail = traceback.format_exc()
            return {"state": "fail", "info": f"点云生成失败：{str(e)}\n详细错误：{error_detail}",
                    "x": None, "y": None, "z": None,
                    "x_min": 0, "y_min": 0, "z_min": 0, "x_max": 0, "y_max": 0, "z_max": 0,
                    "point_count": 0,
                    "processing_mode": processing_mode}

def point(flag,rx,ry,rz,objects=None):
    # 2. 相机相对于TCP的6D位姿（示例）
    camera_to_tcp_pose = [0,0,0, rx/ 180 * pi, ry/ 180 * pi, rz / 180 * pi]
    # 3. TCP在基坐标系中的6D位姿（示例）
    tcp_pose = [0, 0, 0, 0, 0, 0]
    # 4. 图像路径配置（核心修改：只需指定路径，自动判断）
    color_path = f"inputs/c{flag}001.png"
    depth_path = f"inputs/d{flag}001.png"
    if not objects:
        mask_path=None
    else:
        mask_path=f"inputs/{flag}_mask_{objects}.png"
    # 5. 读取彩色/深度图像
    try:
        color_img = PointCloudGenerator.read_image_safely(color_path, is_depth=False)
        depth_img = PointCloudGenerator.read_image_safely(depth_path, is_depth=True)
        print("彩色/深度图像读取成功！")
    except Exception as e:
        print(f"图像读取失败：{str(e)}")
        # 生成测试图像
        #print("生成测试图像以验证逻辑...")
        color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_img = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
    pcd_left_generator = PointCloudGenerator(
        fx=1129.8136,fy=1128.6075,cx=961.0022,cy=546.8298,
        tcp_pose=tcp_pose,
        camera_to_tcp_pose=camera_to_tcp_pose,
        visualize=False,
        save_point_cloud=True
    )
    pcd_right_generator = PointCloudGenerator(
        fx=1126.8856,fy=1126.4037,cx=954.9412,cy=536.3848,
        tcp_pose=tcp_pose,
        camera_to_tcp_pose=camera_to_tcp_pose,
        visualize=False,
        save_point_cloud=True
    )
    # 7. 生成点云（自动判断掩码路径）
    #print("\n=== 测试1：自动掩码模式 ===")
    pcd_generator=None
    if flag=='right':

        pcd_generator=pcd_right_generator
    else:
        pcd_generator=pcd_left_generator
    response = pcd_generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=mask_path,  # 只需传入路径，自动判断
        use_mask_auto=True,  # 开启自动判断
        downsample_scale=1.0,
        objects=objects,
        flag=flag,
        type1='normal'
    )
    ans=None
    #print("xmin,ymin,zmin,xmax,ymax,zmax：",response['x_min'],response['y_min'],response['z_min'],response['x_max'],response['y_max'],response['z_max'])
    # 8. 结果输出
    if response["state"] == "success":
        if flag=='right':

            ans=[-response['y'],response['x'],-response['z']]
        else:
            ans=[response['y'],response['z'],-response['x']]
        if objects and objects != 'roboticarm' and response.get("point_cloud_cam") is not None:
            raw_points = np.asarray(response["point_cloud_cam"], dtype=float)
            raw_path = f"outputs/raw_{flag}_{objects}.npy"
            raw_ply_path = f"outputs/raw_{flag}_{objects}.ply"
            np.save(raw_path, raw_points)
            npy_ply(raw_path, raw_ply_path)
    else:
        print(f"点云生成失败：{response['info']}")

    # 等待可视化线程结束
    if pcd_generator.visualization_thread:
        pcd_generator.visualization_thread.join()

    return ans
def point_merge(flag,objects=None):
    # 2. 相机相对于TCP的6D位姿（示例）
    camera_to_tcp_pose = [0,0,0, 0/ 180 * pi, 0/ 180 * pi, 0 / 180 * pi]
    # 3. TCP在基坐标系中的6D位姿（示例）
    tcp_pose = [0, 0, 0, 0, 0, 0]
    # 4. 图像路径配置（核心修改：只需指定路径，自动判断）
    color_path = f"inputs/c{flag}001.png"
    depth_path = f"inputs/d{flag}001.png"
    if not objects:
        mask_path=None
    else:
        mask_path=f"inputs/{flag}_mask_{objects}.png"
    # 5. 读取彩色/深度图像
    try:
        color_img = PointCloudGenerator.read_image_safely(color_path, is_depth=False)
        depth_img = PointCloudGenerator.read_image_safely(depth_path, is_depth=True)
        print("彩色/深度图像读取成功！")
    except Exception as e:
        print(f"图像读取失败：{str(e)}")
        # 生成测试图像
        #print("生成测试图像以验证逻辑...")
        color_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        depth_img = np.random.randint(0, 1000, (480, 640), dtype=np.uint16)
    pcd_left_generator = PointCloudGenerator(
        fx=1129.8136,fy=1128.6075,cx=961.0022,cy=546.8298,
        tcp_pose=tcp_pose,
        camera_to_tcp_pose=camera_to_tcp_pose,
        visualize=True,
        save_point_cloud=True
    )
    pcd_right_generator = PointCloudGenerator(
        fx=1126.8856,fy=1126.4037,cx=954.9412,cy=536.3848,
        tcp_pose=tcp_pose,
        camera_to_tcp_pose=camera_to_tcp_pose,
        visualize=False,
        save_point_cloud=True
    )
    # 7. 生成点云（自动判断掩码路径）
    #print("\n=== 测试1：自动掩码模式 ===")
    pcd_generator=None
    if flag=='right':
        pcd_generator=pcd_right_generator
    else:
        pcd_generator=pcd_left_generator
    response = pcd_generator.generate_point_cloud(
        color_image_ori=color_img,
        depth_image_ori=depth_img,
        mask_path=mask_path,  # 只需传入路径，自动判断
        use_mask_auto=True,  # 开启自动判断
        downsample_scale=1.0,
        objects=objects,
        flag=flag,
        type1='merge'
    )
    
    # 检查响应状态并安全获取点云
    if response["state"] != "success":
        print(f"点云生成失败：{response['info']}")
        return None
    
    pcd = response['point_cloud']
    #print("xmin,ymin,zmin,xmax,ymax,zmax：",response['x_min'],response['y_min'],response['z_min'],response['x_max'],response['y_max'],response['z_max'])
    # 8. 结果输出
    print("点云生成成功")

    # 等待可视化线程结束
    if pcd_generator.visualization_thread:
        pcd_generator.visualization_thread.join()

    return pcd
def pos(objects):
    row=['left','right']
    left=CAMERA_EULER_DEG["left"]
    right=CAMERA_EULER_DEG["right"]
    objects=['roboticarm']+objects
    ans=[]
    for i,r in enumerate(row):
        res=[]
        for j,o in enumerate(objects):
            if r=='left':
                pos=left
            else:
                pos=right
            res.append(point(r,pos[0],pos[1],pos[2],o))
        ans.append(res)
    
    #ans=point('right',-97.480497,-1.078239,-0.070489,"roboticarm")#右正好
    #ans=point('left',-17.08,-40.16,96.59,'purple_box') #左相机，z,y,x y,z相反
    #print(ans)
    m=len(ans)
    n=len(ans[0])
    for i in range(m):
        for j in range(1,n):
            for k in range(3):
                ans[i][j][k]=ans[i][j][k]-ans[i][0][k]
    #print(ans)
    #camera=[ans[0][0],ans[1][0]]
    pos=[]
    for j in range(1,n):
        temp=[]
        for k in range(3):
            temp1=(ans[0][j][k]+ans[1][j][k])/2
            #temp1=ans[1][j][k]
            temp.append(temp1)
        pos.append(temp)

    '''camera[0][0]=-ans[0][0][0]
    camera[1][0]=-ans[1][0][0]
    camera[0][1]=-ans[0][1][0]
    camera[1][1]=-ans[1][1][0]
    '''
    res={}
    for i in range(n-1):
        if pos[i][2]<0:
            pos[i][2]=0
        res[objects[i+1]]=pos[i]
    return res


def estimate_runtime_camera_poses():
    camera_poses = {}
    camera_names = {"left": "cam1", "right": "cam2"}

    for flag, camera_name in camera_names.items():
        rx, ry, rz = CAMERA_EULER_DEG[flag]
        roboticarm_pos = point(flag, rx, ry, rz, "roboticarm")
        if roboticarm_pos is None:
            continue

        mujoco_pose = convert_raw_rotation_to_mujoco(
            rotation_matrix_from_euler_xyz_deg(rx, ry, rz),
            flag,
        )
        camera_poses[camera_name] = {
            "pos": [
                -float(roboticarm_pos[0]),
                -float(roboticarm_pos[1]),
                float(roboticarm_pos[2]),
            ],
            "quat": mujoco_pose.quat_wxyz,
        }

    return camera_poses
# 测试代码
if __name__ == "__main__":
    '''result=pos(['bowl','pear','apple'])
    print(result)'''
    '''pcd_left=point_merge('left','apple')
    pcd_right=point_merge('right','apple')'''
    point('left',0,0,0,'pear')
