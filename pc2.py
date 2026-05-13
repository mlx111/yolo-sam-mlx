import cv2
import numpy as np
import os
from datetime import datetime
import open3d as o3d
import matplotlib.pyplot as plt
from pointcloud_v2 import point_merge
# 解决matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class OrbbecMultiCameraClient:
    def __init__(self, save_dir=f'output/orbbec_multi_2'):
        """
        初始化Orbbec相机客户端（仅本地图片处理）
        :param save_dir: 本地保存根目录
        """
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"结果将保存至：{self.save_dir}")
        self.current_timestamp = None  # 存储当前时间戳

        # 左相机（device_0）彩色内参（1920x1080）
        self.K_left = np.array([
            [1129.8136, 0., 961.0022],
            [0., 1128.6075, 546.8298],
            [0., 0., 1.]
        ], dtype=np.float32)
        self.D_left = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)  # 畸变系数全0

        # 右相机（device_1）彩色内参（1920x1080）
        self.K_right = np.array([
            [1126.8856, 0., 954.9412],
            [0., 1126.4037, 536.3848],
            [0., 0., 1.]
        ], dtype=np.float32)
        self.D_right = np.array([[0., 0., 0., 0., 0.]], dtype=np.float32)  # 畸变系数全0

        # 点云坐标系翻转矩阵（X轴、Y轴翻转）
        self.flip_mat = np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ], dtype=np.float32)

        # 右相机相对于左相机的物理位姿（标定结果）
        original_R_phys = np.array([
            [0.64348702, -0.24681324, 0.72457414],
            [0.17130231, 0.96901536, 0.17794592],
            [-0.74604288, 0.00961534, 0.66582848]
        ], dtype=np.float32)
        original_T_phys = np.array([
            [-890.21218409],
            [-121.98498635],
            [871.98401914]
        ], dtype=np.float32)

        # 转换为「点云坐标系下」的位姿（匹配X/Y翻转后的点云）
        self.R_point = self.flip_mat @ original_R_phys @ self.flip_mat.T  # 旋转矩阵适配翻转
        self.T_point = self.flip_mat @ (original_T_phys / 1000.0)  # 平移向量转米 + 适配翻转

    def load_local_images(self, dev_idx, color_path, depth_path):
        """
        加载本地的彩色图和深度图
        :param dev_idx: 设备索引（0=左，1=右）
        :param color_path: 彩色图本地路径（png/jpg等）
        :param depth_path: 深度图本地路径（16位png，单位mm）
        :return: color_img, depth_img
        """
        try:
            # 加载彩色图（BGR格式）
            color_img = cv2.imread(color_path)
            if color_img is None:
                print(f"设备{dev_idx}彩色图加载失败：{color_path}")
                return None, None

            # 加载深度图（16位无符号，单位mm）
            depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_img is None:
                print(f"设备{dev_idx}深度图加载失败：{depth_path}")
                return None, None
            if depth_img.dtype != np.uint16:
                print(f"警告：设备{dev_idx}深度图不是16位数据，当前类型：{depth_img.dtype}")

            print(f"\n设备{dev_idx}本地图片加载成功：")
            print(f"  彩色图：{color_path} | 尺寸：{color_img.shape}")
            print(f"  深度图：{depth_path} | 尺寸：{depth_img.shape} | 类型：{depth_img.dtype}")

            return color_img, depth_img
        except Exception as e:
            print(f"设备{dev_idx}加载本地图片出错：{str(e)}")
            return None, None

    def undistort_image(self, img, dev_idx):
        """畸变校正（根据设备索引选择内参）"""
        h, w = img.shape[:2]
        # 获取对应相机的内参和畸变
        K = self.K_left if dev_idx == 0 else self.K_right
        D = self.D_left if dev_idx == 0 else self.D_right
        # 计算校正后的内参
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
        # 畸变校正
        undist_img = cv2.undistort(img, K, D, None, new_K)
        # 统一内参类型为float32，避免Open3D隐式转换问题
        new_K = new_K.astype(np.float32)
        return undist_img, new_K

    def depth_to_point_cloud(self, depth_img, color_img, dev_idx):
        """
        从深度图+彩色图生成点云（修复镜像问题：X轴+Y轴翻转）
        :param depth_img: 16位深度图（单位mm）
        :param color_img: 彩色图（BGR格式）
        :param dev_idx: 设备索引（0=左，1=右）
        :return: open3d点云对象
        """
        if depth_img is None or color_img is None:
            print(f"设备{dev_idx}图像为空，无法生成点云")
            return None

        # 畸变校正
        undist_color, new_K = self.undistort_image(color_img, dev_idx)
        undist_depth, _ = self.undistort_image(depth_img, dev_idx)

        # 转换为Open3D格式
        o3d_depth = o3d.geometry.Image(undist_depth.astype(np.uint16))
        o3d_color = o3d.geometry.Image(cv2.cvtColor(undist_color, cv2.COLOR_BGR2RGB))

        # 关键修正：depth_scale=1000（将mm转为m），depth_trunc=6.0（6米）
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d_color, o3d_depth,
            depth_scale=1000.0,  # 深度值/1000 → 米
            depth_trunc=6.0,  # 截断6米以外的深度（米单位）
            convert_rgb_to_intensity=False
        )

        # 生成点云
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=undist_depth.shape[1],
            height=undist_depth.shape[0],
            fx=new_K[0, 0],
            fy=new_K[1, 1],
            cx=new_K[0, 2],
            cy=new_K[1, 2]
        )

        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)

        # 核心：X轴+Y轴翻转（匹配真实场景方向）
        pcd.transform([[-1, 0, 0, 0],
                       [0, -1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return pcd

    def merge_point_clouds(self, pcd_left, pcd_right):
        """
        合并左右相机点云：将左点云变换到右相机坐标系后拼接
        :param pcd_left: 左相机点云
        :param pcd_right: 右相机点云
        :return: 合并后的点云、变换后的左点云、右点云
        """
        if pcd_left is None or pcd_right is None:
            print("左右相机点云为空，无法合并")
            return None, None, None

        # 如果传入的是numpy数组，转换为Open3D PointCloud对象
        if isinstance(pcd_left, np.ndarray):
            pcd_left_o3d = o3d.geometry.PointCloud()
            pcd_left_o3d.points = o3d.utility.Vector3dVector(pcd_left)
            pcd_left = pcd_left_o3d
        
        if isinstance(pcd_right, np.ndarray):
            pcd_right_o3d = o3d.geometry.PointCloud()
            pcd_right_o3d.points = o3d.utility.Vector3dVector(pcd_right)
            pcd_right = pcd_right_o3d

        # 构造点云坐标系下的变换矩阵（旋转+平移）
        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = self.R_point  # 匹配翻转后的旋转矩阵
        transform[:3, 3] = self.T_point.reshape(-1)  # 匹配翻转后的平移向量

        # 左点云变换到右相机坐标系
        pcd_left_transformed = o3d.geometry.PointCloud(pcd_left)
        pcd_left_transformed.transform(transform)

        # 拼接合并两个点云
        merged_pcd = o3d.geometry.PointCloud()
        merged_points = np.vstack([
            np.asarray(pcd_left_transformed.points),
            np.asarray(pcd_right.points)
        ])
        if pcd_left_transformed.has_colors() and pcd_right.has_colors():
            merged_colors = np.vstack([
                np.asarray(pcd_left_transformed.colors),
                np.asarray(pcd_right.colors)
            ])
            merged_pcd.colors = o3d.utility.Vector3dVector(merged_colors)

        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)

        # 去重+降噪（优化点云质量）
        merged_pcd.remove_duplicated_points()
        merged_pcd, _ = merged_pcd.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)

        print(
            f"点云合并完成：左相机{len(pcd_left_transformed.points)}点 + 右相机{len(pcd_right.points)}点 = 合并后{len(merged_pcd.points)}个有效点")
        return merged_pcd, pcd_left_transformed, pcd_right

    def evaluate_point_cloud_alignment(self, pcd_src, pcd_tgt, distance_threshold=0.005):
        """
        评估点云对齐精度（计算误差指标）
        :param pcd_src: 源点云（变换后的左点云）
        :param pcd_tgt: 目标点云（右点云）
        :param distance_threshold: 重合判定阈值（米，默认5mm）
        :return: 精度指标字典、误差数组
        """
        if pcd_src is None or pcd_tgt is None:
            raise ValueError("源/目标点云为空，无法评估精度")

        # 下采样减少计算量（体素大小3mm）
        pcd_src_down = pcd_src.voxel_down_sample(voxel_size=0.003)
        pcd_tgt_down = pcd_tgt.voxel_down_sample(voxel_size=0.003)

        # 构建KDTree
        kdtree = o3d.geometry.KDTreeFlann(pcd_tgt_down)

        # 存储最近邻距离（米）
        distances = []
        src_points = np.asarray(pcd_src_down.points)

        # 遍历源点云，找目标点云最近邻
        for point in src_points:
            [k, idx, dist] = kdtree.search_knn_vector_3d(point, 1)
            if k > 0:
                distances.append(np.sqrt(dist[0]))  # 距离（米）

        distances = np.array(distances)
        if len(distances) == 0:
            raise ValueError("无匹配的点对，无法计算误差")

        # 转换为毫米单位（更直观）
        distances_mm = distances * 1000
        threshold_mm = distance_threshold * 1000

        # 计算量化指标
        metrics = {
            "平均距离误差(mm)": np.mean(distances_mm),
            "均方根误差RMSE(mm)": np.sqrt(np.mean(distances_mm ** 2)),
            "最大误差(mm)": np.max(distances_mm),
            "最小误差(mm)": np.min(distances_mm),
            "重合率(≤{}mm)".format(threshold_mm): np.sum(distances_mm <= threshold_mm) / len(distances_mm) * 100,
            "有效匹配点数": len(distances_mm)
        }

        return metrics, distances_mm

    def remove_outliers(self, pcd, nb_neighbors=20, std_ratio=2.0):
        """使用统计学方法移除点云离群点"""
        print(f"移除离群点前点云数量: {len(pcd.points)}")
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_pcd = pcd.select_by_index(ind)
        outlier_pcd = pcd.select_by_index(ind, invert=True)
        print(f"移除离群点后点云数量: {len(inlier_pcd.points)}")
        print(f"移除离群点数量: {len(outlier_pcd.points)}")
        return inlier_pcd

    def visualize_single_point_cloud(self, pcd, dev_idx, window_name=None):
        """显示单个相机的点云"""
        if pcd is None:
            print(f"设备{dev_idx}点云为空，无法可视化")
            return

        # 如果传入的是numpy数组，转换为Open3D PointCloud对象
        if isinstance(pcd, np.ndarray):
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
            pcd = pcd_o3d

        window_name = window_name or f"Device {dev_idx} Point Cloud"
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)
        vis.add_geometry(pcd)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(coordinate)
        view_control = vis.get_view_control()
        view_control.set_front([0.5, 0.5, -0.5])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        print(f"\n{window_name} 可视化窗口已打开，按ESC关闭")
        vis.run()
        vis.destroy_window()

    def visualize_point_cloud(self, pcd, window_name="Merged Point Cloud"):
        """显示合并后的点云"""
        if pcd is None:
            print("合并点云为空，无法可视化")
            return

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1280, height=720)
        vis.add_geometry(pcd)
        coordinate = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2, origin=[0, 0, 0])
        vis.add_geometry(coordinate)
        view_control = vis.get_view_control()
        view_control.set_front([0.5, 0.5, -0.5])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, -1, 0])
        print("\n合并点云可视化窗口已打开，按ESC关闭")
        vis.run()
        vis.destroy_window()

    def show_images(self, frames_data, window_title="Orbbec 相机图像"):
        """显示彩色+深度可视化图"""
        if not frames_data:
            print("无法显示图像：输入帧数据为空")
            return

        display_imgs = []
        for dev_idx in sorted(frames_data.keys()):
            data = frames_data[dev_idx]
            color_img = data['color_img']
            depth_img = data['depth_img']

            if depth_img is None:
                print(f"设备{dev_idx}无有效深度图，跳过显示")
                continue

            # 深度图可视化处理
            valid_depth = depth_img[depth_img > 0]
            if len(valid_depth) > 0:
                depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_display = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
            else:
                depth_display = np.zeros((depth_img.shape[0], depth_img.shape[1], 3), dtype=np.uint8)

            # 合并彩色图和深度图
            if color_img is not None:
                target_h, target_w = depth_display.shape[:2]
                color_resized = cv2.resize(color_img, (target_w, target_h))
                dev_combined = np.hstack((color_resized, depth_display))
            else:
                dev_combined = depth_display

            cv2.putText(dev_combined, f"Device {dev_idx}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            display_imgs.append(dev_combined)

        if not display_imgs:
            print("无有效图像可显示")
            return

        # 拼接显示
        combined_img = np.vstack(display_imgs) if len(display_imgs) == 2 else display_imgs[0]
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_title, 1280, 720 if len(display_imgs) == 2 else 640)
        cv2.imshow(window_title, combined_img)
        print("\n提示：按 ESC 键关闭图像显示窗口")
        while True:
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyAllWindows()
                break


# ---------------------- 主程序 ----------------------
if __name__ == "__main__":
    # 配置本地图片路径（请替换为自己的路径）
    DEVICE_0_COLOR_PATH = "inputs/cleft001.png"
    #DEVICE_0_DEPTH_PATH = "inputs/vggt_depth_sds_left.png"
    DEVICE_1_COLOR_PATH = "inputs/cright001.png"
    #DEVICE_1_DEPTH_PATH = "inputs/vggt_depth_sds_right.png"
    DEVICE_0_DEPTH_PATH = "inputs/dleft001.png"
    DEVICE_1_DEPTH_PATH = "inputs/dright001.png"
    # 1. 初始化客户端
    client = OrbbecMultiCameraClient()
    client.current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    # 2. 加载本地图片
    print("=== 加载本地图片 ===")
    color_0, depth_0 = client.load_local_images(0, DEVICE_0_COLOR_PATH, DEVICE_0_DEPTH_PATH)
    color_1, depth_1 = client.load_local_images(1, DEVICE_1_COLOR_PATH, DEVICE_1_DEPTH_PATH)

    # 检查加载结果
    if color_0 is None or depth_0 is None or color_1 is None or depth_1 is None:
        print("本地图片加载失败，程序退出")
        exit(1)

    # 构造帧数据结构
    frames_data = {
        0: {"color_img": color_0, "depth_img": depth_0, "timestamp": client.current_timestamp},
        1: {"color_img": color_1, "depth_img": depth_1, "timestamp": client.current_timestamp}
    }

    # 3. 显示彩色+深度图
    client.show_images(frames_data)

    # 4. 生成左右相机点云
    print("\n=== 生成左右相机点云 ===")
    obj='all'
    pcd_left = client.depth_to_point_cloud(depth_0, color_0, 0)
    pcd_right = client.depth_to_point_cloud(depth_1, color_1, 1)
    #obj='pear'
    #pcd_left=point_merge('left',obj)
    #pcd_right=point_merge('right',obj)
    if pcd_left is None or pcd_right is None:
        print("点云生成失败，程序退出")
        exit(1)

    # 5. 显示单点云
    print("\n=== 显示左相机点云 ===")
    client.visualize_single_point_cloud(pcd_left, 0)
    print("\n=== 显示右相机点云 ===")
    client.visualize_single_point_cloud(pcd_right, 1)

    # 6. 合并点云
    print("\n=== 合并左右相机点云 ===")
    merged_pcd, pcd_left_t, pcd_right_t = client.merge_point_clouds(pcd_left, pcd_right)
    if merged_pcd is None:
        print("点云合并失败，程序退出")
        exit(1)

    # 7. 移除离群点
    print("\n=== 移除合并点云中的离群点 ===")
    merged_pcd = client.remove_outliers(merged_pcd)

    # 8. 评估点云对齐精度
    print("\n=== 评估点云对齐精度 ===")
    try:
        metrics, distances = client.evaluate_point_cloud_alignment(pcd_left_t, pcd_right_t, 0.005)
        print("\n===== 点云对齐精度指标 =====")
        for key, value in metrics.items():
            print(f"{key}: {value:.2f}%" if "率" in key else f"{key}: {value:.2f}")
    except Exception as e:
        print(f"精度评估失败：{str(e)}")

    # 9. 保存合并点云
    print("\n=== 保存合并点云 ===")
    # 保存为PLY格式

    ply_path = f"outputs/merged_{obj}_pointcloud.ply"
    os.makedirs(os.path.dirname(ply_path), exist_ok=True)
    success = o3d.io.write_point_cloud(ply_path, merged_pcd)
    if success:
        print(f"PLY文件已保存至: {ply_path}")
    else:
        print(f"PLY保存失败: {ply_path}")

    # 保存为NPY格式
    npy_path = f"outputs/merged_{obj}_pointcloud.npy"
    points_array = np.asarray(merged_pcd.points)

    # 如果有颜色信息，也保存颜色
    if merged_pcd.has_colors():
        colors_array = np.asarray(merged_pcd.colors)
        # 将点和颜色组合保存
        point_cloud_data = {
            'points': points_array,
            'colors': colors_array
        }
        np.save(npy_path, point_cloud_data)
        print(f"NPY文件已保存至: {npy_path} (包含点和颜色)")
    else:
        # 只保存点坐标
        np.save(npy_path, points_array)
        print(f"NPY文件已保存至: {npy_path} (仅点坐标)")

    print(f"点云数据: {len(points_array)} 个点")

    # 9. 显示合并点云
    print("\n=== 显示合并点云 ===")
    client.visualize_point_cloud(merged_pcd)

    print("\n程序执行完成！")