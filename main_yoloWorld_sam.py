import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

import cv2
import mujoco

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'graspnet-baseline', 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'manipulator_grasp'))
sys.path.append(os.path.join(ROOT_DIR, 'Grounded-SAM-2'))
sys.path.append(os.path.join(ROOT_DIR, 'anygrasp_sdk','grasp_detection'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from cv_proc import segment_image_ground
from get_grasp import getGrasp
from manipulator_grasp.arm.motion_planning import *
from manipulator_grasp.env.ur5_grasp_env import UR5GraspEnv
history_grasps={}
target=None
# 全局动作缓存，供预抓取等流程复用
action = np.zeros(7)
from cv_process import segment_image
# ================= 数据处理并生成输入 ====================
def get_and_process_data(color_path, depth_path, mask_path):
    """
    根据给定的 RGB 图、深度图、掩码图（可以是 文件路径 或 NumPy 数组），生成输入点云及其它必要数据
    """
#---------------------------------------
    # 1. 加载 color（可能是路径，也可能是数组）
    if isinstance(color_path, str):
        '''
        Image.open(color_path)：使用PIL库的Image模块打开指定路径的图像文件
np.array(..., dtype=np.float32)：将图像转换为NumPy数组，数据类型为32位浮点数
/ 255.0：将像素值从0-255的范围归一化到0-1的范围
这是深度学习中常见的预处理步骤，使数据更适合神经网络处理'''
        color = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    elif isinstance(color_path, np.ndarray):
        '''
        astype(np.float32)：将NumPy数组的数据类型转换为32位浮点数
这确保了数据格式的一致性，便于后续计算
第5行：color /= 255.0
同样执行归一化操作，将像素值范围从0-255转换到0-1
即使输入是数组，也需要进行相同的预处理'''
        color = color_path.astype(np.float32)
        color /= 255.0
    else:
        raise TypeError("color_path 既不是字符串路径也不是 NumPy 数组！")

    # 2. 加载 depth（可能是路径，也可能是数组）
    if isinstance(depth_path, str):
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img)
    elif isinstance(depth_path, np.ndarray):
        depth = depth_path
    else:
        raise TypeError("depth_path 既不是字符串路径也不是 NumPy 数组！")

    # 3. 加载 mask（可能是路径，也可能是数组）
    if isinstance(mask_path, str):
        workspace_mask = np.array(Image.open(mask_path))
    elif isinstance(mask_path, np.ndarray):
        workspace_mask = mask_path
    else:
        raise TypeError("mask_path 既不是字符串路径也不是 NumPy 数组！")

    # print("\n=== 尺寸验证 ===")
    # print("深度图尺寸:", depth.shape)
    # print("颜色图尺寸:", color.shape[:2])
    # print("工作空间尺寸:", workspace_mask.shape)

    # 构造相机内参矩阵
    '''
    相机内参矩阵是什么？
相机内参矩阵是一个3x3的矩阵，它包含了相机内部光学系统和图像传感器（如CCD或CMOS）特性对图像形成过程的影响参数。
简单来说，它描述了相机的内部属性如何将3D世界中的点投影到2D图像平面上。

为什么需要相机内参矩阵？
相机就像一个黑箱，3D世界中的物体通过这个黑箱，在图像传感器上形成了一个2D投影。相机内参矩阵就是这个黑箱的“说明书”，
它包含了相机内部所有影响这种投影的参数。理解这些参数对于许多计算机视觉任务至关重要，例如：'''

    '''
    从颜色图数组的形状属性中获取高度值
color.shape是一个元组，表示图像的维度，对于RGB图像通常是(height, width, channels)
shape[0]获取第一维的长度，即图像高度


焦距 (f_x, f_y):

f_x 和 f_y 分别是图像在水平和垂直方向上的焦距（通常以像素为单位）。
焦距衡量了镜头弯曲的程度。焦距越长，看到的景物范围越窄，图像越“放大”；焦距越短，看到的景物范围越宽，图像越“缩小”。
在理想的针孔相机模型中，f_x 和 f_y 通常是相等的（除非图像不是正方形像素）。

主点 (c_x, c_y):

(c_x, c_y) 是光学中心（或称为主点）在图像平面上的像素坐标。
在理想针孔模型中，光学中心是光线穿过透镜并投影到图像平面的点。主点通常位于图像的中心，
即 (c_x = 图像宽度/2, c_y = 图像高度/2)，但有时会因镜头设计、相机制造等原因而有所偏移。

相机内参矩阵的作用
相机内参矩阵 K 在计算机视觉中用于执行投影模型（或称相机模型）的数学运算。

基本投影模型 (针孔模型)
想象一个简化的相机模型：3D世界中的点 P = (X, Y, Z)，通过一个位于 (0, 0, f) 的针孔（光学中心），投影到与图像平面（位于 z=f 处）相交。

针孔模型的简化公式（忽略畸变）： x' = f * X / Z y' = f * Y / Z

其中 (x', y') 是3D点 P 在图像平面上的投影坐标（以光学中心为原点，单位是像素）。

'''
    height = color.shape[0]

    width = color.shape[1]
    '''
    定义垂直视场角(VFOV)为π/4弧度（45度）
这是一个简化的假设，实际相机的视场角可能不同
使用π/4可能是因为这是一个标准的测试场景或简化模型
'''
    fovy = np.pi / 4 # 定义的仿真相机
    focal = height / (2.0 * np.tan(fovy / 2.0))  # 焦距计算（基于垂直视场角fovy和高度height）
    c_x = width / 2.0   # 水平中心
    c_y = height / 2.0  # 垂直中心
    '''
    创建相机内参矩阵(intrinsic matrix)
这是一个3x3的矩阵，定义了相机的内部参数
标准形式：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
在理想情况下，fx=fy（焦距相等）'''
    intrinsic = np.array([
        [focal, 0.0, c_x],    
        [0.0, focal, c_y],   
        [0.0, 0.0, 1.0]
    ])
    factor_depth = 1.0  # 深度因子，根据实际数据调整

    # 利用深度图生成点云 (H,W,3) 并保留组织结构
    camera = CameraInfo(width, height, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    '''
    这段代码的核心是将二维深度图转换为三维点云，使用了相机成像模型的基本原理：

首先验证深度图尺寸与相机参数匹配
创建图像坐标网格
计算每个像素的深度值
使用相机内参将二维图像坐标转换为三维空间坐标
根据参数决定点云的组织方式
'''
    
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    '''
    workspace_mask > 0
这部分创建了一个布尔掩码，表示工作空间的有效区域：

workspace_mask 是一个与图像尺寸相同的二维数组
它通常包含0和1或0和255等值（具体取决于创建方式）
这个掩码通常由用户手动绘制或通过算法生成，标记出想要关注的区域
workspace_mask > 0 检查每个像素是否属于工作空间：
如果像素属于工作空间，返回 True
如果像素不属于工作空间，返回 False
'''
    # mask = depth < 2.0
    '''
    这部分创建了另一个布尔掩码，表示深度范围：

depth 是深度图，包含每个像素的深度值
2.0 是一个阈值，表示距离相机2米的距离
depth < 2.0 检查每个像素的深度是否小于2米：
如果像素深度小于2米，返回 True
如果像素深度大于或等于2米，返回 False

这段代码用于创建一个过滤条件，只保留那些位于工作空间内且距离相机较近的点。这在3D重建和点云处理中有几个重要作用：

空间限制：只关注预定义的工作空间区域，排除背景和其他无关区域
深度限制：只保留较近的点，排除远处的点
点云稀疏化：减少点云中的无效点，提高后续处理效率
质量控制：通常较近的点具有更高的深度精度


'''
    mask = (workspace_mask > 0) & (depth < 2.0)
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    # print(f"mask过滤后的点云数量 (color_masked): {len(color_masked)}") # 在采样前打印原始过滤后的点数

    NUM_POINT = 5000 # 10000或5000
    # 如果点数足够，随机采样NUM_POINT个点（不重复）
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    # 如果点数不足，先保留所有点，再随机重复补足NUM_POINT个点
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), NUM_POINT - len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs] # 提取点云和颜色
    '''
    创建一个空的Open3D点云对象
o3d.geometry.PointCloud()是Open3D库中定义点云数据结构的标准方式
这类似于创建一个容器，用于存储三维点云数据
'''
    cloud_o3d = o3d.geometry.PointCloud()
    '''
    设置点云的点坐标数据
o3d.utility.Vector3dVector是Open3D的专用数据结构，用于存储三维点坐标
cloud_masked.astype(np.float32)将过滤后的点云数据转换为32位浮点数格式
.astype(np.float32)确保数据类型一致性，提高精度同时减少内存占用'''
    cloud_o3d.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud_o3d.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #将数据移动到gpu上
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32)).to(device)
    end_points = {'point_clouds': cloud_sampled}
    ''' n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.05, 0 ,1.2])
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam1")
    t_wc = data.cam_xpos[cam_id].copy()
    R_wc = data.cam_xmat[cam_id].reshape(3, 3).copy()
    T_wc = sm.SE3.Rt(sm.SO3(R_wc), t_wc)
    #T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    R_wc = T_wc.R       # 3x3
    t_wc = T_wc.t       # 3x1
    points_world = (R_wc @ cloud_masked.T).T + t_wc.reshape(1, 3)
    print('中心点:',get_point_cloud_center(points_world))'''
    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d



# =================== 获取抓取预测 ====================
def generate_grasps(end_points, cloud, visual=False):
    """
    主推理流程：
    0. 数据处理并生成输入
    1. 加载网络
    2. 前向推理（进行抓取预测解码）
    3. 碰撞检测
    4. NMS 去重 + 按置信度/得分排序（降序）
    5. 对抓取预测进行垂直角度筛选
    """

    # 1. 加载网络
    net = GraspNet(input_feature_dim=0, 
                   num_view=300, 
                   num_angle=12, 
                   num_depth=4,
                   cylinder_radius=0.05, 
                   hmin=-0.02, 
                   hmax_list=[0.01, 0.02, 0.03, 0.04], 
                   is_training=False)
    net.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    checkpoint = torch.load('./logs/log_rs/checkpoint-rs.tar') # checkpoint_path
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    # 2. 前向推理
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg = GraspGroup(grasp_preds[0].detach().cpu().numpy()) 

    # 3. 碰撞检测
    '''
    这段代码实现了一个基于体素的无模型碰撞检测功能，主要用于过滤掉与已有点云发生碰撞的点。让我们逐行解析：

第一行：COLLISION_THRESH = 0.01
定义了一个碰撞阈值常量，值为0.01米
这个阈值用于判断两点之间是否发生碰撞
较小的值意味着更精确的碰撞检测，但计算量更大
    '''
    COLLISION_THRESH = 0.005
    if COLLISION_THRESH > 0:
        '''
        设置体素大小为0.01米
体素是三维空间中的小立方体，用于简化碰撞检测
较小的体素大小意味着更精确的检测，但计算量更大
第四行：collision_thresh = 0.01
设置碰撞检测阈值为0.01米
这是检测两点之间是否发生碰撞的临界距离
当两点距离小于这个阈值时，认为发生碰撞
        '''
        voxel_size = 0.005
        collision_thresh = 0.005
        '''
        创建一个无模型碰撞检测器对象
np.asarray(cloud.points)将点云数据转换为NumPy数组
voxel_size参数指定体素大小
这个检测器使用体素网格方法进行碰撞检测
        '''
        mfcdetector = ModelFreeCollisionDetector(np.asarray(cloud.points), voxel_size=voxel_size)
        '''
        执行碰撞检测
gg是要检测的点云
approach_dist=0.05是接近距离，用于判断点是否接近障碍物
collision_thresh是碰撞阈值
返回一个掩码数组，标记哪些点发生碰撞
        '''
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=collision_thresh)
        gg = gg[~collision_mask]

    # 4. NMS 去重 + 按置信度/得分排序（降序）
    '''
    NMS是"Non-Maximum Suppression"（非极大值抑制）的缩写，是一种在目标检测中常用的技术，用于去除冗余的检测结果。它通过比较检测框的置信度分数，保留最高分的检测框，同时抑制与其高度重叠的其他检测框。

NMS去重的工作原理
检测候选框：首先使用目标检测算法（如YOLO、SSD等）生成多个候选框
排序：按置信度分数对候选框进行排序
抑制：从置信度最高的开始，抑制与其重叠度超过阈值的所有其他候选框
    '''
    gg.nms().sort_by_score()

    ''# 5. 返回抓取得分最高的抓取（对抓取预测的接近方向进行垂直角度限制）
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面） np.array([0, 0, 1])
    angle_threshold = np.deg2rad(10)  # 30度的弧度值 np.deg2rad(30)
    filtered = []
    for grasp in all_grasps:
        # 抓取的接近方向取 grasp.rotation_matrix 的第三列[:, 0]
        '''
        从抓取位姿的旋转矩阵中提取接近方向
grasp.rotation_matrix是一个3x3的旋转矩阵，表示抓取工具的朝向
[:, 0]表示提取旋转矩阵的第一列，这代表抓取工具的x轴方向
在机器人抓取中，第一列通常表示工具的x轴方向

        '''
        approach_dir = grasp.rotation_matrix[:, 0]
        '''
        计算抓取方向与期望方向之间的点积
点积公式：cosθ = A·B = |A||B|cosθ
这里|A|和|B|都是单位向量，所以cosθ = A·B
点积结果范围在[-1, 1]之间，1表示两个方向完全相同，-1表示完全相反
        '''
        # 计算夹角：cos(angle)=dot(approach_dir, vertical)
        cos_angle = np.dot(approach_dir, vertical)
        '''
        将点积结果限制在[-1, 1]范围内
这是因为浮点数计算可能导致点积结果略微超出[-1, 1]范围
使用np.clip确保输入值在有效范围内'''
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        '''
        计算两个方向之间的夹角（以弧度为单位）
arccos是反余弦函数，将cosθ值转换为角度值
结果范围在0到π弧度之间（0到180度）
        '''
        angle = np.arccos(cos_angle)
        if angle < angle_threshold:
            filtered.append(grasp)
    if len(filtered) == 0:
        print("\n[Warning] No grasp predictions within vertical angle threshold. Using all predictions.")
        filtered = all_grasps
    # else:
        print(f"\nFiltered {len(filtered)} grasps within ±30° of vertical out of {len(all_grasps)} total predictions.")


    # 对过滤后的抓取根据 score 排序（降序）
    filtered.sort(key=lambda g: g.score, reverse=True)

    # 取前20个抓取（如果少于20个，则全部使用）
    top_grasps = filtered[:20]
    # top_grasps = filtered[:1]

    # 可视化过滤后的抓取，手动转换为 Open3D 物体
    grippers = [g.to_open3d_geometry() for g in top_grasps]
    # print(f"\nVisualizing top {len(top_grasps)} grasps after vertical filtering...")
    # o3d.visualization.draw_geometries([cloud, *grippers])
    # for gripper in grippers:
    #     o3d.visualization.draw_geometries([cloud, gripper])
    
    # 选择得分最高的抓取（filtered 列表已按得分降序排序）
    best_grasp = top_grasps[0]
    global history_grasps
    if history_grasps.get(target)==None:
        history_grasps[target]=[best_grasp]
    else:
        history_grasps[target].append(best_grasp)
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取''
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    return new_gg
    # return best_translation, best_rotation, best_width


# ================= 仿真执行抓取动作 ====================
def execute_grasp(env, gg):
    """
    执行抓取动作，控制机器人从初始位置移动到抓取位置，并完成抓取操作。

    参数:
    env (UR5GraspEnv): 机器人环境对象。
    gg (GraspGroup): 抓取预测结果。
    """
    robot = env.robot
    T_wb = robot.base
    data=env.mj_data
    model=env.mj_model
    # 1. 获取相机在世界坐标系下的位置和旋转矩阵
    # cam_xpos: 相机中心的世界坐标
    # cam_xmat: 相机的旋转矩阵 (3x3)
    '''cam_id = model.camera('ee_camera').id
    t_wc_raw = data.cam_xpos[cam_id]
    R_wc_raw = data.cam_xmat[cam_id].reshape(3, 3)
    # 2. 坐标系对齐 (MuJoCo -> AnyGrasp)
    # 转换为 SE3 格式
    T_wc_raw = sm.SE3.Rt(sm.SO3(R_wc_raw), t_wc_raw)
    # 绕 X 轴旋转 180 度，使 Z 轴向前，Y 轴向下
    T_align = sm.SE3.Rx(np.pi)
    T_wc = T_wc_raw * T_align'''
    # 0.初始准备阶段
    # 目标：计算抓取位姿 T_wo（物体相对于世界坐标系的位姿）
    # n_wc = np.array([0.0, -1.0, 0.0]) # 相机朝向
    # o_wc = np.array([-1.0, 0.0, -0.5]) # 相机朝向 [0.5, 0.0, -1.0] -> [-1.0, 0.0, -0.5]
    #t_wc = np.array([1.0, 0.6, 2.0]) # 相机的位置。2.0是相机高度，与scene.xml中保持一致。
    #n_wc = np.array([0.0, -1.0, 0.0]) 
    #o_wc = np.array([-1.0, 0.0, -0.5]) 
    #t_wc = np.array([0.05, 0 ,1.2]) 
    '''n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.05, 0 ,1.2])'''
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
    t_wc_raw = data.cam_xpos[cam_id].copy()
    R_wc_raw = data.cam_xmat[cam_id].reshape(3, 3).copy()

    T_wc_raw = sm.SE3.Rt(sm.SO3(R_wc_raw), t_wc_raw)    
    T_align = sm.SE3.Rx(np.pi)
    T_wc = T_wc_raw * T_align
    #1.6 1.3 0 1 0.8 0.5
    #t_wc = np.array([0.85 ,0.8, 1.6]) 
    '''
    sm.SE3.Trans(t_wc)：创建一个平移变换，将坐标系原点平移到t_wc位置
sm.SO3.TwoVectors(x=n_wc, y=o_wc)：使用两个向量创建旋转部分
这创建了一个旋转，使得新坐标系的x轴和y轴分别与n_wc和o_wc平行
两个变换相乘：T_wc = 平移变换 * 旋转变换
这表示先进行旋转，再进行平移
'''
    
    '''
    gg.translations[0]：从某个对象gg中获取第一个平移向量
gg.rotation_matrices[0][:, 0]和gg.rotation_matrices[0][:, 1]：从旋转矩阵中提取前两列，作为两个基向量
创建变换：先进行旋转，再进行平移
    '''
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    '''
    将两个变换相乘：T_wo = T_wc * T_co
这表示从坐标系w到坐标系o的变换路径：w->c->o
    '''
    T_wo = T_wc * T_co
    print("T_wo:",T_wo)
    action = np.zeros(7)

    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    '''
    这部分代码设置机器人运动的基本参数
q0和q1定义了机器人运动的起点和终点
JointParameter、QuinticVelocityParameter和TrajectoryParameter类定义了运动的约束条件
TrajectoryPlanner类实现了五次多项式轨迹规划算法
    '''
    time1 = 1
    q0 = robot.get_joint()
    #预抓取位置
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    parameter0 = JointParameter(q0, q1)#关节参赛
    velocity_parameter0 = QuinticVelocityParameter(time1)# 五次多项式速度参数
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)# 轨迹参数
    planner1 = TrajectoryPlanner(trajectory_parameter0) # 轨迹规划器
    # 执行planner_array = [planner1]

    '''
    这部分代码实现了时间插值和运动执行的核心逻辑
time_array和time_cumsum用于管理多段运动的总时间
对于每个时间点，计算对应的机器人状态
根据计算结果执行关节运动或笛卡尔空间运动
将机器人状态更新到动作数组并传递给环境
    '''
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    '''
    total_time：机器人运动的总时间（秒）
0.002：时间步长（秒）
round(...)：四舍五入取整
+1：确保覆盖整个时间范围
这个计算确定了需要生成多少个时间点，以便精确控制机器人的运动。
    '''
    time_step_num = round(total_time / 0.002) + 1
    '''
    np.linspace：生成等间隔的数值
0.0：起始时间
total_time：结束时间
time_step_num：生成的时间点数量
    '''
    times = np.linspace(0.0, total_time, time_step_num)
    '''
    np.cumsum：计算累积和
例如，如果time_array = [t1, t2, t3]，那么time_cumsum = [t1, t1+t2, t1+t2+t3]。
    '''
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        '''
        time_cumsum 是累积时间数组，记录每个时间段结束时的总时间
对于当前时间点timei，代码会找到它所属的时间段索引j
如果timei是0.0，直接跳过（可能用于初始化）
如果timei小于等于某个时间段的累积时间，确定当前时间段
        '''
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                '''
                planner_array[j - 1]：获取当前时间段对应的轨迹规划器
timei - time_cumsum[j - 1]：计算当前时间点距离该时间段开始的时间差
interpolate() 方法根据时间差计算机器人在该时刻的状态
                '''
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                ''''
                如果插值结果是数组，表示机器人在关节空间状态，执行move_joint操作
如果插值结果不是数组，表示机器人在笛卡尔空间状态，执行move_cartesian操作
无论哪种情况，都会获取当前关节状态并更新动作数组
                '''
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                    '''
                    将计算出的关节状态更新到动作数组的前6维
执行环境步骤，将动作发送给机器人
使用break跳出循环，继续处理下一个时间点

                    '''
                action[:6] = joint
                env.step(action)
                break

    # 2.接近抓取位姿
    # 目标：从预抓取位姿直线移动到抓取点附近（T2）
    # 关键点：T2 是 T_wo 沿负 x 方向偏移 0.1m，确保安全接近物体。
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0,0.0)
    print("T_pregrasp(T2):", T2)
    position_parameter1 = LinePositionParameter(T1.t, T2.t) #  位置规划（直线路径）
    attitude_parameter1 = OneAttitudeParameter(sm.SO3(T1.R), sm.SO3(T2.R)) # 姿态规划（插值旋转）
    cartesian_parameter1 = CartesianParameter(position_parameter1, attitude_parameter1) # 组合笛卡尔参数
    velocity_parameter1 = QuinticVelocityParameter(time2) # 速度曲线（五次多项式插值）
    trajectory_parameter1 = TrajectoryParameter(cartesian_parameter1, velocity_parameter1) # 将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    planner2 = TrajectoryPlanner(trajectory_parameter1) # 轨迹规划器，将笛卡尔空间路径和速度曲线结合，生成完整的轨迹参数
    # 执行planner_array = [planner2]
    time_array = [0.0, time2]
    planner_array = [planner2]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 3.执行抓取
    # 目标：从 T2 移动到 T3（精确抓取位姿）。通过逐步增加 action[-1]（夹爪控制信号）闭合夹爪，抓取物体。
    time3 = 1
    T3 = T_wo
    print("T_grasp(T3):", T3)
    position_parameter2 = LinePositionParameter(T2.t, T3.t)
    attitude_parameter2 = OneAttitudeParameter(sm.SO3(T2.R), sm.SO3(T3.R))
    cartesian_parameter2 = CartesianParameter(position_parameter2, attitude_parameter2)
    velocity_parameter2 = QuinticVelocityParameter(time3)
    trajectory_parameter2 = TrajectoryParameter(cartesian_parameter2, velocity_parameter2)
    planner3 = TrajectoryPlanner(trajectory_parameter2)
    # 执行planner_array = [planner3]
    time_array = [0.0, time3]
    planner_array = [planner3]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num) 
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)): 
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    #夹爪闭合
    for i in range(1000):
        action[-1] += 0.2
        '''
        使用np.min()函数确保动作值不超过255
这是一个安全边界，防止动作值超出环境可接受范围
        '''
        action[-1] = np.min([action[-1], 255])
        env.step(action)

    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)

    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    '''
    创建一个平移变换矩阵，将坐标系平移(1.6, 0.3, z)
T4.t[2]表示从T4变换矩阵中获取z坐标值
这意味着新变换的x和y平移分量是固定值，而z分量与T4保持一致
从T4的旋转矩阵R中提取旋转部分
sm.SO3是旋转矩阵的包装类
这确保了新变换保留了T4原有的旋转特性
    '''
    #T5 = sm.SE3.Trans(1.4, 0.6, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
    T5 = sm.SE3.Trans(0.3, 0.3, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) 
    position_parameter4 = LinePositionParameter(T4.t, T5.t)
    attitude_parameter4 = OneAttitudeParameter(sm.SO3(T4.R), sm.SO3(T5.R))
    cartesian_parameter4 = CartesianParameter(position_parameter4, attitude_parameter4)
    velocity_parameter4 = QuinticVelocityParameter(time5)
    trajectory_parameter4 = TrajectoryParameter(cartesian_parameter4, velocity_parameter4)
    planner5 = TrajectoryPlanner(trajectory_parameter4)

    # 6.放置物体
    # 目标：垂直下降物体到接触面（T7）。逐步减小 action[-1]（夹爪信号）以释放物体。
    time6 = 1
    T6 = sm.SE3.Trans(0.0, 0.0, -0.1) * T5 # 通过在T5的基础上向下偏移0.1单位得到的，用于控制机器人下降一定的高度
    position_parameter6 = LinePositionParameter(T5.t, T6.t)
    attitude_parameter6 = OneAttitudeParameter(sm.SO3(T5.R), sm.SO3(T6.R))
    cartesian_parameter6 = CartesianParameter(position_parameter6, attitude_parameter6)
    velocity_parameter6 = QuinticVelocityParameter(time6)
    trajectory_parameter6 = TrajectoryParameter(cartesian_parameter6, velocity_parameter6)
    planner6 = TrajectoryPlanner(trajectory_parameter6)

    # 执行planner_array = [planner4, planner5, planner6]
    time_array = [0.0, time4, time5, time6]
    planner_array = [planner4, planner5, planner6]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
    
    for i in range(1000):
        action[-1] -= 0.2
        action[-1] = np.max([action[-1], 0])
        env.step(action)

    # 7.抬起夹爪
    # 目标：放置后抬起夹爪，避免碰撞物体。
    time7 = 1
    T7 = sm.SE3.Trans(0.0, 0.0, 0.1) * T6
    position_parameter7 = LinePositionParameter(T6.t, T7.t)
    attitude_parameter7 = OneAttitudeParameter(sm.SO3(T6.R), sm.SO3(T7.R))
    cartesian_parameter7 = CartesianParameter(position_parameter7, attitude_parameter7)
    velocity_parameter7 = QuinticVelocityParameter(time7)
    trajectory_parameter7 = TrajectoryParameter(cartesian_parameter7, velocity_parameter7)
    planner7 = TrajectoryPlanner(trajectory_parameter7)
    # 执行planner_array = [planner7]
    time_array = [0.0, time7]
    planner_array = [planner7]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    time8 = 1
    q8 = robot.get_joint()
    q9 = q0
    parameter8 = JointParameter(q8, q9)
    velocity_parameter8 = QuinticVelocityParameter(time8)
    trajectory_parameter8 = TrajectoryParameter(parameter8, velocity_parameter8)
    planner8 = TrajectoryPlanner(trajectory_parameter8)
    # 执行planner_array = [planner8]
    time_array = [0.0, time8]
    planner_array = [planner8]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break

def get_point_cloud_bounds( target_pcd: np.ndarray = None) -> dict:
        """获取点云X/Y/Z轴的最大值和最小值（纯numpy实现）"""
        
        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        x_min, y_min, z_min = np.min(target_pcd, axis=0)
        x_max, y_max, z_max = np.max(target_pcd, axis=0)
        print("xmin:",x_min)
        print("y_min:",y_min)
        print("z_min:",z_min)
        print("x_max:",x_max)
        print("y_max:",y_max)
        print("z_max:",z_max)
        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max
        }

def get_point_cloud_center( target_pcd: np.ndarray = None, use_bounds: bool = True) -> np.ndarray:
        """计算点云的中心点坐标（纯numpy实现）"""
        

        if target_pcd is None or not isinstance(target_pcd, np.ndarray) or target_pcd.shape[1] != 3:
            raise ValueError("输入点云无效！需为N×3的numpy数组，或先调用generate_point_cloud生成点云")

        if use_bounds:
            bounds = get_point_cloud_bounds(target_pcd)
            x_center = (bounds["x_min"] + bounds["x_max"]) / 2
            y_center = (bounds["y_min"] + bounds["y_max"]) / 2
            z_center = (bounds["z_min"] + bounds["z_max"]) / 2
        else:
            x_center, y_center, z_center = np.mean(target_pcd, axis=0)

        return np.array([x_center, y_center, z_center])
def pasue_1(env):
    #T_wb = robot.base
    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    #global robot
    global action
    if action is None or not isinstance(action, np.ndarray) or action.shape[0] != 7:
        action = np.zeros(7)
    robot = env.robot
    time1 = 1
    q0 = robot.get_joint()
    print("q0:",q0)
    #预抓取位置
    q1 = np.array([0.0, 0.0, np.pi / 2 * 0, 0, 0 , 0.0])
    #q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    parameter0 = JointParameter(q0, q1)#关节参赛
    velocity_parameter0 = QuinticVelocityParameter(time1)# 五次多项式速度参数
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)# 轨迹参数
    planner1 = TrajectoryPlanner(trajectory_parameter0) # 轨迹规划器
    time_array = [0.0, time1]
    planner_array = [planner1]
    total_time = np.sum(time_array)
    time_step_num = round(total_time / 0.002) + 1
    times = np.linspace(0.0, total_time, time_step_num)
    time_cumsum = np.cumsum(time_array)
    for timei in times:
        for j in range(len(time_cumsum)):
            if timei == 0.0:
                break
            if timei <= time_cumsum[j]:
                planner_interpolate = planner_array[j - 1].interpolate(timei - time_cumsum[j - 1])
                if isinstance(planner_interpolate, np.ndarray):
                    joint = planner_interpolate
                    robot.move_joint(joint)
                else:
                    robot.move_cartesian(planner_interpolate)
                    joint = robot.get_joint()
                action[:6] = joint
                env.step(action)
                break
if __name__ == '__main__':
    env = UR5GraspEnv()
    try:
        env.reset()

        n = 4 # 循环次数，连续抓取物体
        for _ in range(n): 
            for i in range(500): # 1000
                env.step()
            # 1. 获取图像和深度图
            pasue_1(env)
            imgs = env.render(0)
            color_img_path = imgs['img'] # MuJoCo 渲染的是 RGB
            depth_img_path = imgs['depth']
            #cv2.imwrite("color_img_path",color_img_path)
            # 将MuJoCo渲染的是RGB转化为OpenCV默认使用BGR颜色空间
            color_img_path = cv2.cvtColor(color_img_path, cv2.COLOR_RGB2BGR)
            # 保存/查看图片
            cv2.imwrite('color_img_path.jpg', color_img_path)
            cv2.imwrite('color_img_depth.jpg',depth_img_path)
            #cv2.imshow('color', color_img_path)
            #cv2.waitKey(0)
            # 2. SAM分割图像
            target_class = input("\n"
                        "===============\n"
                        "Enter class name: ").strip()

            target=target_class
            
            mask_img_path = segment_image_ground('color_img_path.jpg',target_class)
            print('color_img_path:',color_img_path)
            print('mask_img_path:',mask_img_path)
            print('depth_img_path:',depth_img_path)
            # 3. 获取物体的点云数据
            end_points, cloud_o3d = get_and_process_data(color_img_path, depth_img_path, mask_img_path)
            print(type(end_points))
            print(type(cloud_o3d))
            cloud=end_points['cloud_colors']
            
            # 4. 获取抓取点对应的夹爪姿态
            #gg = generate_grasps(end_points, cloud_o3d, True) # True or False
            gg=getGrasp(color_img_path,depth_img_path,mask_img_path)
            print("gg:",gg)
            
            # 5. 仿真执行抓取
            execute_grasp(env, gg)
    finally:
        env.close()
