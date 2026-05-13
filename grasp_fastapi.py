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
import base64
import io
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Any
import uuid
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
import datetime
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
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
#from doubao import gen_content,fault_recover,get_sorce
from qianwen import gen_content,fault_recover,get_sorce
#from reactive_planner import ReactiveController
from contextlib import asynccontextmanager
#全局变量
depth_img_path=None
color_img_path=None
mask_img_path=None
grasp_gg=None
detections=None
end_points=None
cloud_o3d=None
robot=None
T_wo=None
action=np.zeros(7)
q0=None
path=0
history_grasps={}
target=None

content_list_all=[]
dot_pos=None
import queue
import threading
import py_trees
from py_trees.common import Status

# 定义一个基础技能类，封装对你现有逻辑的调用
class SkillAction(py_trees.behaviour.Behaviour):
    def __init__(self, name, action_func, *args, **kwargs):
        super(SkillAction, self).__init__(name)
        self.action_func = action_func
        self.args = args
        self.kwargs = kwargs

    def update(self):
        # 调用你现有的 FastAPI 逻辑函数
        try:
            # 假设你的函数返回 {"status": "success/failure"}
            result = self.action_func(*self.args, **self.kwargs)
            if result.get("status") == "success":
                return Status.SUCCESS
            else:
                return Status.FAILURE
        except Exception:
            return Status.FAILURE

# 定义一个“豆包大模型”判断节点
class LLMCheck(py_trees.behaviour.Behaviour):
    def __init__(self, name, task_name):
        super(LLMCheck, self).__init__(name)
        self.task_name = task_name

    def update(self,content_list):
        # 调用豆包判断逻辑
        # 传入 content_list_all
        ans = gen_content(content_list, self.task_name)
        res = json.loads(ans)
        return Status.SUCCESS if res['status'] == 'SUCCESS' else Status.FAILURE
    
# 创建一个全局队列，maxsize=10 防止图片堆积过多占用内存
image_task_queue = queue.Queue(maxsize=10)

task_list=[]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动逻辑：替代原来的 @app.on_event("startup")
    # 在线程池中初始化环境，避免阻塞事件循环
    grasp_service.init_environment()
    global env
    pasue_1(env)
    move_xyz(0,0,0.1)
    print("机器人抓取服务启动成功")
    yield  # 应用在此处开始接收和处理请求
    # 关闭逻辑：替代原来的 @app.on_event("shutdown")
    print("开始清理资源...")
    
    if env is not None:
        env.close()
    #executor.shutdown(wait=False)
    print("资源清理完成")

# 创建 FastAPI 应用
app = FastAPI(
    title="机器人抓取服务",
    description="基于 GraspNet 和 UR5 机器人的抓取服务 API",
    version="1.0.0",
    lifespan=lifespan
)

# 全局变量和线程池
env = None
env_lock = asyncio.Lock()
#executor = ThreadPoolExecutor(max_workers=2)


class GraspData(BaseModel):
    translation: List[float]
    rotation: List[List[float]]
    width: float
    score: float
    height:float
    depth:float
# 服务类
class GraspService:
    def __init__(self):
        self.env_initialized = False
        self.task_results = {}  # 存储异步任务结果
        
    def init_environment(self):
        """初始化机器人环境"""
        global env
        try:
            if env is None:
                env = UR5GraspEnv()
                env.reset()
                for i in range(500): # 1000
                    env.step()
                self.env_initialized = True
                print("机器人环境初始化成功")
            else:
                self.env_initialized = True
        except Exception as e:
            print(f"环境初始化失败: {e}")
            self.env_initialized = False


# 初始化服务
grasp_service = GraspService()

# ================= 事件处理 ====================

@app.post("/move-to",description="移动机械臂到指定位置")
async def move_xyz(
    x:float,
    y:float,
    z:float
):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        async with env_lock:
            #pasue_1(env=env)
            move_to(x,y,z)
            
        return {
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移动机械臂时出错: {str(e)}")

@app.post("/rotate-to",description="旋转机械臂到指定位置")
async def rotate_to(
    roll:float,
    pitch:float,
    yaw:float
):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        async with env_lock:
            #pasue_1(env=env)
            rotate_to(roll, pitch, yaw)
            
        return {
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移动机械臂时出错: {str(e)}")


@app.post("/camera-image",description="获取相机的图像")
async def get_camera_image():
    """获取当前相机图像"""
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        async with env_lock:
            #pasue_1(env=env)
            global color_img_path
            global depth_img_path
            # 获取当前系统时间
            global content_list_all
            content_list_all.clear()
            #wirte_quan(0)
            imgs1 = env.render(0)
            color_img = imgs1['img']
            depth_img = imgs1['depth']
            depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('color_img1.jpg', color_img)
            cv2.imwrite('color_img_depth1.jpg',depth_img)
            global task_list
            task_list.append('获取相机的图像:SUCCESS')
            color_img_path=color_img
        return {
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取相机图像时出错: {str(e)}")

@app.post("/detect-object",description="识别与分割")
async def detect(
    target_class:str
):
    try:
        #detections, vis_img = detect_objects(color_img_path, target_class)
        global mask_img_path,target
        target=target_class
        #mask_img_path = segment_image(color_img_path,detections)
        mask_img_path=segment_image_ground('color_img1.jpg',target_class)
        global task_list
        task_list.append(f'识别并分割物体{target_class}:SUCCESS')
        return {
                "status":"success",
                #"mask":mask_img_path
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"识别与分割时出错: {str(e)}")




@app.post("/create-cloud",description="生成点云数据")
async def cloud():
    try:

        end_point, cloud,ans = get_and_process_data(color_img_path, depth_img_path, mask_img_path)
        
        global end_points,cloud_o3d
        end_points=end_point
        cloud_o3d=cloud

        print("end_point:",end_point)
        print("cloud_o3d:",cloud_o3d)
        global task_list
        task_list.append('生成点云数据:SUCCESS')
        return{
        "status":"success",
        
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成点云时出错: {str(e)}")

# 文件上传版本的处理图像接口
@app.post("/create-grasp",description="生成抓取姿势")
async def create_grasp(
):
    try:
        #gg = generate_grasps(end_points, cloud_o3d, False) # True or False
        gg=getGrasp(color_img_path,depth_img_path,mask_img_path)
        print("gg:",gg)
        global grasp_gg
        grasp_gg=gg
        best_gg=gg[0]
        garsp_data=GraspData(translation=best_gg.translation.tolist(),
                rotation=best_gg.rotation_matrix.tolist(),
                width=float(best_gg.width),
                score=float(best_gg.score),
                height=float(best_gg.height),
                depth=float(best_gg.depth))
        global task_list
        task_list.append('生成抓取姿势:SUCCESS')
        response_data = {
            "status": "success",
            "best_grasp": garsp_data
        }
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成抓取时出错: {str(e)}")

@app.post("/move-pregrasp",description="机械臂移动到预抓取位置")
async def move_pregrasp(
    #request: ExecuteGraspRequest
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        # 从抓取数据重建GraspGroup
        gg=grasp_gg
        init_world(gg)
        # 使用锁确保同一时间只有一个抓取操作
        content_list=[]
        extute_pre(content_list)
        
        content_list=tobase(content_list)
        global content_list_all
        for content in content_list:
            content_list_all.append(content)
        ans= fault_dection(content_list,'移动到预抓取位置')
        print('ans:',ans)
        return ans
        '''ans=gen_content(content_list,'移动到预抓取位置')
        res=json.loads(ans)
        status=res['status']
        global task_list
        if status=='FAILURE':
            task_list.append('移动到预抓取位置:FAILURE')
            return {
                "status": "failure",
                "message": "移动到预抓取位置失败",
            }
        else:
            task_list.append('移动到预抓取位置:SUCCESS')
            return {
                "status": "success",
                "message": "移动到预抓取位置完成",
            }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移动到预抓取位置时出错: {str(e)}")
    
@app.post("/gripper-action")
async def gripper(
    flag:int
):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        content_list=[]
        global content_list_all
        global task_list
        gripper_action(flag,content_list)
        if flag==1:
            content_list=tobase(content_list)
            
            for content in content_list:
                content_list_all.append(content)
            ans= fault_dection(content_list,'夹爪闭合')
            print('ans:',ans)
            return ans
            '''ans=gen_content(content_list,'夹爪闭合')
            
            res=json.loads(ans)
            
            #task_list.append('夹爪闭合')
            status=res['status']
            if status=='FAILURE':
                task_list.append('夹爪闭合:FAILURE')
                return {
                "status": "failure",
                }
            else:
                task_list.append('夹爪闭合:SUCCESS')
                return {
                "status": "success",
                }'''
        else:
            content_list=tobase(content_list)
            for content in content_list:
                content_list_all.append(content)
            ans= fault_dection(content_list,'夹爪开启')
            print('ans:',ans)
            return ans
            '''ans=gen_content(content_list,'夹爪开启')
            res=json.loads(ans)
            status=res['status']
            
            #task_list.append('夹爪开启')
            if status=='FAILURE':
                task_list.append('夹爪开启:FAILURE')
                return {
                "status": "failure",
                }
            else:
                task_list.append('夹爪开启:SUCCESS')
                return {
                "status": "success",
                }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"夹爪开合时出错: {str(e)}")

@app.post("/move-grasp",description="移动到抓取位置")
async def move_grasp(
    #request: ExecuteGraspRequest
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        # 从抓取数据重建GraspGroup
        gg=grasp_gg
        init_world(gg)
        content_list=[]
        exeute_grasp(content_list)
        content_list=tobase(content_list)
        
        global content_list_all
        for content in content_list:
            content_list_all.append(content)
        #content_list_all.append(content_list)
        ans= fault_dection(content_list,'移动到抓取位置')
        print('ans:',ans)
        return ans
        '''ans=gen_content(content_list,'移动到抓取位置')
        res=json.loads(ans)
        status=res['status']
        global task_list
        #task_list.append('移动到抓取位置')
        if status=='FAILURE':
            task_list.append('移动到抓取位置:FAILURE')
            return {
                "status": "failure",
                "message": "移动到抓取位置失败",
            }
        else:
            task_list.append('移动到抓取位置:SUCCESS')
            return {
                "status": "success",
                "message": "移动到抓取位置成功",
            }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移动到抓取位置时出错: {str(e)}")

@app.post("/vertical-grasp",description="提升物体")
async def vertical(
    #request: ExecuteGraspRequest
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        # 从抓取数据重建GraspGroup
        #vertical_lift()
        content_list=[]
        vertical_lift(content_list)
        content_list=tobase(content_list)
        global content_list_all
        for content in content_list:
            content_list_all.append(content)
        ans=fault_dection(content_list,'提升物体')
        print('ans:',ans)
        return ans
        #content_list_all.append(content_list)
        '''ans=gen_content(content_list,'提升物体')
        res=json.loads(ans)
        status=res['status']
        global task_list
        if status=='FAILURE':
            task_list.append(f"提升物体:FAILURE,reason:{res['reason']}")
            print('task_list:',task_list)
            fault_recover(content_list,task_list,target)
            return {
                "status": "failure",
                "message": "提升物体失败",
            }
        else:
            task_list.append('提升物体:SUCCESS')
            return {
                "status": "success",
                "message": "提升物体成功",
            }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"提升物体时出错: {str(e)}")
    
@app.post("/execute-grasp2",description="移动到碗的上方")
async def execute_grasp_2(
    #x:float,y:float
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        pos=get_bowl_pos()

        content_list=[]
        grasp_action2(pos[0],pos[1],content_list)
        content_list=tobase(content_list)
        global content_list_all
        for content in content_list:
            content_list_all.append(content)
        
        

        ans= fault_dection(content_list,'移动到碗的上方')
        print('ans:',ans)
        return ans
        '''ans=gen_content(content_list,'移动到碗的上方')
        res=json.loads(ans)
        status=res['status']
        if status=='FAILURE':
            task_list.append('移动到碗的上方:FAILURE')
            return {
                "status": "failure",
                "message": "抓取执行失败",
                #"executed_grasp_index": request.grasp_index
            }
        else:
            task_list.append('移动到碗的上方:SUCCESS')
            return {
                "status": "success",
                "message": "抓取执行完成",
                #"executed_grasp_index": request.grasp_index
            }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"移动到碗的上方时出错: {str(e)}")
    
@app.post("/execute-init",description="恢复初始状态")
async def execute_init(
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        content_list=[]
        garsp_init(content_list)
        global color_img_path,depth_img_path,mask_img_path
        color_img_path=None
        depth_img_path=None
        mask_img_path=None
        content_list=tobase(content_list)
        global content_list_all
        for content in content_list:
            content_list_all.append(content)
        ans= fault_dection(content_list,'回到初始位置')
        print('ans:',ans)
        return ans
        '''ans=gen_content(content_list,'回到初始位置')
        res=json.loads(ans)
        status=res['status']
        if status=='FAILURE':
            task_list.append('回到初始位置:FAILURE')
            return {
                "status": "failure",
                "message": "恢复初始状态失败",
                #"executed_grasp_index": request.grasp_index
            }
        else:
            task_list.append('回到初始位置:SUCCESS')
            return {
                "status": "success",
                "message": "恢复初始状态",
            #"executed_grasp_index": request.grasp_index
            }'''
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复初始状态时出错: {str(e)}")
@app.post("/reset-environment",description="重置环境")
async def reset_environment():
    """重置机器人环境"""
    try:
        async with env_lock:
            global env
            if env is not None:
                env.close()
            env = UR5GraspEnv()
            env.reset()
            for i in range(500): # 1000
                env.step()
            grasp_service.env_initialized = True
            global content_list_all
            content_list_all.clear()
        return {
            "status": "success",
            "message": "环境重置成功"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置环境时出错: {str(e)}")

@app.post("/gen-sorce",description="方案验证")
async def fanganyanzheng(
    #x:float,y:float
    ):
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
        global content_list_all,task_list
        ans= get_sorce(content_list_all,task_list)
        print('异常处理方案验证分数:',ans)
        return ans
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"方案验证: {str(e)}")

'''@app.post("/start-intelligent-task", description="启动基于行为树和VLM的智能抓取任务")
async def start_intelligent_task(target_name: str = "apple"):
    """
    启动统一故障处理框架
    """
    try:
        if not grasp_service.env_initialized:
            raise HTTPException(status_code=500, detail="机器人环境未初始化")
            
        # 使用后台任务或线程来运行 BT，避免阻塞 FastAPI 主线程
        # 这里为了演示直接在 ThreadPool 中运行
        loop = asyncio.get_running_loop()
        controller = ReactiveController()
        controller.current_target = target_name
        
        # 由于 BT 循环包含大量 IO 和推理，建议放入线程
        await loop.run_in_executor(None, controller.run_reactive_loop)
        
        return {
            "status": "finished",
            "final_history": task_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"智能任务执行出错: {str(e)}")'''

def fault_dection(content_list,action):
    ans=gen_content(content_list,action)
    res=json.loads(ans)
    status=res['status']
    global task_list
        #task_list.append('移动到抓取位置')
    if status=='FAILURE':
        task_list.append(f'{action}:FAILURE')
        fault_recover(content_list_all,task_list,target)
        return {
                "status": "failure",
                "message": f"{action}失败",
            }
    else:
        task_list.append(f'{action}:SUCCESS')
        return {
                "status": "success",
                "message": f"{action}成功",
            }


'''def wirte_quan(i,j):
    # 获取当前系统时间
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 设置字体、大小、颜色等参数
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)  # 白色
    line_type = 2
    # 将时间文字添加到图像的左上角 (10, 30) 位置
            #cv2.putText(frame, current_time, (10, 30), font, font_scale, font_color, line_type)
            #全局相机
    imgs0 = env.render(j)
    color_img = imgs0['img']
            #depth_img_path=depth_img
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    cv2.putText(color_img, current_time, (10, 30), font, font_scale, font_color, line_type)
    cv2.imwrite(f'{ROOT_DIR}/scenes_imgs/color_quan{j}__{i}.jpg', color_img)'''

def baseapi(content_list,j):
    imgs0 = env.render(j)
    color_img = imgs0['img']
            #depth_img_path=depth_img
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
    content_list.append(color_img)
    
def tobase(content_list):
    ans=[]
    for img in content_list:
        success, encoded_image = cv2.imencode('.jpg', img)
        if success:
    # 3. 将二进制数据转换为 Base64 字符串
            base64_data = base64.b64encode(encoded_image).decode('utf-8')
            ans.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_data}"
            }
            })
    return ans
def choose_model():
    """Initialize SAM predictor with proper parameters"""
    model_weight = 'sam_b.pt'
    '''
    task='segment'：指定任务为分割
mode='predict'：设置模式为预测模式
model=model_weight：加载指定的模型权重
conf=0.25：设置置信度阈值为0.25，影响检测结果的筛选
save=False：不保存预测结果

    '''
    overrides = dict(
        task='segment',
        mode='predict',
        # imgsz=1024,
        model=model_weight,
        conf=0.25,
        save=False
    )
    '''
    创建并返回SAM预测器实例
所有配置通过overrides参数传递
    '''
    return SAMPredictor(overrides=overrides)

'''
接受两个参数：YOLO-World模型实例和目标类别名称
函数名清晰地表明其功能：设置要检测的类别
'''

def set_classes(model, target_class):
    """Set YOLO-World model to detect specific class"""
    model.set_classes([target_class])


def detect_objects(image_or_path, target_class=None):
    """
    Detect objects with YOLO-World
    image_or_path: can be a file path (str) or a numpy array (image).
    Returns: (list of bboxes in xyxy format, detected classes list, visualization image)
    """
    model = YOLO("yolov8s-world.pt")
    if target_class:
        set_classes(model, target_class)

    # YOLOv8 的 predict 可同时处理 文件路径(str) 或 图像数组(np.ndarray)
    results = model.predict(image_or_path)
    ''''
    功能：执行目标检测
参数：image_or_path - 输入可以是文件路径或图像数组
技术细节：YOLO模型使用深度神经网络进行目标检测，包括：
特征提取
边界框预测
类别概率计算
说明：predict方法返回检测结果，包含目标位置、类别和置信度信息

功能：获取检测结果中的边界框信息
技术细节：YOLO检测结果是一个包含多个属性的对象
boxes属性包含所有检测到的边界框
每个边界框包含：
坐标信息
置信度
类别信息
说明：0索引表示获取第一个检测结果（通常对应输入图像）
Python
    '''
    boxes = results[0].boxes
    '''
    功能：获取可视化后的检测结果图像
技术细节：YOLO模型的plot方法会：
在原始图像上绘制检测框
绘制类别标签
绘制置信度分数
说明：返回的vis_img是带有检测结果标注的图像，可以直接显示或保存
    '''
    vis_img = results[0].plot()  # Get visualized detection results

    # Extract valid detections
    '''
    功能：初始化一个空列表，用于存储有效检测结果
技术细节：每个有效检测结果将是一个字典，包含：
xyxy: 边界框坐标 (x1, y1, x2, y2)
conf: 置信度分数
cls: 检测到的类别
说明：列表将存储所有有效的检测结果字典
    '''
    valid_boxes = []
    for box in boxes:
        if box.conf.item() > 0.25:  # Confidence threshold 0.25 
            '''
            
            功能：将有效的检测框信息添加到结果列表
技术细节：
box.xyxy: 包含边界框坐标信息的Tensor
.tolist(): 将NumPy数组转换为Python列表
box.cls: 包含预测类别的Tensor
results0.names: 将类别索引映射到类别名称的字典
说明：每个有效检测框包含坐标、置信度和类别信息
            '''
            valid_boxes.append({
                "xyxy": box.xyxy[0].tolist(),
                "conf": box.conf.item(),
                "cls": results[0].names[box.cls.item()]
            })

    return valid_boxes, vis_img

''''

功能：定义一个函数来处理SAM(Segment Anything Model)的结果
参数：results - SAM模型的输出结果
文档：函数目的是从SAM结果中获取掩码(mask)和中心点(center point)

'''
def process_sam_results(results):
    """Process SAM results to get mask and center point"""

    '''
    
    功能：检查输入结果是否有效
技术细节：
如果results为空或第一个结果没有掩码信息
则返回(None, None)表示处理失败
说明：这是基本的错误检查，确保输入数据有效
    '''
    if not results or not results[0].masks:
        return None, None
    '''
    
    功能：获取SAM检测到的第一个掩码
技术细节：
results[0].masks.data：获取SAM结果中的掩码数据
[0]：假设只处理第一个检测到的对象
.cpu().numpy()：将张量从GPU移动到CPU，并转换为NumPy数组
说明：这是从SAM结果中提取掩码信息的关键步骤
    '''
    # Get first mask (assuming single object segmentation)
    mask = results[0].masks.data[0].cpu().numpy()
    '''
    功能：将掩码转换为二值图像
技术细节：
(mask > 0)：创建一个布尔数组，将大于0的值设为True
.astype(np.uint8)：将布尔值转换为8位无符号整数(0或255)
* 255：将True(1)转换为255，False(0)保持不变
说明：这一步将掩码从概率图转换为黑白图像，其中白色表示目标区域
    '''
    mask = (mask > 0).astype(np.uint8) * 255
    '''
    功能：在二值掩码上查找轮廓
技术细节：
cv2.findContours：OpenCV函数，用于查找图像中的轮廓
cv2.RETR_EXTERNAL：只检测外部轮廓
cv2.CHAIN_APPROX_SIMPLE：压缩水平轮廓，只保留拐点
说明：这一步将掩码转换为轮廓，为计算中心点做准备
    '''
    # Find contour and center
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    '''
    功能：计算轮廓的几何矩
技术细节：cv2.moments函数计算轮廓的矩，包括：
面积(area)
重心(center)
一阶、二阶矩等
说明：矩是轮廓分析的基础，可以用来计算中心点、惯性矩等
    
    '''
    M = cv2.moments(contours[0])
    '''
    功能：检查轮廓是否有有效面积
技术细节：m00是轮廓的面积
如果面积为0，说明轮廓退化为点或不存在
避免除以零的错误
说明：这是重要的有效性检查，确保轮廓确实包含目标

    '''
    if M["m00"] == 0:
        return None, mask
    '''
    功能：计算轮廓的中心点坐标
技术细节：
m10：x坐标的一阶矩
m01：y坐标的矩
m00：轮廓的面积
中心点公式：cx = m10/m00, cy = m01/m00
说明：这是计算轮廓重心的标准方法
    '''
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy), mask


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
    # end_points = {'point_clouds': cloud_sampled}
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.05, 0 ,1.2]) 
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    R_wc = T_wc.R       # 3x3
    t_wc = T_wc.t       # 3x1
    points_world = (R_wc @ cloud_masked.T).T + t_wc.reshape(1, 3)
    print('中心点:',get_point_cloud_center(points_world))
    ans=get_point_cloud_center(points_world)
    end_points = dict()
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud_o3d,ans

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
    COLLISION_THRESH = 0.01
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
        voxel_size = 0.01
        collision_thresh = 0.01
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

    # 5. 返回抓取得分最高的抓取（对抓取预测的接近方向进行垂直角度限制）
    # 将 gg 转换为普通列表
    all_grasps = list(gg)
    vertical = np.array([0, 0, 1])  # 期望抓取接近方向（垂直桌面） np.array([0, 0, 1])
    angle_threshold = np.deg2rad(30)  # 30度的弧度值 np.deg2rad(30)
    filtered = []
    global history_grasps
    for grasp in all_grasps:
        if history_grasps.get(target) is not None:
            if grasp in history_grasps[target]:
                continue
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

    if history_grasps.get(target)==None:
        history_grasps[target]=[best_grasp]
    else:
        history_grasps[target].append(best_grasp)
    best_translation = best_grasp.translation
    best_rotation = best_grasp.rotation_matrix
    best_width = best_grasp.width

    # 创建一个新的 GraspGroup 并添加最佳抓取
    new_gg = GraspGroup()            # 初始化空的 GraspGroup
    new_gg.add(best_grasp)           # 添加最佳抓取
    if visual:
        grippers = new_gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    return new_gg
    # return best_translation, best_rotation, best_width

def pasue_1(env):
    #T_wb = robot.base
    # 1.机器人运动到预抓取位姿
    # 目标：将机器人从当前位置移动到预抓取姿态（q1）
    global robot
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
                global action
                action[:6] = joint
                env.step(action)
                break
    


def gripper_action(flag:int,content_list):
    if flag==1:
        for i in range(1000):
            action[-1] += 0.2
            action[-1] = np.min([action[-1], 255])
            env.step(action)
            if i %100==0:
                imgs0 = env.render(1)
                color_img = imgs0['img']
            #depth_img_path=depth_img
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                content_list.append(color_img)
                imgs0 = env.render(2)
                color_img = imgs0['img']
            #depth_img_path=depth_img
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                content_list.append(color_img)
    elif flag==0:
        for i in range(1000):
            action[-1] -= 0.2
            action[-1] = np.max([action[-1], 0])
            env.step(action)
            if i %100==0:
                imgs0 = env.render(1)
                color_img = imgs0['img']
            #depth_img_path=depth_img
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                content_list.append(color_img)
                imgs0 = env.render(2)
                color_img = imgs0['img']
            #depth_img_path=depth_img
                color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
                content_list.append(color_img)
def init_world(gg):
    '''data=env.mj_data
    model=env.mj_model
    # 1. 获取相机在世界坐标系下的位置和旋转矩阵
    # cam_xpos: 相机中心的世界坐标
    # cam_xmat: 相机的旋转矩阵 (3x3)
    cam_id = model.camera('ee_camera').id
    t_wc_raw = data.cam_xpos[cam_id]
    R_wc_raw = data.cam_xmat[cam_id].reshape(3, 3)
    # 2. 坐标系对齐 (MuJoCo -> AnyGrasp)
    # 转换为 SE3 格式
    T_wc_raw = sm.SE3.Rt(sm.SO3(R_wc_raw), t_wc_raw)
    # 绕 X 轴旋转 180 度，使 Z 轴向前，Y 轴向下
    T_align = sm.SE3.Rx(np.pi)
    T_wc = T_wc_raw * T_align'''
    # 3. 获取 AnyGrasp 抓取位姿 T_co
    # 注意：gg.translations[0] 和 gg.rotation_matrices[0] 是 AnyGrasp 预测的
    n_wc = np.array([0.0, -1.0, 0.0]) 
    o_wc = np.array([-1.0, 0.0, -0.5]) 
    t_wc = np.array([0.05, 0 ,1.2]) 
    #t_wc = np.array([0.85, 0.8, 1.6]) 
    T_wc = sm.SE3.Trans(t_wc) * sm.SE3(sm.SO3.TwoVectors(x=n_wc, y=o_wc))
    T_co = sm.SE3.Trans(gg.translations[0]) * sm.SE3(sm.SO3.TwoVectors(x=gg.rotation_matrices[0][:, 0], y=gg.rotation_matrices[0][:, 1]))
    global T_wo
    T_wo = T_wc * T_co
    global q0
    q0 = robot.get_joint()

def get_bowl_pos():
    mask_img_path=segment_image_ground('color_img1.jpg','bowl')
    end_point, cloud,ans = get_and_process_data(color_img_path, depth_img_path, mask_img_path)
    pos=[]
    pos.append(ans[0])
    pos.append(ans[1])
    return pos

def extute_pre(content_list):
    # 1.机器人运动到预抓取位姿
    time1 = 1
    global action
    
    path=0
    content_list1=[]
    content_list2=[]
    content_list3=[]
    #预抓取位置
    q1 = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
    parameter0 = JointParameter(q0, q1)#关节参赛
    velocity_parameter0 = QuinticVelocityParameter(time1)# 五次多项式速度参数
    trajectory_parameter0 = TrajectoryParameter(parameter0, velocity_parameter0)# 轨迹参数
    planner1 = TrajectoryPlanner(trajectory_parameter0) # 轨迹规划器
    # 执行planner_array = [planner1]
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
        if path%100==0:
            #wirte_quan(path,1)
            #wirte_quan(path,2)
            #baseapi(content_list,1)
            #baseapi(content_list,2)
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list1.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list2.append(color_img)
        path+=1
    time2 = 1
    robot.set_joint(q1)
    T1 = robot.get_cartesian()
    T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
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
        if path%100==0:
            '''wirte_quan(path,1)
            wirte_quan(path,2)'''
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list1.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list2.append(color_img)
        path+=1
            #baseapi(content_list,1)
            #baseapi(content_list,2)
    for img in content_list1:
        content_list.append(img)
    for img in content_list2:
        content_list.append(img)
    global dot_pos
    dot_pos=T2
def exeute_grasp(content_list):
    time3 = 1
    path=0
    global dot_pos
    T2=dot_pos
    #T2 = T_wo * sm.SE3(-0.1, 0.0, 0.0)
    T3 = T_wo
    content_list1=[]
    content_list2=[]
    content_list3=[]
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
        if path%100==0:
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list1.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list2.append(color_img)
            imgs0 = env.render(3)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list3.append(color_img)
        path+=1
    for img in content_list1:
        content_list.append(img)
    for img in content_list2:
        content_list.append(img)
    for img in content_list3:
        content_list.append(img)
    dot_pos=T3
def garsp_init(content_list):
    # 8.回到初始位置
    # 目标：机器人返回初始姿态（q0），完成整个任务。
    global dot_pos 
    dot_pos=None
    time8 = 1
    q8 = robot.get_joint()
    path=0
    q9 = np.array([0.0, 0.0, np.pi / 2 * 0, 0, 0 , 0.0])
    
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
        if path%100==0:
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
        path+=1
    
def move_to(x,y,z):
    T_current = robot.get_cartesian()
    time1=1
    # 2. 计算目标位姿 (T_target)
    # 保持旋转矩阵 R 不变，只在平移向量 t 上做叠加
    target_translation = T_current.t + np.array([x, y, z])
    T_target = sm.SE3.Rt(T_current.R, target_translation)
    # 3. 路径规划 (笛卡尔空间直线)
    position_param = LinePositionParameter(T_current.t, T_target.t)
    attitude_param = OneAttitudeParameter(sm.SO3(T_current.R), sm.SO3(T_target.R))
    cartesian_param = CartesianParameter(position_param, attitude_param)
    
    # 4. 速度规划 (五次多项式)
    velocity_param = QuinticVelocityParameter(time1)
    trajectory_param = TrajectoryParameter(cartesian_param, velocity_param)
    planner = TrajectoryPlanner(trajectory_param)
    
    time_array = [0.0, time1]
    planner_array = [planner]
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
    global dot_pos
    dot_pos=T_target

def rotate_to(roll, pitch, yaw,frame='tool'):
    T_current = robot.get_cartesian()
    time1=1
    R_current = sm.SO3(T_current.R)
    #R_delta = sm.SO3.EulerZYX(yaw, pitch, roll)
    T_delta = sm.SE3.RPY(roll, pitch, yaw)
    if frame=='tool':
        T_target = T_current * T_delta
    elif frame=='world':
        T_target = sm.SE3.Rt(T_delta.R @ T_current.R, T_current.t)
    #T_target = sm.SE3.Rt(R_target.R, T_current.t)
    position_param = LinePositionParameter(T_current.t, T_target.t) 
    attitude_param = OneAttitudeParameter(sm.SO3(T_current.R), sm.SO3(T_target.R))
    
    cartesian_param = CartesianParameter(position_param, attitude_param)
    velocity_param = QuinticVelocityParameter(time1)
    trajectory_param = TrajectoryParameter(cartesian_param, velocity_param)
    planner = TrajectoryPlanner(trajectory_param)
    time_array = [0.0, time1]
    planner_array = [planner]
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
    global dot_pos
    dot_pos=T_target

def vertical_lift(content_list):
    path=0
    global dot_pos
    T3=dot_pos
    #T3=T_wo
    time4 = 1
    T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3 # 通过在T3的基础上向上偏移0.3单位得到的，用于控制机器人上升一定的高度
    position_parameter3 = LinePositionParameter(T3.t, T4.t)
    attitude_parameter3 = OneAttitudeParameter(sm.SO3(T3.R), sm.SO3(T4.R))
    cartesian_parameter3 = CartesianParameter(position_parameter3, attitude_parameter3)
    velocity_parameter3 = QuinticVelocityParameter(time4)
    trajectory_parameter3 = TrajectoryParameter(cartesian_parameter3, velocity_parameter3)
    planner4 = TrajectoryPlanner(trajectory_parameter3)
    time_array = [0.0, time4]
    planner_array = [planner4]
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
        if path%50==0:
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
            imgs0 = env.render(3)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
        path+=1
    dot_pos=T4
def grasp_action2(x,y,content_list):
    # 4.提起物体
    # 目标：抓取后垂直提升物体（避免碰撞桌面）。
    global dot_pos
    path=0
    T3=T_wo
    #time4 = 1
    #T4 = sm.SE3.Trans(0.0, 0.0, 0.3) * T3
    T4=dot_pos
    # 5.水平移动物体
    # 目标：将物体水平移动到目标放置位置，保持高度不变。
    time5 = 1
    T5 = sm.SE3.Trans(x, y, T4.t[2]) * sm.SE3(sm.SO3(T4.R)) #  通过在T4的基础上进行平移得到，这里的1.4, 0.3是场景中的固定点坐标，而不是偏移量
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
    time_array = [0.0, time5, time6]
    planner_array = [ planner5, planner6]
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
        if path%100==0:
            imgs0 = env.render(1)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
            imgs0 = env.render(2)
            color_img = imgs0['img']
            #depth_img_path=depth_img
            color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)
            content_list.append(color_img)
        path+=1
    dot_pos=T6
def get_point_cloud_bounds(target_pcd: np.ndarray = None) -> dict:
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

def start():
    import uvicorn
    
    uvicorn.run(
        "grasp_fastapi:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1  # 由于环境是全局的，只能使用单worker
    )
# ================= 原有函数（保持不变） ====================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "grasp_fastapi:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        workers=1  # 由于环境是全局的，只能使用单worker
    )