import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor

import logging
# 禁用 Ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# def choose_model():
#     """Initialize SAM predictor with proper parameters"""
#     model_weight = 'sam_b.pt'
#     overrides = dict(
#         task='segment',
#         mode='predict',
#         #imgsz=1024,
#         model=model_weight,
#         conf=0.01,
#         save=False
#     )
#     return SAMPredictor(overrides=overrides)

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
    model = YOLO("yolov8x-worldv2.pt")
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


def segment_image(image_path,output_mask='mask1.png'):
    """
    image_path: can be either a file path (str) or a numpy array (BGR image).
    output_mask: output mask file name.
    1) 用户可决定是否检测特定类别
    2) 调用 detect_objects 做初步检测
    3) 若 detections 存在，自动选最高分；否则让用户点击选择
    4) 用 SAM 分割并保存结果掩码
    5) 返回分割后的 mask (np.ndarray or None)
    """
    # 1) 用户输入 - 是否检测特定类别
    # use_target_class = input("Detect specific class? (yes/no): ").lower() == 'yes'
    # target_class = input("Enter class name: ").strip() if use_target_class else None
    # 用户直接输入类别
    target_class = input("\n"
                    "===============\n"
                    "Enter class name: ").strip()

    # 2) 初步检测 - YOLO
    detections, vis_img = detect_objects(image_path, target_class)
    # 保存检测可视化结果
    cv2.imwrite('detection_visualization.jpg', vis_img)
    # 3) 准备给 SAM 的图像 (RGB 格式)
    if isinstance(image_path, str):
        # 如果是字符串，说明是图像路径
        bgr_img = cv2.imread(image_path)
        if bgr_img is None:
            raise ValueError(f"Failed to read image from path: {image_path}")
        image_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    else:
        # 否则假设 image_path 就是一个 BGR 的 numpy 数组
        image_rgb = cv2.cvtColor(image_path, cv2.COLOR_BGR2RGB)

    # 4) 初始化 SAM predictor
    predictor = choose_model()
    predictor.set_image(image_rgb)

    # 5) 判断是否有目标检测结果
    if detections:
        # 自动选最高置信度
        best_det = max(detections, key=lambda x: x["conf"])
        results = predictor(bboxes=[best_det["xyxy"]])
        center, mask = process_sam_results(results)
        print(f"Auto-selected {best_det['cls']} with confidence {best_det['conf']:.2f}")
    else:
        # 手动点击
        print("No detections - click on target object")
        cv2.imshow('Select Object', vis_img)

        # 初始化全局变量
        point = []
        clicked = False
        def click_handler(event, x, y, flags, param):
            nonlocal clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at ({x}, {y})")
                point.extend([x, y])
                clicked = True  # 标记点击完成
        cv2.setMouseCallback('Select Object', click_handler)
        print("Waiting for user click...")
        # 循环等待点击或ESC键
        while not clicked:
            key = cv2.waitKey(10)  # 10ms延迟，减少CPU占用
            if key == 27:  # ESC键退出
                raise ValueError("User cancelled selection")
        cv2.destroyAllWindows()  # 安全关闭窗口

        if len(point) == 2:
            results = predictor(points=[point], labels=[1])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No selection made")

    # 6) 保存 mask
    if mask is not None:
        cv2.imwrite(output_mask, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        # print(f"Segmentation saved to {output_mask}")
    else:
        print("[WARNING] Could not generate mask")

    return mask


if __name__ == '__main__':
    seg_mask = segment_image('fruit.jpg')
    print("Segmentation result mask shape:", seg_mask.shape if seg_mask is not None else None)
