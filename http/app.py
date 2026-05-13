from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
import tempfile
from cv_process import detect_objects, choose_model, process_sam_results
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.models.sam import Predictor as SAMPredictor
import os
app = FastAPI(title="YOLOWorld + SAM HTTP Service")

@app.post("/detect/")
async def detect_objects_api(
    file: UploadFile = File(...),
    target_class: str = Form(None)
):
    """
    上传图片并执行 YOLOWorld 检测（可选：指定类别）
    """
    # 将上传的文件保存到临时路径
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        detections, vis_img = detect_objects(tmp_path, target_class)
        path = os.path.join("imgs",file.filename)
        cv2.imwrite(path,vis_img)
        # 将可视化结果转为base64返回
        _, buffer = cv2.imencode('.jpg', vis_img)
        import base64
        vis_base64 = base64.b64encode(buffer).decode('utf-8')
        '''return {
            "status":"成功"
        }'''
        return JSONResponse({
            "status": "success",
            "target_class": target_class,
            "detections": detections
            #"visualization": vis_base64
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

@app.post("/segment/")
async def segment_api(
    file: UploadFile = File(...),
    target_class: str = Form(None)
):
    """
    上传图片并执行 YOLOWorld + SAM 分割
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # YOLO 目标检测
        detections, vis_img = detect_objects(tmp_path, target_class)
        predictor: SAMPredictor = choose_model()
        path = os.path.join("imgs",file.filename)
        cv2.imwrite(path,vis_img)
        # 读取图像并转RGB
        image_bgr = cv2.imread(tmp_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)

        # 选择最高置信度检测框进行分割
        if detections:
            best_det = max(detections, key=lambda x: x["conf"])
            results = predictor(bboxes=[best_det["xyxy"]])
            center, mask = process_sam_results(results)
        else:
            raise ValueError("No detections found for segmentation")
        if mask is not None:
            base, ext = os.path.splitext(file.filename)
            output_path = f"{base}-mask{ext}"
            path = os.path.join("imgs",output_path)
            cv2.imwrite(path, mask, [cv2.IMWRITE_PNG_BILEVEL, 1])
        # 转为base64
        _, buffer = cv2.imencode('.png', mask)
        import base64
        mask_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # print(f"Segmentation saved to {output_mask}")
        return JSONResponse({
            "status": "success",
            "target_class": target_class,
            "center": center
            #"mask": mask_base64
        })
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
