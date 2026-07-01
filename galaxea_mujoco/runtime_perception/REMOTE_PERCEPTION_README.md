# 远程感知服务 — 使用说明

让机器人通过 HTTP 调用电脑上的 GroundedSAM2 进行物体识别和 3D 定位。

## 架构概览

```
┌─────────────────────────┐          HTTP / multipart          ┌──────────────────────┐
│  机器人 (真机)           │  ──── POST /detect_pose ──────▶   │  你的电脑 (服务端)     │
│                          │  ◀────── JSON result ────────    │                      │
│  remote_perception_     │                                   │  remote_perception_  │
│  client.py              │                                   │  server.py           │
│                          │                                   │  ├ GroundedSAM2      │
│  RealSense → RGB + depth│                                   │  ├ PointCloudGenerator│
│  ↓ 上传                  │                                   │  └ 返回 3D 位置       │
└─────────────────────────┘                                   └──────────────────────┘
```

**核心思路：** 模型（GroundedSAM2）跑在你的电脑上，机器人只需把 RGB 和 depth 图通过 HTTP 传过来，就能拿到物体的 3D 坐标（米）。机器人端不需要 GPU，不需要装 PyTorch，一个 `requests` 就够了。

---

## 文件说明

| 文件 | 用途 | 部署位置 |
|---|---|---|
| `remote_perception_server.py` | FastAPI 服务端，加载模型、接收请求 | **你的电脑**（有 GPU） |
| `remote_perception_client.py` | 客户端 SDK + CLI，拍照、上传、拿结果 | **机器人** |

---

## 一、服务端部署（在你的电脑上）

### 1.1 启动服务

```bash
cd /home/lt/malanxuan/yolo-sam-mlx

# 用 conda mujoco1 环境启动
nohup /home/lt/anaconda3/envs/mujoco1/bin/python -m uvicorn \
  galaxea_mujoco.runtime_perception.remote_perception_server:app \
  --host 0.0.0.0 --port 8088 \
  > /tmp/perception_server.log 2>&1 &
```

参数说明：

| 参数 | 说明 |
|---|---|
| `--host 0.0.0.0` | 监听所有网络接口，局域网内的机器人都能访问 |
| `--port 8088` | 端口号，可按需修改 |
| `nohup ... &` | 后台运行，关掉终端也不会停 |

### 1.2 验证服务状态

```bash
curl -s http://localhost:8088/health | python -m json.tool
```

预期输出（关键字段）：

```json
{
    "status": "ok",
    "model_loaded": true,
    "camera_intrinsics": { "fx": 385.78, "fy": 385.22, "cx": 327.58, "cy": 238.69 },
    "expected_rgb_shape": [480, 640],
    "expected_depth_shape": [480, 640],
    "depth_unit": "uint16_mm"
}
```

`model_loaded: true` 表示 GroundedSAM2、GroundingDINO、BERT 全部加载成功。

### 1.3 查看日志

```bash
tail -f /tmp/perception_server.log
```

### 1.4 停止服务

```bash
kill <PID>
# PID 在启动时打印，或者用：
ps aux | grep uvicorn | grep perception_server
```

---

## 二、服务端 API 说明

### `GET /health`

健康检查。返回模型状态、内外参、期望的图像尺寸。

### `POST /detect_pose`

核心接口。接收 RGB + depth 图，返回物体 3D 位置。

**请求格式：** `multipart/form-data`

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `rgb` | file | 是 | JPEG 或 PNG 彩色图，640×480 |
| `depth` | file | 是 | uint16 PNG 深度图，640×480，单位 mm，0=无效 |
| `target_class` | string | 是 | 要识别的物体名称，如 `"red box"`、`"apple"` |
| `coordinate_system` | string | 否 | `"camera"`/`"base"`/`"world"`，默认 `"world"` |

**成功响应示例：**

```json
{
    "success": true,
    "target_class": "red box",
    "coordinate_system": "world",
    "position_m": [0.520, -0.056, 0.783],
    "position_raw_mm": [520.24, -56.44, 782.99],
    "bbox_xyxy": [318.7, 430.2, 345.1, 465.8],
    "mask_pixel_count": 795,
    "valid_depth_count": 781,
    "depth_unit": "uint16_mm",
    "point_count": 777,
    "candidate": {
        "score": 0.785,
        "label": "single red box object",
        "detector": "grounding_dino"
    }
}
```

**失败响应示例：**

```json
{
    "success": false,
    "target_class": "unicorn",
    "coordinate_system": "world",
    "error": "Grounded-SAM2 did not detect: unicorn"
}
```

---

## 三、客户端使用（在机器人上）

把 `galaxea_mujoco/runtime_perception/remote_perception_client.py` 拷贝到机器人上即可使用。

### 3.1 机器人端依赖

```bash
pip install requests opencv-python numpy
# 如果要用 RealSense 实时拍照，还需要：
pip install pyrealsense2
```

### 3.2 方式一：命令行（快速验证）

```bash
# 用 RealSense 实时拍照检测
python remote_perception_client.py \
  --server http://192.168.1.100:8088 \
  --target-class "red box"

# 用已有的图片文件检测（不需要 RealSense）
python remote_perception_client.py \
  --server http://192.168.1.100:8088 \
  --target-class "red box" \
  --rgb /path/to/color.png \
  --depth /path/to/depth.png

# 指定坐标系
python remote_perception_client.py \
  --server http://192.168.1.100:8088 \
  --target-class "apple" \
  --coordinate-system "camera"
```

命令行参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--server` | `http://127.0.0.1:8088` | 服务端地址 |
| `--target-class` | `red box` | 检测目标 |
| `--coordinate-system` | `world` | `camera`/`base`/`world` |
| `--rgb` | — | 彩色图路径（不传则用 RealSense 拍照） |
| `--depth` | — | 深度图路径（不传则用 RealSense 拍照） |
| `--timeout` | `60.0` | HTTP 超时秒数 |

### 3.3 方式二：作为 SDK 集成到你的代码里

```python
from remote_perception_client import RemotePerceptionClient, capture_realsense

# 在整个程序生命周期只创建一次（模型在服务端常驻）
client = RemotePerceptionClient(
    server_url="http://192.168.1.100:8088",
    timeout_seconds=60.0,
)

# ----- 场景 A：用 RealSense 拍照 -----
rgb, depth = capture_realsense()   # 返回 (480,640,3) uint8 + (480,640) uint16

# ----- 场景 B：用你自己的图（比如 ROS topic 订阅来的） -----
# 你只需要保证：
#   rgb.shape   == (480, 640, 3),  dtype=np.uint8,  BGR 格式
#   depth.shape == (480, 640),     dtype=np.uint16, 单位 mm, 0=无效

# 调用检测
result = client.detect_object(
    rgb_img=rgb,
    depth_img=depth,
    target_class="red box",
    coordinate_system="world",
)

# 处理结果
if result.success:
    x, y, z = result.position_m            # [米]
    print(f"物体位置: ({x:.3f}, {y:.3f}, {z:.3f}) m")
    print(f"置信度:   {result.candidate_score:.3f}")
    print(f"Bbox:     {result.bbox_xyxy}")
else:
    print(f"检测失败: {result.error}")
```

### 3.4 方式三：集成到抓取工作流

```python
from remote_perception_client import RemotePerceptionClient, capture_realsense
from use_arm_control_execute_trajectory_v3.1 import R1ProArmController, Pose
import time

# 初始化（只做一次）
client = RemotePerceptionClient("http://192.168.1.100:8088")
arm = R1ProArmController()

# 检测 → 抓取循环
while True:
    # 1. 拍照
    rgb, depth = capture_realsense()

    # 2. 检测
    result = client.detect_object(rgb, depth, "red box")
    if not result.success:
        print("未检测到目标，稍等后重试...")
        time.sleep(2)
        continue

    # 3. 移动到物体上方 15 cm
    x, y, z = result.position_m
    approach_pose = Pose(x=x, y=y, z=z + 0.15, roll=0.0, pitch=1.57, yaw=0.0)
    arm.move_to_pose(approach_pose, arm_name="left")

    # 4. 下去抓
    grasp_pose = Pose(x=x, y=y, z=z + 0.02, roll=0.0, pitch=1.57, yaw=0.0)
    arm.move_to_pose(grasp_pose, arm_name="left")

    # 5. 闭合夹爪
    arm.gripper_close(arm_name="left")
    break
```

### 3.5 客户端 API 参考

```python
class RemotePerceptionClient:
    def __init__(self, server_url: str, timeout_seconds: float = 60.0)

    # 健康检查
    def health() -> dict

    # 核心检测方法
    def detect_object(
        self,
        rgb_img: np.ndarray,          # (480,640,3) uint8 BGR
        depth_img: np.ndarray,        # (480,640) uint16 mm
        target_class: str,            # "red box", "apple", ...
        coordinate_system: str = "world",  # camera/base/world
        rgb_format: str = "jpg",      # jpg(更快) 或 png(无损)
        jpeg_quality: int = 90,       # 仅 jpg 时有效
    ) -> DetectionResult
```

```python
class DetectionResult:
    success: bool                     # 是否检测成功
    target_class: str                 # 目标物体名称
    coordinate_system: str           # 坐标系
    position_m: list[float] | None   # [x, y, z] 米 ← 直接给机械臂用
    position_raw_mm: list[float]     # [x, y, z] 毫米
    bbox_xyxy: list[float] | None    # 像素 bbox [x1,y1,x2,y2]
    mask_pixel_count: int            # 分割掩码像素数
    valid_depth_count: int           # 有效深度像素数
    point_count: int                 # 3D 点数
    candidate_score: float           # 检测置信度 0~1
    candidate_label: str             # 匹配到的文本提示
    error: str | None                # 失败原因
    raw: dict                        # 原始 JSON 响应
```

辅助函数：

```python
def capture_realsense(
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    timeout_sec: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """从 Intel RealSense 拍摄一帧对齐的 RGB + Depth。
    返回 (bgr_image, depth_image)。
    """
```

---

## 四、约定的协议（必须遵守）

以下约定服务端和客户端之间必须一致，否则结果错误：

| 项目 | 约定值 |
|---|---|
| depth 格式 | 16-bit PNG |
| depth dtype | `uint16` |
| depth 单位 | **毫米**（millimeter） |
| depth 无效值 | `0` |
| RGB shape | `640 × 480` |
| depth shape | `640 × 480` |
| 相机内参 | `fx=385.78, fy=385.22, cx=327.58, cy=238.69` |
| 相机外参 | 见服务端 `/health` 返回的 `camera_extrinsics` |

> 如果真机换了相机，只需要修改 `remote_perception_server.py` 和 `remote_perception_client.py` 头部对应的内外参常量即可。

---

## 五、网络拓扑建议

```
                   局域网 (192.168.1.x)
                        │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
    ▼                    ▼                    ▼
  你的电脑               机器人             手机/调试机
  192.168.1.100         192.168.1.233       任意 IP
  :8088                 调用客户端            curl 调试
```

- 保证电脑和机器人在同一网段
- 如果机器人无法直连电脑，检查防火墙：`sudo ufw allow 8088`
- 建议用有线网络（USB 转网口也行），Wi-Fi 在高负载下可能有延迟

---

## 六、故障排查

### 服务端起不来

```bash
# 查看启动日志
tail -f /tmp/perception_server.log

# 常见原因：
#   1. 端口占用 → 换端口： --port 8089
#   2. 模型路径不对 → 检查 ../Grounded-SAM-2 是否存在及其目录结构
#   3. CUDA OOM   → 关掉其他占用显存的程序
```

### 客户端连不上服务端

```bash
# 在机器人上测试连通性
ping 192.168.1.100
curl http://192.168.1.100:8088/health
```

### 检测结果不准

```bash
# 检查上传的 depth 图是否正确
python -c "
import cv2, numpy as np
d = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED)
print(f'shape={d.shape}, dtype={d.dtype}, min={d.min()}, max={d.max()}')
"
# 预期: shape=(480, 640), dtype=uint16, min>=0, max>0
```
