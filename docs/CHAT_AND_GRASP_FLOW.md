# Chat 与抓取流程说明

## 总览

当前系统分成两层：

1. Chat/任务调度层

   负责把用户自然语言任务转换成一组 HTTP API 调用。

   主要文件：

   ```text
   chat.py
   doubao.py
   doubao_recheck_v4.py
   exception_handling_agent.py
   ```

2. 抓取服务层

   负责相机图像、识别分割、点云、抓取生成、机械臂动作、动作复核和异常恢复。

   主要文件：

   ```text
   grasp_fastapi_completion_v4.py
   main_yoloWorld_sam_completion.py
   manipulator_grasp/env/ur5_grasp_env.py
   ```

如果你要把整套系统迁到另一台机器，先看：

- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

服务启动后，`UR5GraspEnv` 会加载：

```text
manipulator_grasp/assets/scenes/apple_pear_runtime_refined.xml
```

所以如果重新生成了虚拟环境，需要重启 `grasp_fastapi_completion_v4.py`。

## 推荐运行顺序

先生成虚拟场景：

```bash
python build_runtime_scene_from_sim_camera.py \
  --xml manipulator_grasp/assets/scenes/apple_pear_runtime_refined111_no_gripper.xml
```

再启动抓取服务：

```bash
python grasp_fastapi_completion_v4.py
```

然后可以用 curl 或 chat 调接口。

标准抓取链路：

```text
camera-image
detect-object
create-cloud
create-grasp
move-pregrasp
move-grasp
gripper-action(flag=1)
vertical-grasp
execute-grasp2
gripper-action(flag=0)
execute-init
gen-sorce
```

其中 `gen-sorce` 是异常处理/任务完成后的方案评分接口。

## Chat 调度层

### chat.py

`chat.py` 是一个简单的任务执行脚本。它的核心逻辑是：

1. 从 `renwu.json` 读取主任务动作序列。
2. 按顺序把每个 action 转成 curl 请求。
3. 如果某个接口返回 `status=failure`，则读取 `yichang.json` 执行异常处理动作。
4. 最后调用 `/gen-sorce` 评分。

当前 `chat.py` 对 detect 的参数读取是：

```python
res_json["parameters"]["object"]
```

然后请求：

```text
/detect-object?target_class=<object>
```

需要注意：新的异常恢复链路要求 detect 参数名为 `target_class`。因此如果你让模型直接生成任务 JSON，更推荐使用：

```json
{"action": "detect-object", "parameters": {"target_class": "pear"}}
```

如果继续使用 `chat.py` 当前写法，则 `renwu.json` 里 detect 动作需要保留：

```json
{"action": "detect-object", "parameters": {"object": "pear"}}
```

否则 `chat.py` 会取不到参数。

### doubao.py 与 exception_handling_agent.py

`doubao.py` 和 `exception_handling_agent.py` 负责更完整的 VLM 复核与异常恢复计划生成。

它们会使用：

```text
content_list_all
task_list
target
place_target_xy
```

这些运行时状态来判断当前任务执行到了哪里、哪里失败、下一步该怎么恢复。

恢复计划中的动作字段统一是：

```text
camera-image
detect-object
create-cloud
create-grasp
move-pregrasp
move-grasp
vertical-grasp
gripper-action
execute-grasp2
execute-init
```

异常恢复里要求：

```text
detect-object 参数名必须是 target_class
gripper-action 参数名必须是 state
execute-grasp2 必须带 x/y
```

## 抓取服务 API

所有主要接口在 `grasp_fastapi_completion_v4.py` 中。

### /camera-image

作用：

```text
从 cam1 渲染当前 RGB/depth 图像
```

输出文件：

```text
color_img1.jpg
color_img_depth1.jpg
```

同时会清理旧的识别、点云和抓取状态，并记录一次“获取相机图像”的 action evidence。

### /detect-object

作用：

```text
根据 color_img1.jpg 识别并分割目标物体
```

示例：

```bash
curl -X POST "http://localhost:8080/detect-object?target_class=pear"
```

当前支持中文规范化：

```text
梨 / 梨子 -> pear
苹果 -> apple
碗 -> bowl
```

输出：

```text
inputs/mask1.png
outputs/sim_groundingdino_mask1.png.jpg
outputs/sim_groundingdino_selected_mask1.png.jpg
```

成功后会设置全局：

```text
target
mask_img_path
```

### /create-cloud

作用：

```text
根据 color_img1.jpg、depth_img_path 和 mask_img_path 生成目标点云
```

成功后会设置：

```text
end_points
cloud_o3d
```

如果 mask 为空，会直接返回 failure。

### /create-grasp

作用：

```text
根据目标点云生成抓取姿势
```

核心函数：

```python
getGraspCompletion(color_img_path, depth_img_path, mask_img_path)
```

内部流程：

1. 调用 `get_and_process_data(...)` 得到目标原始点云。
2. 调用 `complete_point_cloud(...)` 做点云补全。
3. 优先生成 analytic top grasp。
4. 如果 analytic grasp 不可执行，则用 GraspNet 候选。
5. 用 `_select_surface_preferred_grasp(...)` 选择更适合顶部抓取的姿态。
6. 对目标类别应用静态抓取平移修正。
7. 返回单个 `GraspGroup`。

成功后会设置：

```text
grasp_gg
```

返回中包含：

```text
translation
rotation
width
score
height
depth
```

### /move-pregrasp

作用：

```text
移动到预抓取位姿
```

核心过程：

1. 调用 `init_world(grasp_gg)`。
2. 根据抓取姿态和当前相机位姿计算：

   ```text
   T_wo
   T_pregrasp
   ```

3. 调用 `extute_pre(content_list)` 执行关节和笛卡尔轨迹。
4. 构建图像和状态 evidence。
5. 调用 `_run_checked_action(...)` 做规则/VLM 复核。

如果 `grasp_gg` 不存在，会返回 failure。

### /move-grasp

作用：

```text
从预抓取位姿移动到真正抓取位姿
```

核心函数：

```python
exeute_grasp(content_list)
```

依赖：

```text
dot_pos
T_wo
T_pregrasp
```

如果还没有执行过 `/move-pregrasp`，`dot_pos` 为空，会返回 failure。

### /gripper-action

作用：

```text
控制夹爪打开或闭合
```

参数：

```text
flag=1 闭合
flag=0 打开
```

示例：

```bash
curl -X POST "http://localhost:8080/gripper-action?flag=1"
```

内部会逐步修改：

```python
action[-1]
```

并记录多相机图像用于复核。

### /vertical-grasp

作用：

```text
夹住物体后垂直提升
```

核心函数：

```python
vertical_lift(content_list)
```

它会把末端位姿向上偏移约 `0.3m`，并记录目标物体 z 方向变化：

```text
target_start_z
target_end_z
target_max_z
target_max_lift_dz
```

这些指标会进入规则检测和 VLM 复核。

### /execute-grasp2

作用：

```text
移动到指定放置位置并下降
```

参数：

```text
x
y
```

示例：

```bash
curl -X POST "http://localhost:8080/execute-grasp2?x=0.5&y=0.2"
```

内部会设置：

```text
place_target_xy
```

后续 `/gen-sorce` 会用这个目标位置评分。

### /execute-init

作用：

```text
回到初始位姿
```

成功后会清理运行时状态：

```text
color_img_path
depth_img_path
mask_img_path
target
end_points
cloud_o3d
grasp_gg
T_wo
T_pregrasp
dot_pos
history_grasps
place_target_xy
```

### /reset-environment

作用：

```text
只清理运行时状态，不重建 MuJoCo 环境
```

注意：它不会重新加载 XML。如果重新生成了场景 XML，需要重启服务。

### /gen-sorce

作用：

```text
评估异常处理或任务最终执行效果
```

内部调用：

```python
score_recovery(content_list_all, task_list, place_target_xy=get_current_place_target_xy())
```

## 动作复核与异常恢复

每个关键动作执行后都会调用：

```python
_run_checked_action(evidence, action_name, auto_recover=True, before_state=before_state)
```

它的流程是：

1. `capture_world_state(...)` 采集动作前后状态。
2. `detect_action(...)` 先做规则判断。
3. 如果规则结果是 `UNCERTAIN` 或 `FAILURE_SOFT`，调用 `recheck_action(...)` 做 VLM 图像复核。
4. 如果动作成功，写入：

   ```text
   task_list: <动作名>:SUCCESS
   content_list_all: action_evidence
   ```

5. 如果动作失败，写入：

   ```text
   task_list: <动作名>:FAILURE
   content_list_all: action_evidence
   ```

6. 如果 `auto_recover=True`，调用：

   ```python
   generate_recovery(content_list_all, task_list, target, place_target_xy=...)
   ```

   生成异常恢复方案。

## 关键全局状态

`grasp_fastapi_completion_v4.py` 使用较多全局状态：

```text
color_img_path      当前 RGB 图像
depth_img_path      当前深度图
mask_img_path       当前目标 mask
target              当前目标类别
end_points          GraspNet 输入
cloud_o3d           目标点云
grasp_gg            当前抓取结果
T_wo                抓取位姿
T_pregrasp          预抓取位姿
dot_pos             当前动作阶段的目标末端位姿
action              MuJoCo 控制向量，action[-1] 是夹爪信号
task_list           技能执行状态列表
content_list_all    动作证据历史，用于 VLM 复核和异常恢复
place_target_xy     放置目标位置
```

这些状态要求接口按顺序调用。跳过前置步骤通常会导致 failure。

## 常见失败点

### detect-object 失败

检查：

```text
color_img1.jpg
inputs/mask1.png
outputs/sim_groundingdino_mask1.png.jpg
outputs/sim_groundingdino_selected_mask1.png.jpg
```

如果 mask 为空，后续 `create-cloud` 和 `create-grasp` 都会失败。

### create-grasp 失败

常见原因：

```text
mask 为空
点云太少
点云补全失败
analytic grasp 不可执行
GraspNet 候选为空
```

补全失败时会 fallback 到 `getGrasp(...)`。

### move-pregrasp 失败

常见原因：

```text
grasp_gg 不存在
_select_tcp_mapping 找不到可执行 TCP 映射
IK 不可解
```

### move-grasp 失败

常见原因：

```text
没有先执行 move-pregrasp
dot_pos 为空
T_wo 为空
夹爪路径碰撞或未到位
```

### vertical-grasp 失败

常见原因：

```text
没有先闭合夹爪
空抓
提升时物体没有跟随夹爪上升
物体中途掉落
```

异常恢复里对 vertical-grasp 有特殊约束：

```text
如果失败原因是未闭合夹爪或缺少前置条件，
优先恢复为：
gripper-action(state=0) -> move-grasp -> gripper-action(state=1) -> vertical-grasp
```

在重新成功执行 `vertical-grasp` 前，不允许直接 `execute-grasp2`。

## 建议的手动调试命令

不经过 chat，直接验证抓取链路：

```bash
curl -X POST "http://localhost:8080/camera-image"
curl -X POST "http://localhost:8080/detect-object?target_class=pear"
curl -X POST "http://localhost:8080/create-cloud"
curl -X POST "http://localhost:8080/create-grasp"
curl -X POST "http://localhost:8080/move-pregrasp"
curl -X POST "http://localhost:8080/move-grasp"
curl -X POST "http://localhost:8080/gripper-action?flag=1"
curl -X POST "http://localhost:8080/vertical-grasp"
curl -X POST "http://localhost:8080/execute-grasp2?x=0.5&y=0.2"
curl -X POST "http://localhost:8080/gripper-action?flag=0"
curl -X POST "http://localhost:8080/execute-init"
curl -X POST "http://localhost:8080/gen-sorce"
```

如果手动 curl 成功但 chat 失败，问题通常在 `chat.py` 输出 JSON 的参数名或 action 名不匹配。

## chat.py 当前需要注意的点

`chat.py` 的系统提示里仍然写着：

```text
执行第一阶段的抓取：execute-grasp1
```

但当前服务实际接口是：

```text
move-pregrasp
move-grasp
```

因此如果要继续用 `chat.py` 自动生成任务，建议让任务 JSON 使用当前服务真实接口名，而不是 `execute-grasp1`。

另外，`chat.py` 的 detect 参数读取是：

```python
parameters["object"]
```

而异常恢复系统使用：

```python
parameters["target_class"]
```

这两个约定不完全统一。手写 `renwu.json` 时要按 `chat.py` 当前代码使用 `object`；如果后续统一接口，建议把 `chat.py` 改为同时兼容：

```text
target_class / object / target
```

这样和异常恢复链路更一致。
