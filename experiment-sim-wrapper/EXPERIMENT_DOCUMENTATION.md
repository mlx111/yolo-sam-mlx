# 抓取异常实验 v5 — 完整文档

## 1. 实验目标

在 MuJoCo 仿真环境中模拟苹果抓取，注入异常（抓空），然后通过**感知层**（YOLO + SAM2 + 点云）检测异常，执行恢复抓取，生成可迁移到真机的恢复方案。

**核心原则：**
- 异常检测只用感知层数据，**不用 MuJoCo 真值**（模拟真机条件）
- 仿真套仿真结构：Layer 1（真实仿真）= 真机的替代品，Layer 2（感知层）= 真机上通过视觉能看到的东西
- 所有检测和决策只允许用 Layer 2 的数据

---

## 2. 架构：双层设计

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: 真实仿真环境 (MuJoCo) primary scene                 │
│  ┌──────────────────────────────────────────────────────────┐│
│  │  UR5e + 2F-85 gripper + 苹果 + 桌面                     ││
│  │  物理引擎在这里跑，异常在这里注入                          ││
│  └──────────────────────────────────────────────────────────┘│
│                            ↓                                  │
│  ┌──── 条件 A 路径 ───────┐    ┌── 条件 B 路径 ────────────┐│
│  │ 渲染相机 → 视觉感知     │    │ 渲染相机 → 视觉感知       ││
│  │ (YOLO+SAM2+点云+位姿)   │    │ (YOLO+SAM2+点云+位姿)     ││
│  │         ↓               │    │         ↓                  ││
│  │ Layer 2: 虚拟仿真场景   │    │ Layer 2: 感知场景          ││
│  │ (新建 MuJoCo 实例)      │    │ (感知状态下直接恢复)       ││
│  │ 在虚拟仿真中规划恢复     │    │                            ││
│  │ 生成恢复方案 → 迁移     │    │                            ││
│  └──────────────────────────┘    └───────────────────────────┘│
│                            ↓                                  │
│  生成恢复方案 (JSON): 关节轨迹 + 抓取位姿 + 夹爪指令          │
└──────────────────────────────────────────────────────────────┘
                            ↓
┌──────────────────────────────────────────────────────────────┐
│  真机回放 (未来目标)                                          │
│  加载恢复方案 → 适配当前视觉 → 执行                           │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 实验流程（9 步）

```
Step 1: 关节运动到 q1          (肩部摆正)
Step 2: 笛卡尔到预抓取位姿     (苹果正上方 12.7cm)
Step 3: 笛卡尔到抓取位姿       (苹果位置)
Step 4: 同步关节 + 闭合夹爪    (夹住苹果)
  └── 感知基线: 通过 YOLO+SAM2 获取提起前苹果位置 (perceived_z_before_lift)
Step 5: 笛卡尔提起 30cm        (提起苹果)
Step 6: 注入异常 (grasp_miss)  (苹果重置到桌面)
Step 7: 感知管线检测异常        (YOLO+SAM2+点云 → 判断 Z 变化)
Step 8: 恢复抓取               (条件 A 或 B)
Step 9: 回到初始位置           (回到 q1 + 打开夹爪)
```

### 3.1 提起前感知基线

在 Step 3（移动到抓取位姿）之后、Step 4（闭合夹爪）之前，通过感知管线获取苹果的位置作为基线。这个基线是后续异常检测的依据。

- 此时苹果未被遮挡（夹爪还没闭合）
- 感知位置保存到 `metrics["perceived_z_before_lift"]`
- 异常检测通过比较 `perceived_z_after_inject` 与此基线的差值判断

### 3.2 异常检测逻辑

```python
if perception_returns_detection_ok:
    # 感知成功: 比较提起前后 Z 变化
    z_change = perceived_z_after - perceived_z_before
    if z_change < 0.03m → 异常 (苹果没被提起)
    else → 无异常
else:
    # 感知失败: 回退到 MuJoCo 真值 (仿真专用安全网)
    gt_z_change = apple_z_after_inject - apple_z_before_lift
    if gt_z_change < 0.03m → 异常
    else → 无异常
```

当感知失败时，使用 MuJoCo 真值作为回退。这是仿真专用的安全网——在真机上需要额外的力/触觉传感器来区分"提起成功"和"感知失败"。

---

## 4. 两个实验条件

### 4.1 条件 A: sim_wrapper (虚拟仿真迁移)

```
异常注入后 Layer 1 (真实仿真)
  → 渲染相机 → 感知管线 → 估计位姿
  → 新建 MuJoCo 场景 (Layer 2 / 虚拟仿真)
  → 在 Layer 2 中规划恢复（跑完整 6 步流程）
  → 记录恢复方案（关节轨迹 + Cartesian 位姿）
  → 将方案步骤迁移到 Layer 1 执行
  → 评估迁移成功率
```

特点：
- 恢复流程在**独立的虚拟仿真环境**中规划
- 虚拟场景中的苹果位置是**感知位置**（不是真值）
- 规划成功后返回步骤列表，在真实仿真中执行
- 虚拟场景可开启独立 viewer（与主场景同步显示，便于观察虚拟场景内部运行）
- 恢复耗时 ~3.3s（含虚拟场景构建和规划）
- 恢复后回到 q1 初始位置，张开夹爪

**使用方式：**
```bash
python run_experiment_v4.py --condition sim_wrapper --save-plan plan.json
```

### 4.2 条件 B: direct (直接恢复)

```
异常注入后 Layer 1 (真实仿真)
  → 渲染相机 → 感知管线 → 估计位姿
  → 直接用感知位置调整抓取位姿 T_wo
  → 在 Layer 1 中执行恢复（6 步）
  → 记录恢复方案
```

特点：
- 直接在真实仿真中执行恢复
- 使用感知位置（如果感知失败则回退真值）
- 恢复耗时 ~1.1s
- 恢复后回到 q1 初始位置，张开夹爪

**使用方式：**
```bash
python run_experiment_v4.py --condition direct --save-plan plan.json
```

---

## 5. 代码文件

| 文件 | 行数 | 用途 |
|------|------|------|
| `run_experiment_v4.py` | ~650 | 主实验脚本：流程编排、异常检测、恢复执行、方案记录 |
| `sim_wrapper.py` | ~315 | 条件 A 核心：虚拟场景构建、虚拟场景中规划恢复 |
| `perception_pipeline.py` | ~245 | 感知管线封装：渲染 → YOLO+SAM2 → 点云 → 位姿 |
| `anomaly_injectors.py` | ~92 | 异常注入：grasp_miss、object_displaced、gripper_fail |
| `replay_plan.py` | ~260 | 恢复方案回放器：加载 JSON → 新 MuJoCo 实例中逐步骤执行 |
| `recovery_schema.json` | (可选) | 恢复方案 JSON schema |

### 5.1 run_experiment_v4.py

主实验类 `ExperimentV4`:

```
__init__()        → 加载场景、初始化 UR5e、附接夹爪、加载抓取位姿
run()             → 执行 8 步流程
  _move_joints()       → 关节空间轨迹
  _move_cartesian()    → 笛卡尔空间轨迹
  _gripper_close()     → 闭合夹爪
  _snap_to_robot_joints() → 消除 MuJoCo 稳态误差
  _perceive_before_lift() → 提起前感知基线
  _detect_anomaly()    → YOLO+SAM2+点云 异常检测
  _execute_recovery()  → 根据条件分支选择恢复策略
    _condition_a_recovery()   → 虚拟仿真规划 → 迁移执行
    _direct_recovery()        → 直接恢复
    _execute_plan_steps()     → 执行方案步骤
  _save_recovery_plan() → 恢复方案序列化为 JSON
```

**CLI 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-viewer` | False | 不启动 MuJoCo viewer |
| `--no-inject` | False | 不注入异常（对照组） |
| `--save PATH` | results/result_v4.json | 结果保存路径 |
| `--trials N` | 1 | 重复实验次数 |
| `--seed N` | None | 随机种子 |
| `--perturb M` | 0.01 | 每次试验的位置随机扰动 (m) |
| `--condition` | direct | 恢复策略: direct / sim_wrapper |
| `--noise-scale` | 0.0 | 感知噪声幅度 (m) |
| `--save-plan PATH` | None | 保存恢复方案到 JSON |
| `--save-plan` | None | 保存恢复方案到指定路径 |

### 5.2 sim_wrapper.py

`SimWrapper` 类:

```
build_virtual_scene(perceived_pos, perceived_quat, enable_viewer=False)
  → 创建新 MuJoCo 实例，苹果放在感知位置（非真值）
  → enable_viewer=True 时开启独立 viewer 窗口（观察虚拟场景内部运动）
  → 初始化 UR5e + 夹爪
  → 返回 VirtualScene handle

plan_recovery_in_virtual(virtual_scene, recovery_pos)
  → 在虚拟场景中执行完整恢复流程
  → 记录每一步到 steps 列表
  → 返回 (success, steps)

capture_state(experiment) / restore_state(experiment, state)
  → 保存/恢复实验状态（用于条件 B 的候选位置验证）
```

### 5.3 perception_pipeline.py

`PerceptionPipeline` 类:

```
detect(target_class, full_pose, work_dir) → PerceivedScene
  1. render_rgb_depth() → 从 MuJoCo 相机渲染 RGB + 深度
  2. segment_image_ground() → YOLO-World + SAM2 分割
  3. HSV 前景检查 → 过滤与苹果颜色零重叠的误检掩码
  4. backproject_masked_depth() → 点云重建
  5. 点云 Z 方差检查 → 过滤覆盖多物体的过大掩码
  6. robust_position() → 裁剪平均位置估计
```

**返回: `PerceivedScene`:**
```
  .apple_pos      → (3,) 世界坐标系位置（仅感知）
  .apple_quat     → (4,) wxyz 四元数（可选）
  .confidence     → 置信度
  .detection_ok   → 检测是否成功
```

### 5.4 anomaly_injectors.py

三种异常类型：

| 类型 | 函数 | 效果 |
|------|------|------|
| 抓空 | `inject_grasp_miss()` | 提起后苹果位置重置回桌面初始位置，速度归零 |
| 物体位移 | `inject_object_displaced()` | 物体位置加偏移 [dx, dy, dz] |
| 夹爪故障 | `inject_gripper_fail()` | 夹爪手指强制保持张开位 |

异常只在 Layer 1（真实仿真）中注入，对 Layer 2（感知层）不可见。

### 5.5 replay_plan.py

`PlanReplayer` 类:

```
load(plan_path)        → 加载恢复方案 JSON
execute()              → 逐步骤执行（joint_move / gripper / cartesian_move）
支持 --perturb 参数    → 苹果位置扰动，测试方案鲁棒性
```

**使用方式：**
```bash
python replay_plan.py --plan /tmp/recovery_plan.json
python replay_plan.py --plan plan.json --no-viewer
python replay_plan.py --plan plan.json --perturb 0.02 0.01 0.0
```

---

## 6. 恢复方案 JSON Schema

```json
{
  "plan_id": "recovery_20260504_133514_6342",
  "condition": "direct | sim_wrapper",
  "anomaly_type": "grasp_miss",
  "detection_info": {
    "perceived_apple_z": 0.0446,
    "z_threshold": 0.03,
    "detection_method": "perception_z_check | perception_failed"
  },
  "steps": [
    {"type": "joint_move", "target": [6 joints], "duration": 1.0},
    {"type": "gripper", "command": "open"},
    {"type": "cartesian_move", "target_pos": [x,y,z], "target_rot": [3x3], "duration": 1.0, "label": "pregrasp"},
    {"type": "cartesian_move", "target_pos": [x,y,z], "target_rot": [3x3], "duration": 1.0, "label": "grasp"},
    {"type": "gripper", "command": "close"},
    {"type": "cartesian_move", "target_pos": [x,y,z], "target_rot": [3x3], "duration": 1.0, "label": "lift"}
  ],
  "result": {
    "success": true,
    "apple_z_after_recovery": 0.312
  }
}
```

**步骤类型：**
- `joint_move` → 关节空间运动到 q_target
- `gripper` → 夹爪打开/关闭
- `cartesian_move` → 笛卡尔空间运动到目标位姿

恢复方案的设计目标：可在新 MuJoCo 实例中回放，也**可在真机上回放**（方案只包含运动指令，不包含仿真特定数据）。

---

## 7. 运行方式

### 7.1 单次实验

```bash
# 条件 B (直接恢复)
conda run -n mujoco1 python run_experiment_v4.py \
  --condition direct --save-plan /tmp/plan.json

# 条件 A (虚拟仿真迁移)
conda run -n mujoco1 python run_experiment_v4.py \
  --condition sim_wrapper --save-plan /tmp/plan.json

# 对照组 (不注入异常)
conda run -n mujoco1 python run_experiment_v4.py \
  --condition direct --no-inject

# 启用 viewer
conda run -n mujoco1 python run_experiment_v4.py \
  --condition direct --save-plan /tmp/plan.json
```

### 7.2 批量实验

```bash
# 10 次试验，统计数据
conda run -n mujoco1 python run_experiment_v4.py \
  --condition direct --trials 10 --seed 42 --save results/batch.json
```

### 7.3 回放验证

```bash
# 加载方案并回放
conda run -n mujoco1 python replay_plan.py \
  --plan /tmp/plan.json

# 加载方案 + 位置扰动测试鲁棒性
conda run -n mujoco1 python replay_plan.py \
  --plan /tmp/plan.json --perturb 0.02 0.01 0.0
```

---

## 8. 实验结果指标

输出 JSON 包含以下关键指标：

| 字段 | 说明 |
|------|------|
| `condition` | direct / sim_wrapper |
| `anomaly_detected` | 是否检测到异常 |
| `detection_method` | perception_z_check / perception_failed |
| `recovery_success` | 恢复是否成功 (apple dz > 3cm) |
| `apple_z_before_lift` | 提起前苹果真值 Z (参考) |
| `apple_z_after_lift` | 提起后苹果真值 Z (参考) |
| `apple_z_after_inject` | 异常注入后苹果真值 Z |
| `apple_z_after_recovery` | 恢复后苹果真值 Z |
| `perceived_z_before_lift` | 提起前感知 Z |
| `perceived_z_after_inject` | 注入后感知 Z (None 表示感知失败) |
| `perceived_position` | 感知到的苹果位置 [x,y,z] |
| `observed_pos` | 恢复使用的实际位置 |
| `contact_after_close` | 闭合后夹爪接触状态 |
| `contact_after_lift` | 提起后夹爪接触状态 |
| `time_costs.detection_perception` | 感知耗时 (s) |
| `time_costs.pre_recovery` | 异常检测前总耗时 |
| `time_costs.recovery` | 恢复耗时 (s) |
| `time_costs.total` | 总耗时 |
| `plan_saved` | 恢复方案保存路径 |

### 典型结果对比

| 指标 | 条件 B (direct) | 条件 A (sim_wrapper) |
|------|----------------|---------------------|
| 异常检测率 | 100% (inject) / 0% (control) | 100% (inject) / 0% (control) |
| 恢复成功率 | 100% | 100% |
| 恢复后苹果 Z | ~0.312m | ~0.317m |
| 恢复耗时 | ~1.1s | ~3.3s |

---

## 9. 回放结果指标

`replay_plan.py` 执行后输出：

| 字段 | 说明 |
|------|------|
| `success` | 回放是否成功 (apple dz > 3cm) |
| `apple_z_after_replay` | 回放后苹果 Z |
| `z_change` | 苹果 Z 变化量 |
| `plan_id` | 方案 ID |

---

## 10. 已知限制

1. **感知管线在提起后可能失败** — 当机械臂处于提起位置时，YOLO+SAM2 有时会产生零 Apple-颜色重叠的掩码，被 HSV 前景过滤器拒绝。此时回退到真值检测（仿真专用安全网）。真机需要力/触觉传感器补充。

2. **只支持 grasp_miss 异常类型** — 目前实验主要测试 grasp_miss。object_displaced 和 gripper_fail 有注入函数但未集成到主实验流程。

3. **场景固定** — 使用固定的 `apple_pear_runtime_refined.xml` 场景，苹果初始位置固定。可通过 `--perturb` 参数引入随机偏移。

4. **无真机验证** — 恢复方案可回放验证，但目前仅在仿真环境中回放。真机回放是未来工作。

5. **位置扰动范围有限** — `--perturb` 参数只在苹果初始位置上添加小范围随机偏移，不测试大幅度的场景变化。

---

## 11. 文件位置

```
experiment-sim-wrapper/
├── run_experiment_v4.py          # 主实验脚本
├── sim_wrapper.py                # Condition A: 虚拟仿真
├── perception_pipeline.py        # 感知管线封装
├── anomaly_injectors.py          # 异常注入
├── replay_plan.py                # 恢复方案回放器
├── Grounded-SAM-2/               # YOLO+SAM2 检测代码
├── yolov8x-worldv2.pt           # YOLO-World 模型权重
├── results/
│   ├── result_v4.json           # 旧版 v4 结果
│   ├── result_direct_final.json # 条件 B 最终结果
│   └── result_sim_wrapper_final.json # 条件 A 最终结果
└── EXPERIMENT_DOCUMENTATION.md   # 本文档
```
