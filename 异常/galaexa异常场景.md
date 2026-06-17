# Galaxea 异常场景实验设计

## 1. 设计目标

本文档面向 Galaxea R1Pro 类移动操作机器人，不沿用 UR5E 的固定桌面单臂设定，而是把机器人能力拆成以下三层来设计异常：

1. 移动底盘与停位
2. 躯干与臂的全身可达性
3. 双臂协同、夹爪接触和多目标感知

Galaxea 的实验重点不是“一个桌面点位是否能抓到”，而是“机器人能否在室内任务区间移动、对准、抓取、搬运、放置，并在异常后恢复到正确任务策略”。因此，场景应尽量体现：

```text
移动底盘 + 4-DOF 躯干 + 双 7-DOF 机械臂 + 双夹爪 + 多视角感知
```

当前文档将 Galaxea 的异常细化为 4 个场景，每个场景 5 个异常条件。建议在实验数据中保留两级标签：

```text
scenario_id: 场景编号，例如 G1
condition_id: 具体异常条件，例如 G1-3
anomaly_type: 代码层异常标签，例如 target_displaced / blocked_reach / wrong_object / gripper_fail
```

这样做的好处是：

1. 同一个 `anomaly_type` 可以在不同场景中复用。
2. 论文统计可以按场景聚合，也可以按条件展开。
3. 经验库检索可以同时利用“异常名称”和“任务阶段”。

## 2. 机器人与任务定位

Galaxea 相关代码和模型表明，它不是单纯的桌面机械臂，而是一个具备移动底盘、躯干、双臂、双夹爪和视觉传感器的室内操作平台。对应本地代码能力包括：

```text
wheel_move.py
whole_body_move.py
follow_target.py
r1pro_experiment_smoke.py
r1pro_memory_benchmark.py
r1pro_llm_strategy_benchmark.py
```

因此，Galaxea 的异常场景应围绕以下任务区设计：

```text
货架区
抽屉柜区
杂乱桌面区
转运/放置区
```

建议任务对象不要只用单一 cube，而是引入可区分的多类物体，例如：

```text
盒子、瓶子、杯子、工具件、长条盒、托盘
```

这样更容易制造“错误对象”“遮挡”“堆叠”“滑落”“双臂不同步”等异常。

## 3. 场景 G1：多层货架取物

### 3.1 场景定义

机器人需要移动到底盘停位点，调整躯干高度和朝向，从三层或两层货架中取出指定物体，放到工作台或转运车上。

这个场景最适合测试：

- 移动停位误差
- 躯干高度选择
- 远距离可达性
- 视觉定位更新
- 抓取后高位搬运稳定性

建议把该场景的主要异常类型统一映射为：

```text
target_displaced
blocked_reach
wrong_object
grasp_miss
collision
```

### 3.2 5 个异常条件

| condition_id | 异常情况 | 注入阶段 | 失败信号 | 期望恢复动作 |
|---|---|---|---|---|
| G1-1 | 目标从中层被移到上层 | 视觉定位后、规划前 | 原抓取高度不够，末端停在目标下方 | 重新检测目标层级，抬高躯干或改用更高预抓位 |
| G1-2 | 目标从左侧货架移到右侧相邻货位 | 规划前 | 机器人到达旧货位后找不到目标 | 重新感知货架坐标，更新目标位姿与横向停位 |
| G1-3 | 货架前出现窄障碍或临时挡板 | 移动到底盘前 | 直线接近失败或末端路径被挡 | 改为绕行停位，先完成底盘重定位再抓取 |
| G1-4 | 目标被相似物部分遮挡 | 视觉感知阶段 | bbox/分割不稳定，抓取点跳变 | 切换多视角复核，重新生成 grasp |
| G1-5 | 抓取后物体从货架边缘滑落 | close 后 / lift 中 | 物体 z 下降，或相对夹爪距离增大 | 先停止提升，回到当前货位重新闭合或改用更深抓取位 |

### 3.3 场景成功判据

```text
robot reaches correct shelf region
and selected target is the intended object
and object is lifted or removed from shelf without collision
and robot transitions to safe transport pose
```

### 3.4 场景价值

这个场景能体现 Galaxea 的“长程定位 + 纵向可达性 + 搬运稳定性”，比 UR5E 更接近真实移动操作任务。

## 4. 场景 G2：抽屉/柜门取物

### 4.1 场景定义

机器人需要先打开抽屉或柜门，再从内部取出目标物体，最后把物体放到外部容器或工作台。

这个场景最适合测试：

- 顺序依赖
- 双臂协作
- 接触任务恢复
- 门体/抽屉开合状态
- 目标暴露后再抓取

建议映射异常类型：

```text
wrong_sequence
blocked_reach
grasp_miss
collision
wrong_object
```

### 4.2 5 个异常条件

| condition_id | 异常情况 | 注入阶段 | 失败信号 | 期望恢复动作 |
|---|---|---|---|---|
| G2-1 | 把手检测框偏到门板边缘 | 识别把手阶段 | 夹爪接近错误位置，开门动作失败 | 重新定位把手，再执行开门 |
| G2-2 | 抽屉只打开一半就卡住 | 开门过程中 | 抽屉位移不足，内部目标未暴露 | 改用更稳的开门路径，或先回退再二次开门 |
| G2-3 | 开门方向判断错误 | 规划开门动作时 | 门被推向相反方向，接触冲突 | 重新判断铰链/滑轨方向，修正动作顺序 |
| G2-4 | 一只手挡住另一只手的进入路径 | 双臂协作阶段 | 另一只手无法进入抽屉内部 | 改变手臂分工，先收回一只手再进入 |
| G2-5 | 取物时目标被抽屉边缘带偏 | 抓取/抽取阶段 | 目标相对位置变化，抓取失败或掉落 | 重新对准内部目标，降低速度并重新抓取 |

### 4.3 场景成功判据

```text
drawer_or_door reaches intended open state
and target object is extracted without wrong-object contact
and both arms return to a safe posture
```

### 4.4 场景价值

这个场景更能体现 Galaxea 的“动作顺序正确性”和“双臂冲突规避”，适合检验 `wrong_sequence` 和 `collision` 类恢复策略。

## 5. 场景 G3：杂乱桌面分拣

### 5.1 场景定义

机器人位于工作台前，从多个相似物体中识别目标，将其抓起并放入指定容器，或者根据类别放到不同区域。

这个场景最适合测试：

- 多目标识别
- 错误对象恢复
- 遮挡恢复
- 堆叠物体抓取
- 放置目标选择

建议映射异常类型：

```text
wrong_object
grasp_miss
object_displaced
collision
gripper_fail
```

### 5.2 5 个异常条件

| condition_id | 异常情况 | 注入阶段 | 失败信号 | 期望恢复动作 |
|---|---|---|---|---|
| G3-1 | 抓错外观相近的物体 | 感知/抓取点生成阶段 | 抓起的不是目标类别 | 重新做目标识别，禁止沿用相似物体的 grasp |
| G3-2 | 目标被部分遮挡，分割只看见局部 | 识别阶段 | bbox 不稳定，抓取点偏到边缘 | 切换多视角或重新观察后再抓取 |
| G3-3 | 目标与其他物体堆叠 | 抓取规划阶段 | 抓取点被遮挡或不可达 | 先移开上层干扰物，再抓目标 |
| G3-4 | 放置到错误容器 | 运输/放置阶段 | 目标被放入错误区域 | 重新检查任务目标和分类规则，纠正放置动作 |
| G3-5 | 抓取时碰倒邻近物体 | 接近/闭合阶段 | 邻近物体姿态变化或位移明显 | 重新整理局部场景后再抓，避免粗暴接近 |

### 5.3 场景成功判据

```text
correct object is selected
and grasp is stable
and object is placed into the intended container or zone
and neighboring objects remain within allowed disturbance limits
```

### 5.4 场景价值

这个场景最适合体现经验库在“错误对象规避”和“视觉重检”方面的价值，也最容易和 `wrong_object`、`grasp_miss`、`object_displaced` 对齐。

## 6. 场景 G4：双臂搬运与放置

### 6.1 场景定义

机器人需要使用双臂搬运长盒、托盘、工具箱或较大物体，将其移动到另一位置后平稳放置。

这个场景最适合测试：

- 双臂同步
- 负载平衡
- 夹持一致性
- 移动中稳定保持
- 放置前后恢复

建议映射异常类型：

```text
gripper_fail
slip
incipient_slip
collision
blocked_reach
```

### 6.2 5 个异常条件

| condition_id | 异常情况 | 注入阶段 | 失败信号 | 期望恢复动作 |
|---|---|---|---|---|
| G4-1 | 左手先松开，右手还在抬升 | 双手搬运中段 | 物体姿态明显倾斜 | 立即减速并重新建立双手同步 |
| G4-2 | 双臂抬升不同步 | lift 过程中 | 左右端高度差持续增大 | 回退到同步高度，重新执行双臂 lift |
| G4-3 | 搬运途中底盘被窄通道阻挡 | 移动阶段 | 机器人到达不了放置位 | 更换路径或改为分段搬运 |
| G4-4 | 一侧夹爪保持力不足导致滑移 | 搬运阶段 | 物体逐步下滑或偏转 | 提前降落、重新夹紧、补偿另一侧支撑 |
| G4-5 | 放置区域被占用 | 放置前 | 目标位置已有物体或障碍 | 改变放置点，或先清空区域再放置 |

### 6.3 场景成功判据

```text
both arms maintain stable grasp
and object remains level during transport
and final placement succeeds without slip or collision
```

### 6.4 场景价值

这个场景是 Galaxea 与 UR5E 最大的差异点之一，最能体现双臂协作和全身恢复能力。

## 7. 细化后的实验分层

为了避免“异常太粗”，建议把每个场景继续分成三层：

### 7.1 场景层

```text
G1 货架取物
G2 抽屉/柜门取物
G3 杂乱桌面分拣
G4 双臂搬运与放置
```

### 7.2 条件层

每个场景 5 个异常条件，对应具体注入参数和失败证据。

### 7.3 事件层

每个条件还可以细分为事件序列，例如：

```text
detect -> move -> grasp -> close -> lift -> transport -> place
```

这样就可以记录异常发生在：

```text
pregrasp / approach / close / lift / transport / place
```

这一层对后续经验库检索很重要，因为同样是 `slip`，发生在 lift 早期和搬运末期，恢复策略是不一样的。

## 8. 推荐的结果字段

建议每条 trial 结果至少包含：

```json
{
  "robot": "galaxea_r1pro",
  "scenario_id": "G3",
  "condition_id": "G3-2",
  "anomaly_type": "wrong_object",
  "task_stage": "grasp",
  "injection_stage": "recognition",
  "failure_reason": "wrong_object_selected",
  "recovery_success": true,
  "task_success": true,
  "retrieved_positive_count": 5,
  "retrieved_failed_count": 2,
  "collision_or_block_violation": false
}
```

建议统一的失败原因枚举可以包括：

```text
target_displaced
used_stale_target
wrong_object_selected
blocked_path_violation
gripper_not_closed
partial_close
slip_detected
collision_detected
motion_timeout
sequence_violation
```

## 9. 论文中建议的写法

Galaxea 部分不要写成“桌面抓取 benchmark 的扩展版”，而应写成：

> 为验证经验库能否迁移到更复杂的移动操作平台，本文在 Galaxea R1Pro 上构建了室内移动操作异常恢复场景。与 UR5E 的固定基座桌面抓取不同，Galaxea 场景同时包含底盘移动、躯干姿态调整、双臂协作与多视角感知，因此异常恢复不仅依赖局部抓取策略，还依赖全身级别的路径重规划、目标重定位和动作顺序修正。

如果要把这部分做成 benchmark，建议的最小组合是：

```text
4 scenarios × 5 conditions × 4 methods × N trials
```

其中 `N` 可以先取 3，跑通后再扩到 5。

## 10. 与现有代码的对应关系

虽然当前文档不依赖 `r1pro_grasp_scene.xml` 的测试场景，但它与现有能力是对齐的：

```text
wheel_move.py -> 底盘移动与停位
whole_body_move.py -> 躯干 + 全身姿态调整
follow_target.py -> 动态目标跟踪
r1pro_memory_benchmark.py -> 异常恢复与经验库对比
r1pro_llm_strategy_benchmark.py -> LLM 策略选择
```

因此，Galaxea 异常场景可以在现有能力之上进一步向“真实任务区”扩展，而不是停留在测试 cube 级别。

