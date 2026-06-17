# Isaac Sim 人形替代方案与异常场景生成建议

## 1. 目标

当前 Galaxea 在 Isaac Sim 和 MuJoCo 中都较难稳定控制，因此不适合直接沿用原有的全身移动操作实验设计。更现实的路线是：

1. 先选一个 Isaac Sim 自带的人形机器人作为替代平台。
2. 用 Isaac Lab/Isaac Sim 的现成能力搭建可控任务。
3. 将“异常”做成参数化注入，而不是手工写死场景。

这样可以在保留“移动 + 上肢操作 + 恢复策略”研究目标的同时，降低控制和仿真搭建难度。

## 2. 适合替代的机器人

### 2.1 首选

**Unitree G1**

- Isaac Sim 里有现成资产。
- Isaac Lab 里有直接对应的 pick-and-place / locomanipulation / upper-body IK 环境。
- 更适合做“移动 + 上肢操作”的异常恢复实验。

**Fourier GR-1**

- 也有现成资产。
- Isaac Lab 中有对应的抓放与上肢任务。
- 腰部和全身可达性通常更有优势，适合扩展到更复杂的操作区域。

### 2.2 次选

**SanctuaryAI Phoenix**

- 有资产，但现成的任务支持不如 G1/GR-1 直接。
- 更适合作为自定义研究平台。

**X-Humanoid Tien Kung**

- 资产完整，适合全身控制研究。
- 但要自己搭任务与异常流程。

**Agibot A2D**

- 有资产，结构上适合研究。
- 现成可复用任务较少。

### 2.3 不推荐作为第一选择

**Unitree H1**

- 官方示例更偏 locomotion。
- 适合做人形运动底座，不适合直接承担你这类操作异常实验。

**Digit / Valkyrie / STAR1 / 其他资产**

- 可以作为起点，但现成的操作任务支持不够直接。

## 3. 推荐路线

如果目标是尽快跑通实验，推荐：

1. 优先选 `G1` 或 `GR-1`。
2. 优先用 `Isaac Lab`，不要只依赖 Isaac Sim 基础关节控制。
3. 先做固定基座或上肢操作，再逐步加入移动底盘。

这比一开始就做完整人形全身控制更稳。

## 4. 在 Isaac 中生成异常场景的做法

### 4.1 基础思路

不要把异常直接写死在场景里，而是做成“事件注入器”。

每个 trial 只生成一个组合：

```text
scenario_id + condition_id + anomaly_type
```

然后在不同任务阶段注入不同扰动。

### 4.2 场景模板

建议先做 4 类模板：

1. 货架区
2. 抽屉/柜门区
3. 杂乱桌面区
4. 双臂搬运/放置区

这些模板可以对应你原来设计的 `G1` 到 `G4`。

### 4.3 异常注入方式

可直接用参数控制异常：

- `target_displaced`：目标位置偏移
- `blocked_reach`：路径前插入障碍
- `wrong_object`：替换成相似物体
- `grasp_miss`：抓取位姿偏移
- `slip`：降低摩擦或夹持力
- `collision`：在人行路径/机械臂路径上放障碍
- `sequence_violation`：强行打乱动作顺序

### 4.4 注入时机

建议按任务阶段注入：

- `perception` 后：遮挡、错物、目标偏移
- `pregrasp` 前：位姿变化、障碍插入
- `grasp/close` 中：抓偏、夹持不足
- `lift` 中：滑移、重心偏移
- `transport` 中：通道阻挡
- `place` 前：放置区占用

## 5. 建议的工程结构

建议拆成 5 个模块：

```text
scene
robot
anomaly_injector
logger
evaluation
```

- `scene`：负责货架、桌面、抽屉、障碍物等 USD 模板。
- `robot`：负责选择 G1/GR-1 并加载控制接口。
- `anomaly_injector`：按条件注入扰动。
- `logger`：记录阶段、失败信号、恢复动作。
- `evaluation`：统计成功率和恢复效果。

## 6. 训练/实验顺序

建议按难度推进：

1. 先做 `G3` 杂乱桌面分拣，最容易验证整套链路。
2. 再做 `G1` 货架取物。
3. 再做 `G2` 抽屉/柜门。
4. 最后做 `G4` 双臂搬运。

不要一开始就把全身控制、移动、双臂和恢复策略全部叠在一起。

## 7. 结论

如果要在 Isaac 里替代 Galaxea，最现实的组合是：

```text
Unitree G1 / Fourier GR-1 + Isaac Lab + 参数化异常注入
```

这条路线能最大限度复用现成资产和任务，同时保留你原本“异常恢复”研究的核心。

## 8. 以现有 UR5e 框架为参考的人形设计

现有 `experiment-sim-wrapper_3` 的价值不在 UR5e 本体，而在它的组织方式。做人形机器人实验时，可以直接沿用这套分层：

```text
scenario -> condition -> injector -> skill -> recovery -> result
```

### 8.1 可复用的结构

- `scenario`：任务大类，例如货架取物、抽屉取物、桌面分拣、双臂搬运。
- `condition`：具体异常条件，例如目标偏移、遮挡、错物、滑移、路径阻挡、放置错误。
- `injector`：异常注入器，只负责改场景状态或感知状态，不直接写死恢复逻辑。
- `skill`：人形机器人的技能动作，例如 `base_move`、`torso_adjust`、`arm_reach`、`bimanual_grasp`、`recover_slip`、`safe_retreat`、`place_object`。
- `recovery`：恢复计划生成与执行，可接 LLM、规则策略或混合策略。
- `result`：统一记录成功率、恢复成功率、异常类型、注入阶段和任务阶段。

### 8.2 人形版条件设计草案

可以直接把 UR5e 的条件思路改成下面这样：

```text
H1 目标识别异常
H2 抓取几何异常
H3 夹持保持异常
H4 运输/搬运异常
H5 路径与安全异常
```

对应的异常类型可以包括：

- `wrong_object`
- `target_displaced`
- `partial_occlusion`
- `grasp_miss`
- `slip`
- `collision`
- `blocked_path`
- `sequence_violation`

### 8.3 人形版技能草案

建议先定义一组稳定的基础技能，再让恢复策略调用它们：

```text
detect-object
create-grasp
base-move
torso-adjust
head-look
arm-reach
gripper-action
bimanual-grasp
lift
transport
place-object
safe-retreat
avoid-obstacle
replan-path
recover-slip
switch-strategy
```

这些技能不要求一开始就全做成学习策略，先做成“可执行的离散动作”更稳。

### 8.4 推荐的人形实验编排

可以沿用原有的阶段顺序，但替换成适合人形的版本：

```text
perception -> approach -> align -> grasp -> lift -> transport -> place -> recover
```

每个异常只在一个阶段注入，避免多个故障叠加导致结果不可解释。

## 9. 参考

- [Isaac Sim Robot Assets](https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_robots.html)
- [Isaac Sim Robot Policy Examples](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_simulation/ext_isaacsim_robot_policy_example.html)
- [Isaac Sim Motion Generation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/motion_generation/index.html)
- [Isaac Lab Overview](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [Isaac Lab Environments](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/environments.html)
