# 真机实验迁移与接入简要说明

## 1. 需要迁移的代码

建议至少迁移两个目录：

```text
galaxea_mujoco/
experience_system/
```

`galaxea_mujoco` 提供 R1Pro MuJoCo 沙盒、物理驱动技能、抓取位姿生成、`grasp_tool` 抓取工具坐标系和现场原子技能。

`experience_system` 提供经验库、LLM 异常处理计划生成、语义校验、沙盒推演、critic、计划重写和 `validated_robot_plan` 输出。

当前抓取相关坐标系已经统一为：

```text
grasp_tool
```

后续 GraspNet/AnyGrasp 或现场感知输出的抓取矩阵，应理解为：

```text
grasp_matrix_4x4 = grasp_tool 在世界坐标系下的目标位姿
```

## 2. 推荐环境

当前本地验证主要使用：

```bash
conda activate mujoco1
```

核心依赖包括：

```text
python
mujoco
numpy
pinocchio
scipy
opencv-python
Pillow
open3d
faiss-cpu 或 faiss-gpu
openai 兼容 SDK
```

LLM 配置从项目 `.env` 读取，当前约定使用：

```text
EXPERIENCE_LLM_API_KEY
EXPERIENCE_LLM_BASE_URL
EXPERIENCE_LLM_MODEL
EXPERIENCE_LLM_TIMEOUT
```

现场机器需要能访问对应 LLM API；如果现场不能联网，需要提前准备离线替代或只跑 dry-run。

## 3. 真机侧需要补的接口

经验库当前不能直接控制真机，需要现场提供真机 executor，把 `validated_robot_plan` 中的动作映射到真机 API。

最低需要：

```text
左臂/右臂移动到目标位置或抓取位姿
左/右夹爪开合
躯干移动
底盘移动
获取头部 RGB/RGB-D 图像
获取激光雷达数据
获取腕部外力/接触反馈
读取关节状态和执行结果
```

每个真机技能建议返回统一结果：

```json
{
  "success": true,
  "status": "ok",
  "message": "...",
  "final_error": 0.0,
  "risk_flags": [],
  "raw": {}
}
```

必须现场确认：

```text
机器人 base 坐标系
头部 RGB-D 相机外参
LiDAR 坐标系
grasp_tool 到真实夹爪中心的偏移
速度/加速度/力限制
急停和人工确认机制
```

## 4. 真机场景如何进入沙盒

现场传感器输出需要整理为结构化观测：

```text
RGB/RGB-D/LiDAR/机器人状态
-> field_runtime_scene_observation_v1
-> runtime_sandbox_scene_v1
-> MuJoCo runtime_scene.xml
```

当前 schema 已固定，具体传感器到 XML 的转换器需要等现场数据格式确定后实现。

需要记录的信息：

```text
目标物体位置、尺寸、类别
桌面高度和范围
障碍物位置、尺寸、类别
放置区域位置
相机/雷达外参
机器人当前关节、底盘、躯干状态
```

## 5. 经验库如何使用真机和仿真经验

经验库不是直接控制机器人，而是通过下面链路发挥作用：

```text
真机/仿真经验
-> 检索相关成功经验、失败经验、sim-real gap、参数先验
-> 构造 planner_input
-> 写入 LLM prompt
-> LLM 生成异常处理候选计划
-> 语义校验
-> MuJoCo 沙盒物理驱动推演
-> critic 评分和必要重写
-> 输出 validated_robot_plan
-> 真机 executor 执行
-> 真机和仿真结果写回经验库
```

经验的作用包括：

```text
提示 LLM 哪些恢复动作更可能成功
用失败经验惩罚高风险候选
用 sim-real gap 校准沙盒
用参数先验影响 LLM 生成的底层动作参数
用 critic 反馈推动下一轮重写
```

## 6. 真机 episode 写回

真机执行后要保存为 real episode，再导入经验库。

已有参考：

```text
docs/real_episode_template.json
docs/real_episode_import_format.md
source/validate_real_episode.py
source/import_real_episode.py
```

建议每次实验至少保存：

```text
任务目标
异常类型
执行前/中/后的 RGB 或 RGB-D
LiDAR 数据
腕部受力
关节状态
技能序列
每个技能的输入参数和执行结果
最终是否成功
失败原因
人工备注
```

## 7. 现场最小验证顺序

建议按这个顺序做，不要一开始就跑完整闭环：

```text
1. 验证 MuJoCo 环境能启动
2. 验证 grasp_tool 抓取位姿生成
3. 验证物理驱动抓取 smoke
4. 验证 LLM 配置能调用
5. 验证经验库能检索和生成 planner_input
6. 用现场观测手工构造一个 runtime scene
7. 跑 sandbox candidate rollout
8. 输出 validated_robot_plan dry-run
9. 接入真机 executor 执行单个基础技能
10. 执行完整恢复计划并写回 real episode
```

可先跑：

```bash
conda run -n mujoco1 python source/run_grasp_pose_skill_smoke.py \
  --model r1pro_g3_sorting_scene.xml \
  --object-body target_cube \
  --side left \
  --grasp-mode topdown \
  --pregrasp-distance 0.06
```

```bash
conda run -n mujoco1 python view_grasp_scene.py \
  --headless \
  --model r1pro_g3_sorting_scene.xml \
  --side left \
  --object-body target_cube \
  --grasp-offset-z 0.0 \
  --pregrasp-distance 0.06
```

## 8. 当前边界

当前已经具备：

```text
MuJoCo 物理驱动沙盒
grasp_tool 抓取工具坐标系
抓取位姿矩阵生成
LLM 计划生成
经验检索和 planner_input
语义校验
sandbox rollout
critic 评估
validated_robot_plan dry-run
仿真/真机 episode 导入格式
```

当前还需要现场补：

```text
真机传感器到 runtime scene 的转换
validated_robot_plan 到真机 API 的 executor
真机日志自动保存脚本
真实 grasp_tool 标定
真实速度、力、碰撞安全约束
真机执行后的自动写回闭环
```

不能提前声称：

```text
真机成功率已经提升
沙盒和真机完全一致
任意 6D 抓取姿态都能稳定执行
```

现场第一版目标应该是：

```text
真机异常
-> 经验库生成恢复计划
-> 沙盒验证
-> 输出 validated_robot_plan
-> 人工确认后真机执行
-> 执行结果写回经验库
```
