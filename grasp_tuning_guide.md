# 抓取调参说明

这份文档说明当前左相机抓取流程里，哪些参数可以调、在哪里改、改完会影响什么。

当前链路保持不变：

1. 虚拟相机内参不改
2. world-vertical 不使用
3. 先做 AnyGrasp 候选，再按 STL 面法向筛选
4. 执行阶段使用当前平滑轨迹，不再做机械臂瞬时跳变

## 1. STL 面筛选参数

位置：

- [`anygrasp_sdk/grasp_detection/get_grasp_surface_aligned.py`](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/anygrasp_sdk/grasp_detection/get_grasp_surface_aligned.py)

可调参数：

```python
max_angle_deg: float = 30.0
max_surface_distance: float = 0.02
```

含义：

- `max_angle_deg` 控制抓取方向和 STL 面法向的夹角阈值。
- `max_surface_distance` 控制抓取点到 STL 面的允许距离。

怎么调：

- 抓取太严格、经常没有候选：先把 `max_surface_distance` 调大到 `0.03` 或 `0.04`
- 仍然太严格：再把 `max_angle_deg` 调到 `35` 或 `40`
- 抓取贴到面内部、边缘抓不到：不要继续把 `max_surface_distance` 过度增大，优先回到较小值

建议顺序：

1. 先调 `max_surface_distance`
2. 再调 `max_angle_deg`
3. 最后再看 AnyGrasp 候选质量

## 2. 预抓取退让距离

位置：

- [`main_yoloWorld_sam_left_calibrated.py`](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_left_calibrated.py)

可调参数：

```python
grasp_width = float(gg.width[0]) if hasattr(gg, 'width') and len(gg.width) > 0 else 0.08
approach_distance = float(np.clip(grasp_width + 0.05, 0.08, 0.15))
T_pregrasp = T_wo * sm.SE3(-approach_distance, 0.0, 0.0)
```

含义：

- `approach_distance` 决定机械臂在闭爪前离目标多远。
- 这里沿抓取轴反向退让，不是沿世界轴退。

怎么调：

- 够不到目标：把 `+ 0.05` 改小一点，比如 `+ 0.03`
- 太贴近、容易撞到：把 `+ 0.05` 改大一点，比如 `+ 0.06` 或 `+ 0.07`
- 如果你觉得每次都差不多，但就是差一点点，优先调这个值

## 3. 预备关节姿态

位置：

- [`main_yoloWorld_sam_left_calibrated.py`](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_left_calibrated.py)

可调参数：

```python
q_pre = np.array([0.0, 0.0, np.pi / 2, 0.0, -np.pi / 2, 0.0])
```

含义：

- 这是机械臂开始 Cartesian 接近前的预备姿态。

怎么调：

- 一般先不要动。
- 只有当某个抓取方向总是从这个姿态过去不顺，或者碰到奇怪姿态限制时，再改它。

## 4. 抬升 / 放置 / 下放高度

位置：

- [`main_yoloWorld_sam_left_calibrated.py`](/home/mlx/mujoco/YOLO_World-SAM-GraspNet/main_yoloWorld_sam_left_calibrated.py)

可调参数：

```python
T_lift = sm.SE3.Trans(0.0, 0.0, 0.3) * T_grasp
T_place = sm.SE3.Trans(0.3, 0.3, T_lift.t[2]) * sm.SE3(sm.SO3(T_lift.R))
T_lower = sm.SE3.Trans(0.0, 0.0, -0.1) * T_place
```

含义：

- `T_lift` 是抓住后抬高多少
- `T_place` 是搬运/放置的偏移
- `T_lower` 是放下多少

怎么调：

- 只影响抓完之后怎么走
- 对“能不能抓住”本身影响不大

## 5. 你现在该怎么排查

如果视频里表现是：

- 能靠近，但抓不到：先调 `max_surface_distance`
- 靠近时离目标还有一点距离：再调 `approach_distance`
- 候选本来就很少：再调 `max_angle_deg`
- 闭爪后总是滑脱：优先看 STL 筛选是否太松，或者目标类别的 AnyGrasp 候选本身是否不合适

## 6. 推荐的试参顺序

建议一次只改一个参数，避免看不出是谁起作用。

推荐顺序：

1. `max_surface_distance = 0.03`
2. `max_angle_deg = 35`
3. `approach_distance` 里的 `+0.05 -> +0.03`

如果还是不行，再继续回看视频和日志里的：

- `surface_dist`
- `align`
- `approach_distance`
- `T_pregrasp / T_grasp`

这几个值足够判断是“筛选太严”还是“接近距离不对”。
