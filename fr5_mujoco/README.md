# FR5 MuJoCo

Minimal FR5 MuJoCo assets and control helpers extracted from `fr5_mujoco_env`.

This folder keeps only the pieces needed to load and control the robot:

- `assets/fr5_robot.xml`: FR5 arm model with torque motors.
- `assets/gripper_pgi.xml`: PGI gripper model with two position actuators.
- `assets/meshes/fr5/`: robot and gripper meshes.
- `assets/scene.xml`: small standalone scene.
- `src/fr5_mujoco/control.py`: joint PD, TCP DiffIK, and gripper helper.
- `src/fr5_mujoco/motion_skills.py`: movement skills compatible with the useful UR5e motion surface.
- `skills/`: FR5 field-atomic skill layer modeled after `ur5e_mujoco/skills`.
- `scripts/play_fr5_control.py`: viewer smoke test.
- `scripts/test_move_skills.py`: headless movement skill smoke test.
- `scripts/test_fr5_skills.py`: headless field-atomic skill smoke test.
- `scripts/play_fr5_skills.py`: viewer field-atomic skill demo.
- `scripts/test_fr5_camera_rgbd.py`: end-effector camera RGB-D render smoke test.
- `scripts/test_fr5_grounded_sam2_pose.py`: GroundedSAM2 pose pipeline smoke test.

Run from this repository root:

```bash
PYTHONPATH=fr5_mujoco/src python fr5_mujoco/scripts/play_fr5_control.py
```

Headless movement skill test:

```bash
PYTHONPATH=fr5_mujoco/src python fr5_mujoco/scripts/test_move_skills.py
```

Field-atomic skill test:

```bash
python fr5_mujoco/scripts/test_fr5_skills.py
```

Viewer skill demo:

```bash
python fr5_mujoco/scripts/play_fr5_skills.py
```

End-effector camera RGB-D smoke test:

```bash
python fr5_mujoco/scripts/test_fr5_camera_rgbd.py
```

GroundedSAM2 pose pipeline:

```bash
python fr5_mujoco/scripts/test_fr5_grounded_sam2_pose.py --target-class apple
```

The pose pipeline runs `camera_rgbd_save -> detect_object_pose -> create_fixed_vertical_grasp`. The current standalone scene does not include an apple or placement object by default, so detection may fail cleanly until the scene and `ee_camera` pose are tuned.

Generated RGB-D images, masks, and smoke-test outputs are written under `fr5_mujoco/output/`.

Key names:

- Arm joints/actuators: `shoulder_pan`, `shoulder_lift`, `elbow`, `wrist_1`, `wrist_2`, `wrist_3`
- Gripper actuators: `act_finger1`, `act_finger2`
- TCP site: `tool_center_point`
- End-effector camera: `ee_camera`
