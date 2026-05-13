# Galaxea R1Pro MuJoCo Reach Demo

This folder contains an isolated MuJoCo reach/grasp entrypoint for the
GalaxeaManipSim R1Pro two-finger gripper asset.

## Run

```bash
python -m r1_pro_grasp_mujoco.r1_pro_follow --x 0.62 --y 0.20 --z 0.43
```

This is the right-arm equivalent of the Isaac Sim `follow` demo. It loads the
MuJoCo scene, moves the right arm toward the requested world coordinate, closes
the two-finger gripper, and keeps the target marker visible in the viewer.

Optional flags:

- `--no-viewer` to run headless
- `--dry-run` to solve IK without stepping the simulation
- `--skip-stand` to skip the initial standing pose
- `--approach-height`, `--final-offset`, `--duration` to tune the motion
- `--hold-steps` defaults to `-1`, which keeps the viewer open until you close it

If you want the older interactive left/right reach demo, use:

```bash
python -m r1_pro_grasp_mujoco.r1_pro_reach
```
