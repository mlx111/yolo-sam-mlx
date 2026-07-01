# UR5e Runtime Scenes

This folder stores generated MuJoCo scenes and their reports.

- `scene.xml`: default RGB-D image generated runtime scene.
- `report.json`: detection, point cloud, pose, and orientation report for `scene.xml`.
- `artifacts/`: masks, annotated images, point clouds, and orientation candidate masks.
- `plate`: generated scenes include a static `plate` body for placement/recovery evaluation.
- `manual_scene.xml`: optional debug scene generated from manually supplied object poses.
- `sim_camera_scene.xml`: optional runtime scene generated from simulation camera data.
