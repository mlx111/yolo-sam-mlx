# Torso Frame Skills

This folder contains the torso-link4 coordinate workflow:

- `head_camera_rgbd_save_skill.py`: capture MuJoCo head-camera RGB-D.
- `head_camera_grounded_sam2_pose_skill.py`: segment an object and output `center_reference` / `median_reference` in `torso_link4`.
- `torso_frame_move_to_pregrasp_skill.py`: move to a torso-frame pregrasp pose.
- `torso_frame_approach_object_skill.py`: approach the target object in torso coordinates.
- `torso_frame_close_gripper_skill.py`: close the gripper.
- `torso_frame_open_gripper_skill.py`: open the gripper.
- `torso_frame_lift_skill.py`: lift upward in torso coordinates.

The intended atomic sequence is:

1. `torso_frame_head_camera_rgbd_save`
2. `torso_frame_head_camera_grounded_sam2_pose`
3. `torso_frame_move_to_pregrasp`
4. `torso_frame_approach_object`
5. `torso_frame_close_gripper`
6. `torso_frame_lift`
