import os
import sys
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

from pointcloud_v2 import pos, estimate_runtime_camera_poses
from dong2 import generate_scene

sys.path.append(os.path.join(ROOT_DIR, "Grounded-SAM-2"))


def generate_scenes(objetcs):
    from cv_proc import gen_mask

    gen_mask(objetcs, recognition_mode="real")
    result = pos(objetcs)
    print(result)
    camera_poses = estimate_runtime_camera_poses()
    print({"camera_poses": camera_poses})
    generate_scene(result, camera_poses=camera_poses)


if __name__ == "__main__":
    from cv_proc import gen_mask
    from grasp_fastapi import start

    if len(sys.argv) > 1:
        objects = json.loads(sys.argv[1])
    else:
        raise SystemExit("请传入json数组，例如: python zhenghe2.py '[\"apple\",\"pear\"]'")

    gen_mask(objects, recognition_mode="real")
    result = pos(objects)
    print(result)
    camera_poses = estimate_runtime_camera_poses()
    print({"camera_poses": camera_poses})
    generate_scene(result, camera_poses=camera_poses)
    start()
