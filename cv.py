import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
import torch
from PIL import Image
import spatialmath as sm

import cv2
import mujoco

from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.append(os.path.join(ROOT_DIR, 'Grounded-SAM-2'))


from cv_proc import segment_image_ground

if __name__ == '__main__':
    objs=['apple','pear','bowl']
    for obj in objs:
        maksimg_path=segment_image_ground('inputs/left1.png',obj)
        print(maksimg_path)
        cv2.imwrite(f'inputs/left1_{obj}_mask.png',maksimg_path, [cv2.IMWRITE_PNG_BILEVEL, 1])