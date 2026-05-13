import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
from pointcloud_v2 import pos
from dong import gen_prompt,generate_scene
sys.path.append(os.path.join(ROOT_DIR, 'Grounded-SAM-2'))
from cv_proc import gen_mask
from doubao import get_obj
import json
from grasp_fastapi import start

def generate_scenes(objetcs):
    objects=get_obj()
    res_list = json.loads(objects)
    gen_mask(objetcs, recognition_mode="real")
    result= pos(objetcs)
    print(result)
    prompt = gen_prompt(result)
    generate_scene(prompt)
    start()
if __name__=="__main__":
    objects=get_obj()
    
    res_list = json.loads(objects)
    gen_mask(res_list, recognition_mode="real")
    result= pos(res_list)
    print(result)
    prompt = gen_prompt(result)
    generate_scene(prompt)
    start()
