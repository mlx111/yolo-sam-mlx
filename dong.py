import mujoco
import numpy as np
import time
import mujoco.viewer
import os
import cv2
from matplotlib import colors
apple_index=0
banana_index=0
box_index=0
bowl_index=0
eyeglasses_index=0
camera_index=0
cup_index=0
hammer_index=0
pear_index=0
from openai import OpenAI
import json
import glfw
# 初始化glfw
glfw.init()
glfw.window_hint(glfw.VISIBLE,glfw.FALSE)
window = glfw.create_window(1200,900,"mujoco",None,None)
glfw.make_context_current(window)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def addapple(spec,pos,i):
  world=spec.worldbody
  apple_body=world.add_body(
    name=f'apple{i}',
    quat=[0.7071,0.7071,0,0],
    pos=pos
  )
  # 2. 在该body下添加一个自由关节 (free joint)
# free关节允许物体在三维空间中自由移动和旋转
  apple_body.add_joint(name=f"Apple_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  apple_body.add_geom(
    name=f"apple{i}", 
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname='apple',
    material='apple_mat'
    )
  global apple_index
  apple_index+=1
  

def addpear(spec,pos,i):
  world=spec.worldbody
  apple_body=world.add_body(
    name=f'pear{i}',
    quat=[1,0,0,0],
    pos=pos
  )
  # 2. 在该body下添加一个自由关节 (free joint)
# free关节允许物体在三维空间中自由移动和旋转
  apple_body.add_joint(name=f"pear_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  apple_body.add_geom(
    name=f"pear{i}", 
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname='pear',
    material='banana_mat'
    )
  global pear_index
  pear_index+=1

def addbanana(spec,pos,i):
    world=spec.worldbody
    apple_body=world.add_body(
    name=f'banana{i}',
    quat=[0,1,0,0],
    pos=pos
    )
    
  # 2. 在该body下添加一个自由关节 (free joint)
# free关节允许物体在三维空间中自由移动和旋转
    apple_body.add_joint(name=f"banana_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
    apple_body.add_geom(
      name=f"banana{i}", 
      type=mujoco.mjtGeom.mjGEOM_MESH,
      meshname='banana',
      material='banana_mat'
      )
    global banana_index
    banana_index+=1
def addbox(spec,pos,i,rgba):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"box_{i}",
    quat=[1,0,0,0],
    pos=pos
  )
  
  boxbody.add_joint(name=f"box_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    name=f'box_{i}',
    size=[0.035,0.035,0.035],
    rgba=rgba
  )
  global box_index
  box_index+=1

def addbowl(spec,pos,i):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"bowl{i}",
    pos=pos,
    quat=[1,0,0,0]
  )
  
  boxbody.add_joint(name=f"bowl_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    name=f'bowl{i}',
    meshname='bowl'
  )
  global bowl_index
  bowl_index+=1
def addcamera(spec,pos,i):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"camera{i}",
    pos=pos,
    quat=[1,0,0,0]
  )
  
  boxbody.add_joint(name=f"camera_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    name=f'camera{i}',
    meshname='camera'
  )
  global camera_index
  camera_index+=1
def addcup(spec,pos,i):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"cup{i}",
    pos=pos
  )
  
  boxbody.add_joint(name=f"cup_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    name=f'cup{i}',
    meshname='cup'
  )
  global cup_index
  cup_index+=1
def addeyeglasses(spec,pos,i):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"eyeglasses{i}",
    pos=pos
  )
  
  boxbody.add_joint(name=f"eyegalases_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    name=f'eyeglasses{i}',
    meshname='eyeglasses'
  )
  global eyeglasses_index
  eyeglasses_index+=1

def addhammer(spec,pos,i):
  world=spec.worldbody
  boxbody=world.add_body(
    name=f"hammer{i}",
    pos=pos
  )
  
  boxbody.add_joint(name=f"hammer_joint{i}", type=mujoco.mjtJoint.mjJNT_FREE, damping=0.1)
  boxbody.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    name=f'hammer{i}',
    meshname='hammer'
  )
  global hammer_index
  hammer_index+=1
# 1. 定义对象类型与「函数+索引」的映射关系（核心优化）
object_handler = {
    'apple': (addapple, apple_index),
    'banana': (addbanana, banana_index),
    'box': (addbox, box_index),
    'bowl': (addbowl, bowl_index),
    'camera': (addcamera, camera_index),
    'cup': (addcup, cup_index),
    'eyegalsses': (addeyeglasses, eyeglasses_index),  # 注：建议修正拼写为 eyeglasses
    'hammer': (addhammer, hammer_index),
    'pear':(addpear,pear_index)
}

def generate_scene(prompt,visual=False):
  client = OpenAI(
        api_key='sk-8cf986d1ddac4e64b83723465fc2a6ee',
        base_url="https://api.deepseek.com")
    #content=f"图片名称:{image},检测对象:{object}"
  system_prpmt='''
    你是一个仿真环境生成领域专家，
    现在我需要你解析我输入的文字中需要生成的物体及其个数与位置
    输出要求：
    
    1.使用json数组格式输出指令序列
    2.位置信息需要以数组嵌套的形式给出，只有一个位置信息也需要嵌套 键名是positions
    3.不要有三反印号之类的东西，只要josn
    4.生成的物体需要英文形式
    5.每个物体需要有count字段表示数量
'''
  spec=mujoco.MjSpec.from_file(f'{ROOT_DIR}/manipulator_grasp/assets/scenes/scene2.xml') 
  spec.option.timestep = 0.005  # 仿真步长
  spec.option.gravity = (0, 0, -9.81)  # 重力加速度（z 轴向下）
  #message=input("请用户输入指令:")
  message=prompt
  response = client.chat.completions.create(
        model="deepseek-chat",#模型
        
        messages=[
            {"role": "system", "content": system_prpmt},
            {"role": "user", "content": message},
        ],
        stream=False
    )
  results=response.choices[0].message.content
  print(results)
  res_jsons=json.loads(results)
  for json_1 in res_jsons:
    add_object=json_1['object']
    #print(add_object)
    if add_object=='banana' or add_object=='apple':
      spec.add_mesh(file=f'{ROOT_DIR}/manipulator_grasp/assets/fruit/stl/{add_object}.stl',name=f'{add_object}')
    elif add_object.endswith('box'):
      words=add_object.split('_')
      add_object='box'
      rgba=words[0]
      print('rgba:',rgba)
      rgba = colors.to_rgba(rgba)
    else:
      spec.add_mesh(file=f'{ROOT_DIR}/manipulator_grasp/assets/fruit/stl/{add_object}.stl',name=f'{add_object}')
    count=json_1['count']
    poss=json_1['positions']
    print(add_object)
    if add_object in object_handler:
      if add_object=='box':
        for i in range(count):
          addbox(spec,poss[i],box_index,rgba)
      else:

        for i in range(count):
          func, idx = object_handler[add_object] 
          func(spec,poss[i],idx)
    else:
      print(f"警告：未定义的对象类型 '{add_object}'，跳过处理")
  #print(spec.to_xml())
  model = spec.compile()
  #创建相机
  camera = mujoco.MjvCamera()
  camID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "cam")
  camera.fixedcamid = camID
  camera.type = mujoco.mjtCamera.mjCAMERA_FIXED 

  scene = mujoco.MjvScene(model, maxgeom=1000)
  context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
# 创建 MjData：存储仿真的动态数据（位置、速度、力等）
  data = mujoco.MjData(model)
  if visual:
    viewer =  mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
      mujoco.mj_step(model, data)
      getdep(model,data,640,480)
      #img = get_image(640,480,model,data,camera,scene,context)
      # 更新可视化
      viewer.sync()
      time.sleep(0.002) 
  # 仿真循环（运行1000步）
  # 关闭可视化窗口
    viewer.close()
  xml=spec.to_xml()
  file=open(f'{ROOT_DIR}/manipulator_grasp/assets/scenes/s.xml','w')
  file.write(xml)

def getdep(m,d,h,w):
  mj_renderer = mujoco.renderer.Renderer(m,w,h)
  mj_depth_renderer = mujoco.renderer.Renderer(m,w,h)
          # 更新渲染器中的场景数据
  mj_renderer.update_scene(d, 0)
  mj_depth_renderer.update_scene(d, 0)
        # 启用深度渲染
  mj_depth_renderer.enable_depth_rendering()
  color_img = mj_renderer.render()
  depth_img=mj_depth_renderer.render()
  color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

def get_image(w,h,m,d,camera,scene,context):
    
    # 定义视口大小
    viewport = mujoco.MjrRect(0, 0, w, h)
    # 更新场景
    mujoco.mjv_updateScene(
        m, d, mujoco.MjvOption(), 
        None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    # 渲染到缓冲区
    mujoco.mjr_render(viewport, scene, context)
    # 读取 RGB 数据（格式为 HWC, uint8）
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)  # 深度数组（单通道浮点型）
    mujoco.mjr_readPixels(rgb, depth, viewport, context)
    # 处理深度图像：翻转（与RGB保持一致） + 转换为实际距离
    depth = np.flipud(depth)  # 修正上下颠倒
    # 利用相机近/远裁剪面，将归一化深度转换为实际距离（单位：米）
    znear = m.vis.map.znear  # 相机近裁剪面（从model中读取）
    zfar = m.vis.map.zfar    # 相机远裁剪面
    actual_depth = znear * zfar / (zfar - depth * (zfar - znear))  # 转换公式
    # 转换颜色空间 (OpenCV使用BGR格式)
    cv_image = cv2.cvtColor(np.flipud(rgb), cv2.COLOR_RGB2BGR)
    return {
      'img':cv_image,
      'depth':actual_depth
    }

def gen_prompt(objects:json):
  
  ans=""
  for key in objects:
    ans+=f"{key},{objects[key]},"
  
  return ans
  

if __name__=="__main__":

  prompt=f'''

'''
  