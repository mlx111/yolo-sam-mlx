import os
# Install SDK:  pip install 'volcengine-python-sdk[ark]'
from volcenginesdkarkruntime import Ark 
import base64
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
client = Ark(
    # The base URL for model invocation
    base_url="https://ark.cn-beijing.volces.com/api/v3", 
    # Get API Key：https://console.volcengine.com/ark/region:ark+cn-beijing/apikey
    api_key=os.getenv('ARK_API_KEY'), 
)

from chat import res
def encode_file(file_path):
  with open(file_path, "rb") as read_file:
    return base64.b64encode(read_file.read()).decode('utf-8')

def get_imgs():
    ans=[]
    for i in range(0,3500,50):
        file_base=encode_file(f"{ROOT_DIR}/scenes_imgs/color_quan2__{i}.jpg")
        ans.append(file_base)
    return ans

def get_prompt():
    prompt='''
 "你是一名机器人监控专家。当前机械臂正在执行 [抓取] 技能。 请分析提供的 MuJoCo 仿真图像：

描述机械臂末端与红色方块的空间位置关系。

判断夹爪是否已经牢牢固定在方块两侧？

如果执行错误，请指出原因（如：碰撞、落空、遮挡）。 请以 JSON 格式输出：{'status': 'SUCCESS/FAILURE/IN_PROGRESS', 'reason': '...', 'confidence': 0.95}"

'''
def gen_content(content_list,action):
    if action=='移动到预抓取位置':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。
        当前机械臂正在执行的技能是[移动到预抓取位置]。 
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,
        技能执行正确的条件是机械臂夹爪到达待抓取物体附近，若没有则执行错误
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...'}}'''
    elif action=='移动到抓取位置':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到抓取位置]技能。 
        该机械臂由ur5e机械臂与黑色的2f85夹爪构成
        传入的图像由三个部分组成，分别是由机械臂右侧拍摄的图像，机械臂前方拍摄的图像以及机械臂碗部拍摄的图像
        请按照以下步骤思考并在，输出的consider中输出每一个步骤的思考结果：
        1. 定位：观察图像中物体的中心位置。
        2. 观察：指出黑色夹爪左右两个指尖（fingertips）的具体位置。
        3. 空间推算：判断两个指尖形成的闭合空间是否完全包含了物体的中心点。
        4.当前的指令是将机械臂移动到能够正确抓取物体的位置
        5.预期目标是机械臂夹爪的两指在物体的两侧，确保物体可以被抓住进行下一步动作，且夹爪的两指不能碰触到物体，如果夹爪的黑色两指不是自然伸直而是有弯曲则说明夹爪碰触到了物体
        6.仔细观察机械臂夹爪的位置是否在物体的两侧，与预期情况进行对比
        7.如果机械臂夹爪没有移动到物体的两侧，或者夹爪碰触到物体说明技能执行失败
        否则执行错误。如果没有图片传入返回错误。
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...','consider':'...'}}'''
    elif action=='提升物体':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[提升物体]技能。 
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,
        请按照以下步骤思考并在，输出的consider中输出每一个步骤的思考结果：
        1.机械臂夹爪是否闭合并抓取物体
        2.机械臂夹爪在提升物体的过程中是否持续抓取物体
        3.物体在提升过程中是否掉落
        我草泥马你是不是有病，仔细看看到底夹爪有没有抓住物体，你这个傻逼
        技能执行正确的标准是机械臂夹爪持续抓取物体并将其提升至预定高度，注意在提升物体的过程中夹爪需要始终抓取物体。

        失败的标准是机械臂夹爪在提升阶段开始没有抓取到物体，或者物体在提升过程中掉落，总之，如果在提升物体过程中夹爪没有夹取物体即为失败
        注意机械臂在抓取过程中没有抓取物体就提升要报错
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...','consider':'...'}}'''
    elif action=='移动到碗的上方':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[移动到碗的上方]技能。 
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,
        正确执行的标准是机械臂抓取着物体将夹爪持续抓取的物体移动到碗的上方，
        失败的标准是夹爪没有抓取物体就移动或者物体在抓取过程中从夹爪中掉落
        如果没有图片传入返回错误。
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...'}}'''
    elif action=='回到初始位置':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行[回到初始位置]技能。 
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,正确执行的标准是机械臂恢复到初始时保持竖直的状态，
        执行失败的标准是机械臂没有成功回到初始状态或者机械臂夹爪在回到初始状态后依然夹取着物体
        如果没有图片传入返回错误。
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...'}}'''
    elif action=='夹爪闭合':
        prompt=f'''
        你是一名专业的mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行移动到[夹爪闭合]技能。 
        该机械臂由ur5e机械臂与黑色的2f85夹爪构成
        传入的图像由三个部分组成，分别是由机械臂右侧拍摄的图像，机械臂前方拍摄的图像以及机械臂碗部拍摄的图像
        请按照以下精细步骤思考，并在输出的consider中输出思考结果：
        1. 定位：观察图像中物体的中心位置。
        2. 观察：指出黑色夹爪左右两个指尖（fingertips）的具体位置。
        3. 空间推算：判断两个指尖形成的闭合空间是否完全包含了物体的中心点。
        4. 细节检查：观察夹爪闭合处是否仅仅是空捏（指尖相碰但中间没有物体）？
        5. 成功标准：如果夹爪闭合处包含物体且夹爪两指正确贴合物体则成功
        6. 失败标准：如果夹爪闭合处没有包含物体（空捏）或者夹爪没有正确贴合物体则失败
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,正确执行的标准是机械臂夹爪闭合并抓紧物体。如果没有图片传入返回错误。
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...','consider':'...'}}'''
    elif action=='夹爪开启':
        prompt=f'''
        你是一名mujoco仿真环境中的机械臂监控专家。当前机械臂正在执行移动到[夹爪开启]技能。 
        请分析提供的 MuJoCo 仿真图像,判断技能执行的是否正确,正确执行的标准是机械臂夹爪松开物体。如果没有图片传入返回错误。
        请以 JSON 格式输出：{{'status': 'SUCCESS/FAILURE','reason':'...'}}'''
    content_list.append({
        "type": "text", "text": prompt
        })
    completion = client.chat.completions.create(
    # Replace with Model ID
        model = "doubao-seed-1-6-vision-250815",
        messages=[
        {
            "role": "user",
            "content": content_list,
        }
        ],
    )
    ans=  completion.choices[0].message.content
    print(ans)
    return ans


def get_obj():
    content_list=[]
    for flag in ('left','right'):
        img_data=encode_file(f"{ROOT_DIR}/scenes/c{flag}001.png")

        content_list.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{img_data}"
        }
        })
    content_list.append({
   "type": "text", "text": '''
    这里有两张图片,观察这两张图片的黑色传送带上有哪些物体并返回,注意不需返回传送带上的方块与按钮等物体,只需要黑色传送带上的物体,白色基座上的物体不需要,也不需要返回颜色
    以下是输出要求：
    1.使用数组格式返回
    2.物体名称是英文,不要比中文输出多出来物体
    '''
})
    completion = client.chat.completions.create(
    
    # Replace with Model ID
        model = "doubao-seed-1-6-vision-250815",
        messages=[
        {
            "role": "user",
            "content": content_list,
        }
        ],
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def fault_recover(content_list_all,task_list,target=None):
   
# 2. 循环将所有 base64 图片加入 content 列表
    prompt=f'''
   你是mujoco仿真领域专家,机械臂运动被分成了许多技能,传入的图片序列是mujoco中机械臂在执行[抓取物体{target}并放到碗里]这一任务中的技能序列的图片,
   目前该机械臂执行任务出现了错误,这是目前已经执行的机械臂技能及其执行状态(成功或失败):{task_list},最后执行错误的技能给出了错误的原因。
   这是所有可用的技能序列：
    camera-image   作用:获取图像  无参数
    detect-object 作用:识别物体 参数：用户要抓取的物体，如果是中文翻译成英文
    create-cloud  作用:生成点云数据   无参数
    create-grasp  作用:生成抓取的姿势  无参数
    move-grasp  作用:将机械臂移动到抓取物体的位置 无参数
    vertical-grasp 作用：使机械臂竖直抬起 无参数
    gripper-action 作用:夹爪开合   参数为0或1 1表示夹爪闭合,0表示夹爪开启 参数的key值为state
    execute-grasp2 作用:将机械臂移动到目标位置 无参数
    execute-init  作用:返回初始状态 无参数
    注意已经没有移动到预抓取位置的技能了，不要在输出move-pregrasp 了
    请输出应该如何使用已有的技能序列进行恢复操作使得执行错误的技能能够正确执行,
    以下是输出要求：
    1.使用json数组格式输出指令序列
    2.技能字段名为action
    3.不要有三反印号之类的东西,只要json
    4.需要将最后失败的技能撤回到上一个技能,例如没有抓取到物体直接执行提升物体技能需要重新执行移动到抓取位置的技能,也就是需要再次执行move-grasp技能,
    5.仔细观察被抓取的物体{target}是否偏离了原来的位置，如果偏离了原本的位置需要让机械臂回到初始状态并重新获取抓取姿势再次进行抓取，在回到初始状态后必须使夹爪开启
    6.如果物体{target}位置没有发生变化，就不需要回到初始位置和获取图像
    7.每一个动作类似于{{"action": "camera-image", "parameters":{{}}}},
    【核心逻辑约束】：
   1. 物理位置一致性：如果之前的错误是“抓空”或“提升后发现无物体”，由于机械臂已经执行了 vertical-grasp 抬起动作，你必须首先输出 move-grasp 重新回到物体抓取点，
   如果物体相比抓取前移动了位置，需要回到初始位置重新获取抓取姿势并执行抓取动作。
   2. 逻辑顺序：必须严格遵守 [移动到位置 -> 夹爪动作 -> 验证 -> 提升] 的逻辑。
   3. 状态撤回：检测到 vertical-grasp 失败时，必须包含“撤回到 move-grasp”的动作，禁止在空中直接闭合夹爪。
   4. 当前失败的上下文：{task_list}。请分析图片最后中机械臂末端与物体的距离，如果距离过远，必须重新执行 move-grasp。
   5.获取抓取姿势严格遵守[获取图像-> 识别物体->生成点云->生成抓取姿势]的逻辑，需要识别的物体是{target}
   6.在准备抓取物体前必须确保夹爪开启
   7.拍摄图像前必须保证机械臂处于初始位置
    '''

    prompt1 =f'''
    你是mujoco仿真领域专家，请你仔细观察传入的图片序列，告诉我是否物体{target}的位置是否发生了变化
'''
    content_list_all.append({
   "type": "text", "text": prompt
})
    completion = client.chat.completions.create(
    
    # Replace with Model ID
        model = "doubao-seed-1-6-vision-250815",
        messages=[
        {
            "role": "user",
            "content": content_list_all,
        }
        ],
    )
    ans=  completion.choices[0].message.content
    print(ans)
    with open('yichang.json', 'w', encoding='utf-8') as f:
        res_jsons=ans
        f.write(res_jsons)
    return ans

if __name__ =="__main__":
   #fault_recover()
   get_obj()