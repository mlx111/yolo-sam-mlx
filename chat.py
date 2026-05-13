'''import ollama

prompt='输出一个json格式，其中需要包含图片的名字和需要检测的物体名称，除此之外什么也不要返回'

resp=ollama.generate(
    model='deepseek-r1:8b',
    prompt=prompt
).response

print(resp)'''



# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
import json
import subprocess

count=0

def client():
    client = OpenAI(
        api_key='sk-8cf986d1ddac4e64b83723465fc2a6ee',
        base_url="https://api.deepseek.com")
    #content=f"图片名称:{image},检测对象:{object}"
    system_prpmt='''
    你是一个机械臂控制的专家，请将用户输入的指令解析成一系列标准化动作序列。

    预定义的动作池：
    获取图像：camera-image 无参数
    识别物体：detect-object 参数：用户要抓取的物体，如果是中文翻译成英文
    生成点云数据：create-cloud 无参数
    生成抓取的姿势：create-grasp 无参数
    执行第一阶段的抓取：execute-grasp1 无参数
    夹爪开合：gripper-action 参数为0或1 1表示夹爪闭合，0表示夹爪开启
    执行第二阶段的放置：execute-grasp2 有参数x,y，表示指定放置位置坐标
    返回初始状态：execute-init 无参数

    输出要求：
    需要一开始就获取图像
    1.使用json数组格式输出指令序列
    2.参数需要解析用户的输入
    3.不要有三反印号之类的东西,只要josn
    恢复初始之后需要将夹爪开启'''
    message=input("请用户输入指令:")
    response = client.chat.completions.create(
        model="deepseek-chat",#模型
        
        messages=[
            {"role": "system", "content": system_prpmt},
            {"role": "user", "content": message},
        ],
        stream=False
    )
    results=response.choices[0].message.content
    return results



def res(flag):
    if flag==0:
        with open('renwu.json', 'r', encoding='utf-8') as f:
            res_jsons=json.load(f)
    elif flag==1:
        with open('yichang.json', 'r', encoding='utf-8') as f:
            res_jsons=json.load(f)
    #res_jsons=json.loads(results)
    #print(res_jsons)
    #print(len(res_jsons))
    print(res_jsons)
    #res_jsons=res_jsons['actions']
    global count
    #count=0
    for res_json in res_jsons:
        #res_json=json.loads(res)
        action=res_json["action"]
        if action=='detect-object':
            dection=res_json['parameters']['object']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}?target_class={dection}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        elif action=='gripper-action':
            flag=res_json['parameters']['state']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}?flag={flag}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        elif action=='execute-grasp2':
            x=res_json['parameters']['x']
            y=res_json['parameters']['y']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}?x={x}&y={y}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        else:
            #dection=res_json['object']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        # 执行 curl 命令并获取输出
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        print(result.stdout)
        ans=json.loads(result.stdout)
        status=ans['status']
        if count==3:
            break
        if status=='failure':
            count+=1
            res(1)
        
    '''if count==3 or flag==0:
        curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/gen-sorce",
            "-H", "accept: application/json",
            "-d","''"
            ]
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        print(result.stdout)'''
    #return response.choices[0].message.content

def fault_dection():
    with open('yichang.json', 'r', encoding='utf-8') as f:
        res_jsons=json.load(f)
    for res_json in res_jsons:
        #res_json=json.loads(res)
        action=res_json["action"]
        if action=='detect-object':
            dection=res_json['parameters']['object']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}?target_class={dection}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        elif action=='gripper-action':
            flag=res_json['parameters']['state']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}?flag={flag}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        elif action=='execute-grasp2':
            #x=res_json['parameters']['x']
            #y=res_json['parameters']['y']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        else:
            #dection=res_json['object']
            curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/{action}",
            "-H", "accept: application/json",
            "-d","''"
            ]
            print(curl_cmd)
        # 执行 curl 命令并获取输出
        result = subprocess.run(curl_cmd, capture_output=True, text=True)
        print(result.stdout)
        
def score():
    curl_cmd = [
            "curl", "-X", "POST",
            f"http://localhost:8080/gen-sorce",
            "-H", "accept: application/json",
            "-d","''"
            ]
    result = subprocess.run(curl_cmd, capture_output=True, text=True)
    print(result.stdout)

if __name__=="__main__":
    res(0)
    score()
