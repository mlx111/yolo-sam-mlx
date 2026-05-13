import os
import time
import openai
import json
import datetime
import numpy as np

class LLMPrompter():
    def __init__(self, gpt_version, api_key) -> None:
        self.gpt_version = gpt_version
        if api_key is None:
            raise ValueError("OpenAI API key is not provided.")
        else:
            openai.api_key = api_key

    def query(self, prompt: str, sampling_params: dict, save: bool, save_dir: str) -> str:
        while True: # 开启死循环，直到请求成功
            try:
                if 'gpt-4' in self.gpt_version: # 判断是否为聊天系列模型
                    response = openai.ChatCompletion.create(
                        model=self.gpt_version,
                        messages=[
                                {"role": "system", "content": prompt['system']}, # 系统指令
                                {"role": "user", "content": prompt['user']},     # 用户输入
                            ],
                        **sampling_params # 展开字典，传入 temperature, max_tokens 等参数
                        )
                else: # 针对早期的文本补全模型（如 GPT-3）
                    response = openai.Completion.create(
                        model=self.gpt_version,
                        prompt=prompt,
                        **sampling_params
                    )
            except Exception as e:
                # 如果发生异常（如网络问题或限流），等待 2 秒后继续重试
                print("Request failed, sleep 2 secs and try again...", e)
                time.sleep(2)
                continue
            break # 请求成功，跳出循环

        if save: # 如果开启了保存功能
            key = self.make_key() # 生成当前时间戳作为这条记录的 ID
            output = {}
            os.system('mkdir -p {}'.format(save_dir)) # 确保保存目录存在
            
            # 如果文件已存在，先读取原有内容，避免覆盖之前的记录
            if os.path.exists(os.path.join(save_dir, 'response.json')):
                with open(os.path.join(save_dir, 'response.json'), 'r') as f:
                    prev_response = json.load(f)
                    output = prev_response

            # 写入新数据到 response.json
            with open(os.path.join(save_dir, 'response.json'), 'w') as f:
                if 'gpt-4' in self.gpt_version:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response['choices'][0]['message']["content"].strip() # 提取回复内容
                            }
                else:
                    output[key] = {
                                'prompt': prompt,
                                'sampling_params': sampling_params,
                                'response': response['choices'][0]['text'].strip(),
                                # 计算 Token 的平均对数概率，用于衡量模型生成内容的确定性/信心
                                'logprob': np.mean(response['choices'][0]['logprobs']['token_logprobs'])
                            }
                json.dump(output, f, indent=4) # 格式化写入文件
            
        if 'gpt-4' in self.gpt_version:
            return response['choices'][0]['message']["content"].strip(), None
        else:
            return response['choices'][0]['text'].strip(), np.mean(response['choices'][0]['logprobs']['token_logprobs'])

    def make_key(self):
        return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")