import requests
import time
import json
import os
import re
import numpy as np
import torch
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from verl import DataProto
from verl.utils.reward_score import _default_compute_score

# ==================== 1. 你的重试装饰器 ====================
def retry(max_attempts: int = 5, base_delay: int = 1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # RL 训练中通常建议降低日志频率，避免刷屏
                    if attempt == max_attempts:
                        print(f"  → RM API 最终失败: {e}")
                    if attempt < max_attempts:
                        time.sleep(base_delay * (2 ** (attempt - 1)))
            return 0.0  # 最终失败返回 0 分
        return wrapper
    return decorator

# ==================== 2. 评判模板 (保留) ====================
DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""

def parse_judge_response(content):
    pattern = r'<judge>(.*?)</judge>'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        judge_str = match.group(1).lower()
        if 'true' in judge_str: return True
        if 'false' in judge_str: return False
    return None

# ==================== 3. 核心 RM 管理器 ====================
class RMManager:
    def __init__(self, 
                 tokenizer, 
                 num_examine=1, 
                 compute_score=None, 
                 eval_mode=False, 
                 rm_workers_num=10, 
                 # 默认换成你的千帆配置
                 rm_url="https://qianfan.baidubce.com/v2/chat/completions", 
                 rm_key="EMPTY", 
                 rm_model_name="deepseek-v3.2") -> None:
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.eval_mode = eval_mode
        self.rm_workers_num = rm_workers_num
        self.rm_url = rm_url
        self.rm_key = rm_key
        self.rm_model_name = rm_model_name

    # 使用你的调用逻辑包装 rm_score
    @retry(max_attempts=4, base_delay=2)
    def rm_score(self, data_eval_item):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.rm_key}" # 适配千帆 Bearer Token
        }
        
        payload = {
            "model": self.rm_model_name,
            "messages": [
                {
                    "role": "user", 
                    "content": DEFAULT_SYSTEM_TEMPLATE \
                        .replace("{query}", str(data_eval_item["query"])) \
                        .replace("{reference_answer}", str(data_eval_item["reference_answer"])) \
                        .replace("{generated_answer}", str(data_eval_item["generated_answer"]))
                }
            ],
            "temperature": 0.01, # 评判任务建议低采样
            "top_p": 0.01
        }

        response = requests.post(self.rm_url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        # 适配千帆/OpenAI 标准返回格式
        content = result['choices'][0]['message']['content']
        
        judge_result = parse_judge_response(content)
        if judge_result is True:
            return 1.0
        return 0.0

    def __call__(self, data: DataProto):
        # 1. 如果已有分数则跳过
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        data_eval_list = []

        # 2. 准备待评测数据
        for i in range(len(data)):
            data_item = data[i]
            
            # 这里的索引逻辑必须严格适配你的训练输入
            response_ids = data_item.batch['responses']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # 解析出 <answer> 内的内容
            decoded_response = self.tokenizer.decode(valid_response_ids)
            generated_answer = self.get_answer_from_str(decoded_response)
            
            extra_info = data_item.non_tensor_batch.get('extra_info', {})
            
            data_eval_list.append({
                "query": extra_info.get('question', 'N/A'),
                "generated_answer": generated_answer or "No answer format found",
                "reference_answer": data_item.non_tensor_batch['reward_model']['ground_truth'],
                "index": i,
                "valid_len": valid_response_length
            })

        # 3. 多线程请求你的千帆接口
        with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
            model_scores = list(pool.map(self.rm_score, data_eval_list))

        # 4. 组合最终奖励 (结合 NDCG 和 基础规则分)
        for i, eval_item in enumerate(data_eval_list):
            model_eval_score = model_scores[i]
            
            # 假设你依然保留之前的复合得分逻辑
            # 这里需要计算 ndcg_value 和基础规则 score (从 data_item 重新获取)
            # 为了简洁，这里示例直接使用 model_eval_score
            final_reward = model_eval_score 
            
            # 将奖励放在响应的最后一个 token 位置
            reward_tensor[i, eval_item["valid_len"] - 1] = final_reward

        return reward_tensor

    def get_answer_from_str(self, text):
        # 匹配 <answer>...</answer>
        match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else None