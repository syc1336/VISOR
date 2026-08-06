import os
import json
import base64
import time
import requests
import re
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from llama_index.core.schema import NodeWithScore, ImageNode

# ================= 配置参数 =================
image_output_dir = './data/image_crop_chart'
raw_image_dir = '/mnt/cfs_algo_bj/models/experiments/shenyucheng/ChartQA-SFT/img'
search_engine_url = 'http://localhost:8008/search'

# 百度千帆 OpenAI 兼容配置

# OPENAI_API_KEY = "REDACTED_API_KEY"
# OPENAI_BASE_URL = "https://qianfan.baidubce.com/v2" 
OPENAI_API_KEY = "REDACTED_API_KEY"
OPENAI_BASE_URL = "https://www.huayanapi.com/v1/"

prompt_inst = """## Character Introduction
You are an intelligent assistant capable of performing searches and providing precise answers to user queries. You need to think step by step and provide actions. Your thought should be as detailed as you can. When you cannot answer a question, please use search tool to search for more relevant information. If you think that a specific region of the last image could help answer the question, please use crop tool to provide a detailed view of the relevant region. Once you have gathered enough information to answer the question, provide your response immediately.

### Available Tools
1. search:  
   - Collect relevant information based on the query.  
   - Parameters: The keywords or question to search for.  
   - Returns: Search results for query.

2. crop:
   - Crop the last image based on the user's query.
   - Parameters: The crop region of the image.
   - Returns: Cropped image.

3. answer:  
   - Respond directly to the user based on search results.  
   - Parameters: The response to the user's query.  

### Requirements
- Ensure tool usage is precise and queries are well-formulated.
- Provide accurate and well-structured answers to user queries.
- Iterate search attempts if initial results are insufficient.
- Follow the response format.


### Reply Format
You **must** response with the following json format:
When you need to search, you need to provide the search query in the following format:
```
{
    "think": ...,
    "search": ...
}
```
When you need to crop the image, you need to provide the following format:
For tables, charts, or any visual elements, please use bounding boxes to completely encapsulate them.
```
{
    "think": ...,
    "bbox": [x1, y1, x2, y2],
    "description": the cropped content ...
}
```
When you have gathered enough information to answer the question, provide your response immediately:
```
{
    "think": ...,
    "answer": ...
}
```
"""

prompt_user_start = """Question: {question}"""

# ================= 工具函数 =================
def extract_json(response):
    """从模型输出中提取并解析 JSON"""
    try:
        # 移除 markdown 代码块标记
        clean_res = re.sub(r'```json\s*|\s*```', '', response).strip()
        return json.loads(clean_res)
    except Exception as e:
        print(f"JSON Parsing Error: {e} | Original: {response[:100]}")
        raise e

def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    except Exception as e:
        print(f"Image Read Error: {e}")
        return None

def crop_and_dump(image_path, bbox, output_folder=image_output_dir):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    try:
        image = Image.open(image_path)
        width, height = image.size
        # 归一化坐标 [0, 1000] 转换为像素坐标
        x1, y1, x2, y2 = bbox
        left = x1 * width / 1000.0
        top = y1 * height / 1000.0
        right = x2 * width / 1000.0
        bottom = y2 * height / 1000.0
        
        cropped_image = image.crop((left, top, right, bottom))
        if cropped_image.mode == 'RGBA':
            cropped_image = cropped_image.convert('RGB')

        timestamp = int(time.time() * 1000)
        output_path = os.path.join(output_folder, f"crop_{timestamp}.jpg")
        cropped_image.save(output_path)
        return output_path
    except Exception as e:
        print(f"Crop Failed: {e}")
        return None

# ================= 模型封装 =================
class Model_Role:
    def __init__(self, model_name):
        self.model = model_name
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    def generate(self, messages):
        formatted_msgs = []
        for msg in messages:
            role = msg["role"]
            content = []
            for item in msg["content"]:
                if "text" in item:
                    content.append({"type": "text", "text": item["text"]})
                elif "image" in item:
                    b64 = encode_image_to_base64(item["image"])
                    if b64:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            formatted_msgs.append({"role": role, "content": content})

        for _ in range(10):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=formatted_msgs,
                    temperature=0.1
                )
                return completion.choices[0].message.content
            except Exception as e:
                print(f"API Retry... {e}")
                time.sleep(32)
        return None

# ================= 核心 RAG 逻辑 =================
class MMRAG:
    def __init__(self, dataset='example', query_file='rag_dataset.json', workers_num=1):
        self.dataset_dir = os.path.join('./', dataset)
        self.results_dir = os.path.join(self.dataset_dir, "results")
        self.query_file = query_file
        self.workers_num = workers_num
        self.output_file_path = os.path.join(self.results_dir, "cot_crop_results.jsonl")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 统一模型名称
        target_model = 'qwen3-vl-235b-a22b-instruct'
        self.vlm = Model_Role(model_name=target_model)
        
        # 新增：已处理的 uid 集合
        self.processed_uids: Set[str] = set()
        self._load_existing_uids()

    def _load_existing_uids(self):
        """读取已有的 jsonl 文件，收集所有 uid"""
        if not os.path.exists(self.output_file_path):
            return
        
        print(f"Loading existing processed uids from {self.output_file_path} ...")
        count = 0
        with open(self.output_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    uid = item.get('uid')
                    if uid:
                        self.processed_uids.add(uid)
                        count += 1
                except Exception:
                    continue  # 跳过损坏行
        print(f"Loaded {count} existing uids.")

    def cot_collect(self, sample):
        # 新增：检查 uid 是否已处理
        uid = sample.get('uid')
        if not uid:
            print("Warning: sample missing 'uid' field, will process anyway.")
        elif uid in self.processed_uids:
            # print(f"Skip already processed: {uid}")
            return None  # 或者 return sample 如果你想保留原样

        # 下面是原有逻辑
        query = sample['query']
        reference_images = [f'{raw_image_dir}/'+sample['meta_info']['file_name'].replace('.pdf', f"_{i}.jpg") 
                           for i in sample['meta_info']['reference_page']]
        
        all_images, history, messages = [], [{"query": query}], []
        messages.append({"role": "system", "content": [{"text": prompt_inst}]})
        messages.append({"role": "user", "content": [{"text": prompt_user_start.format(question=query)}]})

        try_times = 10
        while try_times > 0:
            try_times -= 1
            response = self.vlm.generate(messages)
            if not response:
                return None
            
            try:
                res_json = extract_json(response)
            except:
                continue

            if 'answer' in res_json:
                history.append(res_json)
                sample.update({'response': res_json['answer'], 'history': history})
                # 成功生成后加入已处理集合（防止同一个进程重复写）
                if uid:
                    self.processed_uids.add(uid)
                return sample
            
            elif 'search' in res_json:
                search_res = requests.get(search_engine_url, params={"queries": [res_json['search']]}).json()
                img_list = [item['image_file'] for item in search_res[0]]
                
                new_img = next((img for img in img_list if img in reference_images and img not in all_images), None)
                if not new_img:
                    new_img = next((img for img in img_list if img not in all_images), None)
                if not new_img:
                    return None
                
                messages.append({"role": "assistant", "content": [{"text": response}]})
                user_content = [{"image": new_img}, {"text": "You should call crop tool to crop this image. The selected area must be complete and can be larger than the area that needs attention."}]
                messages.append({"role": "user", "content": user_content})
                all_images.append(new_img)
                history.extend([res_json, user_content])
                
            elif 'bbox' in res_json and all_images:
                crop_path = crop_and_dump(all_images[-1], res_json['bbox'])
                if not crop_path:
                    continue
                
                messages.append({"role": "assistant", "content": [{"text": response}]})
                user_content = [{"image": crop_path}, {"text": "Here is the cropped view. Please provide final answer or further action."}]
                messages.append({"role": "user", "content": user_content})
                history.extend([res_json, user_content])

        return None

    def eval_dataset(self):
        with open(os.path.join(self.dataset_dir, self.query_file), "r") as f:
            data = json.load(f)['examples']
        
        # 过滤出需要处理的样本
        pending = [item for item in data if item.get('uid') not in self.processed_uids]
        
        print(f"Total samples: {len(data)}, already processed: {len(self.processed_uids)}, pending: {len(pending)}")
        
        if not pending:
            print("Nothing to process.")
            return
        
        with ThreadPoolExecutor(max_workers=self.workers_num) as executor:
            futures = [executor.submit(self.cot_collect, item) for item in pending]
            for future in tqdm(as_completed(futures), total=len(pending), desc="Processing"):
                res = future.result()
                if res:
                    with open(self.output_file_path, "a", encoding='utf-8') as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")

# ================= 主程序 =================
if __name__ == "__main__":
    mmrag = MMRAG(
        dataset='chartQA-SFT',
        query_file='/mnt/cfs_algo_bj/models/experiments/shenyucheng/ChartQA-SFT/chartqa_SFT.json',
        workers_num=30
    )
    mmrag.eval_dataset()