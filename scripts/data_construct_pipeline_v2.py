"""
SFT 数据构造 v2 —— 格式对齐 RL rollout，强制 ≥3 次搜索

与 data_construct_pipeline_opneai.py 保持一致：
  - OpenAI 兼容 API + base64 图像
  - 模型输出 JSON 格式（{"think":..., "search":...}），彻底避免 tool call
  - obs 文字对齐 generation.py 的 RL rollout 格式
"""

import os
import json
import time
import re
import requests
import base64
from typing import Optional
from openai import OpenAI
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ================= 配置 =================
image_output_dir  = './data/image_crop_v2'
raw_image_dir     = '/mnt/cfs_algo_bj/workspace/shenyucheng/SlideVQA-RL/img'
search_engine_url = 'http://localhost:8004/search'

OPENAI_API_KEY  = "REDACTED_API_KEY"
OPENAI_BASE_URL = "https://www.huayanapi.com/v1/"
VLM_MODEL       = 'qwen3-vl-235b-a22b-instruct'

# ================= Prompt =================
PROMPT_INST = """## Role
You are an intelligent visual reasoning agent. You think step-by-step and call tools to answer questions about document slide images.

## Tools
1. search — retrieve a relevant slide image by query
2. crop   — zoom into a region of the last image (bbox coordinates normalized 0-1000)
3. answer — provide the final answer

## STRICT RULES
- You MUST call search at least 3 times before using answer.
- Even if the first search returns the correct image, keep searching to verify from a different angle.
- Use diverse queries for subsequent searches (do not repeat the same query).

## Reply Format (strict JSON, no markdown fences)
Search:
{"think": "<your reasoning>", "search": "<query string>"}

Crop:
{"think": "<your reasoning>", "bbox": [x1, y1, x2, y2], "description": "<what you see>"}

Answer (only after >=3 searches):
{"think": "<your reasoning>", "answer": "<final answer>"}
"""

PROMPT_USER_START = "Question: {question}"


# ================= 工具函数 =================

def build_obs_str(search_count: int, retrieved_count: int, query: str) -> str:
    """构造和 generation.py 完全一致的 obs 文本。"""
    if retrieved_count < 3:
        hint = f' You still need at least {3 - retrieved_count} more search(es) before you can use <answer>.'
    else:
        hint = ''
    return (
        f'[Search #{search_count}, retrieved: {retrieved_count} image(s) total] '
        f'Image loaded, analyze any possible useful information for the question: '
        f'{query} in your think, then continue your action after <think> </think>.{hint}'
    )


def extract_json(response: str) -> dict:
    text = response.replace('```json', '').replace('```', '').strip()
    return json.loads(text)


def do_search(query: str) -> list:
    try:
        resp = requests.get(search_engine_url, params={"queries": [query]}, timeout=10)
        return [item['image_file'] for item in resp.json()[0]]
    except Exception:
        return []


def crop_and_dump(image_path: str, bbox: list, output_folder: str = image_output_dir) -> Optional[str]:
    try:
        os.makedirs(output_folder, exist_ok=True)
        img = Image.open(image_path)
        w, h = img.size
        x1, y1, x2, y2 = bbox
        cropped = img.crop((x1*w/1000, y1*h/1000, x2*w/1000, y2*h/1000))
        if cropped.mode == 'RGBA':
            cropped = cropped.convert('RGB')
        out = os.path.join(output_folder, f"crop_{int(time.time()*1000)}.jpg")
        cropped.save(out)
        return out
    except Exception:
        return None


def image_to_b64(path: str) -> str:
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_image_msg(image_path: str, text: str) -> dict:
    """构造含图的 user message（OpenAI 多模态格式，base64）"""
    b64 = image_to_b64(image_path)
    return {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": text},
        ]
    }


# ================= VLM 封装 =================

class VLMClient:
    def __init__(self, model_name: str = VLM_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.model_name = model_name

    def generate(self, messages: list, max_tokens: int = 1024) -> Optional[str]:
        for attempt in range(5):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                )
                return resp.choices[0].message.content
            except Exception:
                time.sleep(5 * (attempt + 1))
        return None


# ================= 核心轨迹构造 =================

class TrajectoryBuilder:
    def __init__(self, dataset: str, query_file: str, workers_num: int = 10):
        self.dataset_dir  = os.path.join('./', dataset)
        self.results_dir  = os.path.join(self.dataset_dir, 'results')
        self.query_file   = query_file
        self.workers_num  = workers_num
        self.output_path  = os.path.join(self.results_dir, 'cot_v2_results.jsonl')
        os.makedirs(self.results_dir, exist_ok=True)

        self.vlm = VLMClient()
        self.processed_uids: set = set()
        self._load_existing_uids()

    def _load_existing_uids(self):
        if not os.path.exists(self.output_path):
            return
        with open(self.output_path) as f:
            for line in f:
                try:
                    uid = json.loads(line.strip()).get('uid')
                    if uid:
                        self.processed_uids.add(uid)
                except Exception:
                    pass
        print(f"Loaded {len(self.processed_uids)} existing uids.")

    def build_one(self, sample: dict) -> Optional[dict]:
        uid   = sample.get('uid')
        query = sample['query']
        reference_images = [
            os.path.join(raw_image_dir,
                         sample['meta_info']['file_name'].replace('.pdf', f"_{i}.jpg"))
            for i in sample['meta_info']['reference_page']
        ]

        all_images      = []
        search_count    = 0
        retrieved_count = 0
        history         = []
        last_is_crop    = False

        messages = [
            {"role": "system", "content": PROMPT_INST},
            {"role": "user",   "content": PROMPT_USER_START.format(question=query)},
        ]

        for _ in range(14):
            response = self.vlm.generate(messages)
            if not response:
                return None

            try:
                res = extract_json(response)
            except Exception:
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": "Invalid format. Reply with valid JSON only, no markdown."})
                continue

            # ── answer ──
            if 'answer' in res:
                if search_count < 3:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        f"You have only done {search_count} search(es). "
                        f"You MUST do at least {3 - search_count} more search(es) before answering."
                    })
                    continue
                history.append({'think': res.get('think', ''), 'answer': res['answer']})
                sample.update({'response': res['answer'], 'history': history})
                if uid:
                    self.processed_uids.add(uid)
                return sample

            # ── search ──
            elif 'search' in res:
                img_list = do_search(res['search'])
                if not img_list:
                    return None

                new_img = next(
                    (img for img in img_list if img in reference_images and img not in all_images), None
                ) or next(
                    (img for img in img_list if img not in all_images), None
                )
                if not new_img:
                    return None

                search_count    += 1
                retrieved_count += 1
                all_images.append(new_img)
                last_is_crop = False

                obs_text = build_obs_str(search_count, retrieved_count, query)
                history.append({'think': res.get('think', ''), 'search': res['search']})
                history.append([{'image': new_img}, {'text': obs_text}])

                messages.append({"role": "assistant", "content": response})
                messages.append(build_image_msg(new_img, obs_text))

            # ── bbox ──
            elif 'bbox' in res:
                if not all_images or last_is_crop:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "Cannot crop a cropped image or without a prior image. Use search instead."
                    })
                    continue

                crop_path = crop_and_dump(all_images[-1], res['bbox'])
                if not crop_path:
                    continue

                all_images.append(crop_path)
                last_is_crop = True

                obs_text = '[Zoomed view] Analyze this cropped region carefully, then continue your action.'
                history.append({'think': res.get('think', ''), 'bbox': res['bbox'], 'description': res.get('description', '')})
                history.append([{'image': crop_path}, {'text': obs_text}])

                messages.append({"role": "assistant", "content": response})
                messages.append(build_image_msg(crop_path, obs_text))

        return None

    def run(self, limit: Optional[int] = None):
        with open(self.query_file) as f:
            data = json.load(f)['examples']

        pending = [s for s in data if s.get('uid') not in self.processed_uids]
        if limit:
            pending = pending[:limit]
        print(f"Total: {len(data)}, processed: {len(self.processed_uids)}, pending: {len(pending)}")

        if not pending:
            print("Nothing to process.")
            return

        with open(self.output_path, 'a', encoding='utf-8') as f_out:
            with ThreadPoolExecutor(max_workers=self.workers_num) as pool:
                futures = {pool.submit(self.build_one, s): s for s in pending}
                for future in tqdm(as_completed(futures), total=len(pending)):
                    result = future.result()
                    if result is not None:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()


# ================= 入口 =================
if __name__ == '__main__':
    builder = TrajectoryBuilder(
        dataset='SlideVQA-SFT-v2',
        query_file='/mnt/cfs_algo_bj/workspace/shenyucheng/SlideVQA-RL/slidevqa_RL.json',
        workers_num=30,
    )
    builder.run(limit=1000)
