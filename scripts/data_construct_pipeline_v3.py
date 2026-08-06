"""
SFT 数据构造 v3 —— 格式对齐 RL rollout，自由搜索 + verification

变化（相比 v2）：
  - 不强制 ≥3 次搜索，模型自由决定搜索次数
  - 检索词强制使用原始问题（search query = original question）
  - 第一次想 answer 时拦截，强制做一次 verification search，之后接受答案
  - obs 文字去掉 search count 前缀和 hint
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
image_output_dir  = './data/image_crop_v3'
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

## RULES
- Search as many times as needed until you believe you have enough information.
- Always use the original question as your search query.
- Before answering, you MUST do one final verification search using the original question.
  After that verification search, give your answer immediately.
- You may use crop when critical details in an image are visually unclear.

## Reply Format (strict JSON, no markdown fences)
Search:
{"think": "<your reasoning>", "search": "<query string>"}

Crop:
{"think": "<your reasoning>", "bbox": [x1, y1, x2, y2], "description": "<what you see>"}

Answer (only after verification search):
{"think": "<your reasoning>", "answer": "<final answer>"}
"""

PROMPT_USER_START = "Question: {question}"


# ================= 工具函数 =================

def build_obs_str(query: str, is_reference: bool = False,
                  all_refs_found: bool = False, answer: str = None) -> str:
    """构造和 generation.py 当前版本一致的 obs 文本。
    is_reference=True 时额外提示模型这张图来自参考页，需要重点分析。
    all_refs_found=True 时表示所有参考页已全部检索到，附带正确答案提示。
    注意：此提示仅用于 SFT 数据构造阶段辅助 VLM 生成高质量 think，
    不出现在 RL rollout 的 generation.py 里。
    """
    base = (
        f'Image loaded, analyze any possible useful information for the question: '
        f'{query} in your think, then continue your action after <think> </think>.'
    )
    if is_reference:
        if all_refs_found and answer is not None:
            hint = (
                ' [DATA CONSTRUCTION HINT] This image comes from the ground-truth reference page. '
                'Analyze this image carefully in your think and extract all relevant information. '
                'You have now retrieved ALL reference pages — you have sufficient information to answer. '
                'After your analysis, do one final verification search, then give your answer immediately. '
                f'The correct answer is: {answer} — keep your answer concise and close to this.'
            )
        else:
            hint = (
                ' [DATA CONSTRUCTION HINT] This image comes from the ground-truth reference page, '
                'but there are still other reference pages you have not retrieved yet. '
                'Keep searching to gather complete information before answering.'
            )
        return base + hint
    else:
        return base + (
            ' [DATA CONSTRUCTION HINT] This image may or may not contain relevant information. '
            'If it does not help answer the question, clearly state so in your think.'
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
        self.output_path  = os.path.join(self.results_dir, 'cot_v3_results.jsonl')
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
        last_is_crop         = False
        verified             = False   # 是否已做过 verification search
        verification_pending = False   # 下一次 search 是 verification search

        # ── 初始检索：对齐 generation.py，循环前先取一张初始图 ──
        init_list = do_search(query)
        if not init_list:
            return None
        initial_img = init_list[0]
        all_images.append(initial_img)
        sample['initial_image'] = initial_img

        # 检查初始图是否是参考页
        init_obs = (
            '[Initial retrieval using your question] This is the initial reference image. '
            'Analyze it, then begin your reasoning and actions.'
        )
        if initial_img in reference_images:
            all_refs_seen = all(r in all_images for r in reference_images)
            if all_refs_seen:
                init_obs += (
                    ' [DATA CONSTRUCTION HINT] This image comes from the ground-truth reference page. '
                    'Analyze this image carefully in your think and extract all relevant information. '
                    'You have now retrieved ALL reference pages — you have sufficient information to answer. '
                    'After your analysis, do one final verification search, then give your answer immediately. '
                    f'The correct answer is: {sample.get("reference_answer", "")} — keep your answer concise and close to this.'
                )
            else:
                init_obs += (
                    ' [DATA CONSTRUCTION HINT] This image comes from the ground-truth reference page, '
                    'but there are still other reference pages you have not retrieved yet. '
                    'Keep searching to gather complete information before answering.'
                )

        messages = [
            {"role": "system", "content": PROMPT_INST},
            build_image_msg(initial_img, init_obs),
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
                if not verified:
                    # 拦截：还未做 verification search，强制再搜一次
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "user", "content":
                        "Before answering, you must do one final verification search "
                        f"using the original question: \"{query}\". Please search now."
                    })
                    # 标记下一次搜索是 verification；verified 等搜索实际发生后再置 True
                    verification_pending = True
                    continue
                history.append({'think': res.get('think', ''), 'answer': res['answer']})
                sample.update({'response': res['answer'], 'history': history})
                if uid:
                    self.processed_uids.add(uid)
                return sample

            # ── search ──
            elif 'search' in res:
                img_list = do_search(query)   # 强制使用原始问题作为检索词
                if not img_list:
                    return None

                new_img = next(
                    (img for img in img_list if img not in all_images), None
                )
                if not new_img:
                    return None

                search_count    += 1
                retrieved_count += 1
                all_images.append(new_img)
                last_is_crop = False

                if verification_pending:
                    verification_pending = False
                    verified = True   # 搜索实际发生，现在才接受下一个 answer
                    obs_text = (
                        f'Image loaded, analyze any possible useful information for the question: '
                        f'{query} in your think, then continue your action after <think> </think>.'
                        ' [DATA CONSTRUCTION HINT] This is your verification image.'
                        ' In your <think>, analyze whether this image contains new relevant information.'
                        ' If it does not, it means you already have all the information needed.'
                        ' Either way, give your <answer> immediately based on ALL evidence collected so far — keep it concise.'
                    )
                else:
                    is_ref = new_img in reference_images
                    all_refs_seen = all(r in all_images for r in reference_images)
                    obs_text = build_obs_str(
                        query,
                        is_reference=is_ref,
                        all_refs_found=all_refs_seen,
                        answer=sample.get('reference_answer') if all_refs_seen else None,
                    )
                history.append({'think': res.get('think', ''), 'search': query})
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

                obs_text = (
                    f'This is the cropped image, analyse it based on the question: {query} '
                    'and after <think> </think> you just can use <search> or <answer> this time. '
                    'If the cropped image is incorrect, please pay attention to previous think.'
                )
                # 如果 crop 的源图是参考页，额外提示
                source_img = all_images[-2] if len(all_images) >= 2 else None
                if source_img in reference_images:
                    all_refs_seen = all(r in all_images for r in reference_images)
                    if all_refs_seen:
                        obs_text += (
                            ' [DATA CONSTRUCTION HINT] This crop is from the ground-truth reference page. '
                            'Analyze this image carefully in your think and extract all relevant information. '
                            'You have now retrieved ALL reference pages — you have sufficient information to answer. '
                            'After your analysis, do one final verification search, then give your answer immediately. '
                            f'The correct answer is: {sample.get("reference_answer", "")} — keep your answer concise and close to this.'
                        )
                    else:
                        obs_text += (
                            ' [DATA CONSTRUCTION HINT] This crop is from the ground-truth reference page. '
                            'Focus on extracting useful information from this region.'
                        )
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
        dataset='SlideVQA-SFT-v3',
        query_file='/mnt/cfs_algo_bj/workspace/shenyucheng/SlideVQA-RL/slidevqa_RL.json',
        workers_num=30,
    )
    builder.run(limit=1000)
