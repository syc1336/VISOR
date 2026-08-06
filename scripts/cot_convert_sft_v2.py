"""
SFT 格式转换 v2 —— 对齐 RL rollout 的 obs 格式

输入：filter.py 筛选后的 cot_v2_results_correct.jsonl
输出：data/SlideVQA-SFT-v2.json（供 SFT-qwen2.5.py 直接训练）

消息格式（Qwen2.5-VL SFT 标准）：
  [0] user:      USER_PROMPT（含问题）
  [1] assistant: <think>...</think><search>q1</search>
  [2] user:      <image>\n[Search #1, retrieved: 1 image(s) total] ... You still need at least 2 more...
  [3] assistant: <think>...</think><search>q2</search>
  [4] user:      <image>\n[Search #2, retrieved: 2 image(s) total] ... You still need at least 1 more...
  [5] assistant: <think>...</think><search>q3</search>   ← 或 <bbox>
  [6] user:      <image>\n[Search #3, retrieved: 3 image(s) total] ...  ← 无软提示
  [7] assistant: <think>...</think><answer>XXX</answer>
"""

import json
import re
from PIL import Image


def convert_v2(input_jsonl: str, output_json: str, min_search: int = 3):
    with open(input_jsonl, 'r') as f:
        data = [json.loads(l) for l in f if l.strip()]

    sft_results = []
    skip_count  = 0

    for item in data:
        query   = item['query']
        history = item['history']

        # ── 统计搜索次数（只数 search，不数 bbox）──
        n_search = sum(
            1 for h in history
            if isinstance(h, dict) and 'search' in h
        )
        if n_search < min_search:
            skip_count += 1
            continue

        conversations = []
        overall_images = []
        good_data = True

        # ── 首条 user 消息：USER_PROMPT ──
        conversations.append({
            "role": "user",
            "content": item.get('prompt', '').strip() or
                       # 兼容旧字段：重新拼 prompt
                       f"You are a visual reasoning agent to answer user's question. Follow these rules without exception:\n\n        1. EVERY response MUST begin with exactly one <think> </think> block containing your internal reasoning.\n        - Do NOT output anything before <think>.\n        - Do NOT skip thinking, even if the answer seems obvious.\n\n        2. AFTER <think> </think>, you must output ONE of the following actions:\n        - <search>query</search> — to retrieve relevant images. Pay attention to the question when formulating your search query.\n        - <bbox>[x1,y1,x2,y2]</bbox> — to zoom into a region for a clearer view (normalized to 0-1000). You cannot use bbox on a cropped image.\n        - <answer>final answer</answer> — only when you have retrieved at least 3 images. You MUST NOT use <answer> before your 3rd search result arrives.\n\n        3. A \"COLLECTED EVIDENCE\" section may appear after the question. It summarizes key findings from your previous reasoning and retrieved images. Use this evidence together with the question and any new images to guide your next step.\n\n        4. When given an image, analyze it fully in <think> </think>, extract any potentially useful information. Only propose <bbox> if critical details are unclear.\n\n        5. NEVER output multiple actions in one response.\n        6. NEVER omit <think>, even for final answers.\n\n        Good examples:\n        <think>I need to find browser requirements for Nordic Swan Ecolabel. I will search for the official portal guide.</think><search>Nordic Ecolabelling Portal browser requirements</search>\n\n        <think>I have now retrieved 3 images. The first image already showed the browser recommendation: \"Use Microsoft Edge or Google Chrome\". I have searched through all retrieved images and no additional relevant information is missing. The answer is complete.</think><answer>Microsoft Edge or Google Chrome</answer>\n\n        The user's question is:\n        " + query
        })

        # ── 遍历 history ──
        for msg in history:
            # assistant: search
            if isinstance(msg, dict) and 'search' in msg:
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{msg["think"]}</think><search>{msg["search"]}</search>'
                })

            # assistant: bbox
            elif isinstance(msg, dict) and 'bbox' in msg:
                if len(conversations) < 2 or 'search' not in conversations[-2].get('content', ''):
                    good_data = False
                    break
                img_path = overall_images[-1] if overall_images else None
                if img_path:
                    try:
                        w, h = Image.open(img_path).size
                        if w * h > 14 * 14 * 4 * 1280 or w * h < 56 * 56:
                            good_data = False
                            break
                    except Exception:
                        good_data = False
                        break
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{msg["think"]}</think><bbox>{json.dumps(msg["bbox"])}</bbox>'
                })

            # assistant: answer
            elif isinstance(msg, dict) and 'answer' in msg:
                conversations.append({
                    "role": "assistant",
                    "content": f'<think>{msg["think"]}</think><answer>{msg["answer"]}</answer>'
                })

            # user: obs（含图 + 文字）—— list 格式 [{'image': path}, {'text': obs_text}]
            elif isinstance(msg, list):
                img_path = None
                obs_text = ''
                for part in msg:
                    if isinstance(part, dict) and 'image' in part:
                        img_path = part['image']
                    if isinstance(part, dict) and 'text' in part:
                        obs_text = part['text']

                if img_path is None:
                    good_data = False
                    break

                overall_images.append(img_path)
                # user 消息：<image> 占位 + obs 文字（和 SFT trainer 读取方式一致）
                conversations.append({
                    "role": "user",
                    "content": f"<image>\n{obs_text}"
                })

        if not good_data:
            skip_count += 1
            continue

        # 必须以 answer 结束
        if not conversations or 'answer' not in conversations[-1].get('content', ''):
            skip_count += 1
            continue

        # 图片数量合法性
        if not (1 <= len(overall_images) <= 12):
            skip_count += 1
            continue

        sft_results.append({
            "messages": conversations,
            "images":   overall_images,
        })

    print(f"Total input: {len(data)}, skipped: {skip_count}, output: {len(sft_results)}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(sft_results, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_json}")


if __name__ == '__main__':
    convert_v2(
        input_jsonl='./SlideVQA-SFT-v2/results/cot_v2_results_correct.jsonl',
        output_json='./data/SlideVQA-SFT-v2.json',
        min_search=3,
    )
