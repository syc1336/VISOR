"""
将 bbox 轨迹（bbox_results_correct.jsonl）转换为 SFT 格式并合并进 SlideVQA-SFT-v2.json

使用方式：
  1. python data_construct_bbox.py          # 生成 bbox_results.jsonl
  2. python filter.py (改 INPUT/OUTPUT 路径) # 筛选答案正确的
  3. python merge_bbox_sft.py               # 合并进 SFT-v2.json
"""

import json
import os
from PIL import Image


BBOX_JSONL   = './SlideVQA-SFT-bbox/results/bbox_results_correct.jsonl'
EXISTING_SFT = '/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-v2.json'
OUTPUT_SFT   = '/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-v2.json'  # 原地覆盖

# 与 cot_convert_sft_v2.py 完全一致的 user prompt 拼接
USER_PROMPT_PREFIX = (
    "You are a visual reasoning agent to answer user's question. Follow these rules without exception:\n\n"
    "        1. EVERY response MUST begin with exactly one <think> </think> block containing your internal reasoning.\n"
    "        - Do NOT output anything before <think>.\n"
    "        - Do NOT skip thinking, even if the answer seems obvious.\n\n"
    "        2. AFTER <think> </think>, you must output ONE of the following actions:\n"
    "        - <search>query</search> — to retrieve relevant images. Pay attention to the question when formulating your search query.\n"
    "        - <bbox>[x1,y1,x2,y2]</bbox> — to zoom into a region for a clearer view (normalized to 0-1000). You cannot use bbox on a cropped image.\n"
    "        - <answer>final answer</answer> — only when you have retrieved at least 3 images. You MUST NOT use <answer> before your 3rd search result arrives.\n\n"
    "        3. A \"COLLECTED EVIDENCE\" section may appear after the question. It summarizes key findings from your previous reasoning and retrieved images. Use this evidence together with the question and any new images to guide your next step.\n\n"
    "        4. When given an image, analyze it fully in <think> </think>, extract any potentially useful information. Only propose <bbox> if critical details are unclear.\n\n"
    "        5. NEVER output multiple actions in one response.\n"
    "        6. NEVER omit <think>, even for final answers.\n\n"
    "        Good examples:\n"
    "        <think>I need to find browser requirements for Nordic Swan Ecolabel. I will search for the official portal guide.</think><search>Nordic Ecolabelling Portal browser requirements</search>\n\n"
    "        <think>I have now retrieved 3 images. The first image already showed the browser recommendation: \"Use Microsoft Edge or Google Chrome\". I have searched through all retrieved images and no additional relevant information is missing. The answer is complete.</think><answer>Microsoft Edge or Google Chrome</answer>\n\n"
    "        The user's question is:\n        "
)


def convert_bbox_item(item: dict) -> dict | None:
    query   = item['query']
    history = item['history']

    n_search = sum(1 for h in history if isinstance(h, dict) and 'search' in h)
    n_bbox   = sum(1 for h in history if isinstance(h, dict) and 'bbox' in h)

    if n_search < 3 or n_bbox < 1:
        return None

    conversations  = []
    overall_images = []

    conversations.append({
        "role": "user",
        "content": USER_PROMPT_PREFIX + query
    })

    for msg in history:
        if isinstance(msg, dict) and 'search' in msg:
            conversations.append({
                "role": "assistant",
                "content": f'<think>{msg["think"]}</think><search>{msg["search"]}</search>'
            })

        elif isinstance(msg, dict) and 'bbox' in msg:
            # 验证 crop 图片存在且尺寸合理
            img_path = overall_images[-1] if overall_images else None
            if img_path:
                try:
                    w, h = Image.open(img_path).size
                    if w * h > 14 * 14 * 4 * 1280 or w * h < 56 * 56:
                        return None
                except Exception:
                    return None
            conversations.append({
                "role": "assistant",
                "content": f'<think>{msg["think"]}</think><bbox>{json.dumps(msg["bbox"])}</bbox>'
            })

        elif isinstance(msg, dict) and 'answer' in msg:
            conversations.append({
                "role": "assistant",
                "content": f'<think>{msg["think"]}</think><answer>{msg["answer"]}</answer>'
            })

        elif isinstance(msg, list):
            img_path = None
            obs_text = ''
            for part in msg:
                if isinstance(part, dict) and 'image' in part:
                    img_path = part['image']
                if isinstance(part, dict) and 'text' in part:
                    obs_text = part['text']
            if img_path is None:
                return None
            overall_images.append(img_path)
            conversations.append({
                "role": "user",
                "content": f"<image>\n{obs_text}"
            })

    if not conversations or 'answer' not in conversations[-1].get('content', ''):
        return None
    if not (1 <= len(overall_images) <= 12):
        return None

    return {"messages": conversations, "images": overall_images}


def main():
    # 1. 加载已有 SFT 数据
    with open(EXISTING_SFT, 'r') as f:
        existing = json.load(f)
    print(f"已有 SFT 样本: {len(existing)}")

    # 2. 转换 bbox jsonl
    with open(BBOX_JSONL, 'r') as f:
        raw = [json.loads(l) for l in f if l.strip()]
    print(f"bbox 原始样本: {len(raw)}")

    new_items = []
    for item in raw:
        converted = convert_bbox_item(item)
        if converted:
            new_items.append(converted)
    print(f"bbox 转换成功: {len(new_items)}")

    # 3. 统计新数据里 bbox action 数量
    total_bbox_actions = sum(
        1 for it in new_items
        for msg in it['messages']
        if msg['role'] == 'assistant' and '<bbox>' in msg.get('content', '')
    )
    print(f"新数据中 bbox action 共: {total_bbox_actions} 条")

    # 4. 合并并写回
    merged = existing + new_items
    with open(OUTPUT_SFT, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"合并后总样本: {len(merged)}，已写入 {OUTPUT_SFT}")


if __name__ == '__main__':
    main()
