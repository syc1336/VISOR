"""
SFT 格式转换 v3 —— 对齐 data_construct_pipeline_v3 输出

输入：cot_v3_results_correct.jsonl（经 filter 筛选后的正确样本）
输出：data/SlideVQA-SFT-v3.json（供 SFT trainer 直接使用）

与 v2 的主要差异：
  - min_search ≥ 2（至少 1 次普通搜索 + 1 次 verification search）
  - 过滤掉 obs 文本中的 [DATA CONSTRUCTION HINT]... 标记（仅用于数据构造阶段）
    包括：参考页提示、搜全提示、验证图提示，strip 后均还原为纯 obs 文本
  - USER_PROMPT 对齐 hf_dataset_convert.py 当前版本

消息格式（Qwen2.5-VL SFT 标准）：
  [0] user:      <image>\nUSER_PROMPT（初始图 + 问题）
  [1] assistant: <think>...</think><search>query</search>
  [2] user:      <image>\nImage loaded, analyze...
  ...
  [N-2] assistant: <think>...</think><search>query</search>   ← verification search
  [N-1] user:      <image>\nImage loaded, analyze...           ← verification obs（hint 已剥离）
  [N]   assistant: <think>...</think><answer>XXX</answer>
"""

import json
import re
from PIL import Image

# ── 对齐 hf_dataset_convert.py 当前 USER_PROMPT ──
USER_PROMPT_TEMPLATE = '''You are a visual reasoning agent. An initial reference image has been retrieved for you below.

        Rules:
        1. Every response must start with <think> </think> where you reason about what you see and what to do next.

        2. After thinking, output exactly one action:
           - <search>query</search> to retrieve more images. Each search returns one new image; if you repeat a query, you will get a different image from the same document. Use the original question as your query unless you have a specific reason to change it.
           - <bbox>[x1,y1,x2,y2]</bbox> to zoom into an unclear region (normalized to 0-1000, only on full images).
           - <answer>your answer</answer> once you are ready to give your final answer.

        3. Before answering, you must do one final search using the original question to verify your answer. After receiving the new image, give your <answer> immediately — unless the new image provides a directly conflicting answer to the question, in which case search once more and then give your <answer> immediately regardless.

        4. When given an image, analyze it fully in <think> </think> and extract every potentially useful piece of information — your thoughts will be recorded into a COLLECTED EVIDENCE table for later reference, so be as thorough as possible. If the image contains no relevant information, explicitly state that (e.g. "This image does not contain information related to the question.") to avoid polluting the evidence. Only propose <bbox> if critical details are visually unclear.

        A "COLLECTED EVIDENCE" section may appear summarizing findings from previous steps — use it alongside new images to guide your reasoning.


        Good examples:

        <think>This image shows the Nordic Ecolabelling Portal login page. The page clearly states that Microsoft Edge or Google Chrome is recommended as the web browser. I have now gathered enough information to answer the question. Before giving my final answer, I will do one last verification search.</think><search>Apply for Nordic Swan Ecolabel license, what is recommended as a web browser according to the Nordic Ecolabelling Portal instructions?</search>

        <think>The final search returned an image that does not contradict my current answer. My answer stands.</think><answer>Microsoft Edge or Google Chrome</answer>

        The user's question is:
        {question}
        '''

# 匹配并去除 [DATA CONSTRUCTION HINT] 及其后内容
_HINT_RE = re.compile(r'\s*\[DATA CONSTRUCTION HINT\].*', re.DOTALL)


def strip_hint(obs_text: str) -> str:
    """去除 build_obs_str 注入的 [DATA CONSTRUCTION HINT] 段落。"""
    return _HINT_RE.sub('', obs_text).strip()


def convert_v3(input_jsonl: str, output_json: str, min_search: int = 1):
    with open(input_jsonl, 'r') as f:
        data = [json.loads(l) for l in f if l.strip()]

    sft_results = []
    skip_count  = 0

    for item in data:
        query         = item['query']
        history       = item['history']
        initial_image = item.get('initial_image')

        # 没有初始图的样本直接跳过
        if initial_image is None:
            skip_count += 1
            continue

        # ── 搜索次数检查（至少有 1 次 verification search）──
        n_search = sum(
            1 for h in history
            if isinstance(h, dict) and 'search' in h
        )
        if n_search < min_search:
            skip_count += 1
            continue

        conversations = []
        overall_images = [initial_image]   # 初始图排在所有图的第一位
        good_data = True

        # ── 首条 user 消息：初始图 + USER_PROMPT（含问题）+ 初始图 obs 文本 ──
        INIT_OBS = (
            '[Initial retrieval using your question] '
            'This is the initial reference image. '
            'Analyze it, then begin your reasoning and actions.'
        )
        conversations.append({
            "role": "user",
            "content": (
                f"{USER_PROMPT_TEMPLATE.format(question=query).strip()}"
                f"\n<image>\n{INIT_OBS}"
            )
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
                # 去除 DATA CONSTRUCTION HINT，保留纯 obs 文本
                clean_obs = strip_hint(obs_text)
                conversations.append({
                    "role": "user",
                    "content": f"<image>\n{clean_obs}"
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
    convert_v3(
        input_jsonl='./SlideVQA-SFT-v3/results/cot_v3_results_rewrite.jsonl',
        output_json='./data/SlideVQA-SFT-v3.json',
        min_search=1,
    )
