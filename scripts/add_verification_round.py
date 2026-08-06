"""
为 SlideVQA-SFT 数据添加 verification 轮。

流程：
1. 找到每条样本最后一条 assistant 消息（含 <think> + <answer>）
2. 把 think 末尾加上 "I want to do a verification round, so I will search again."
3. 把 <answer>...</answer> 替换为新的 <search>...</search>（搜索关键词从问题中提取）
4. 随机挑一张同文件、未出现过的页作为 verification 图片
5. 调用 qwen3-vl-235b 判断新图是否相关，并生成 verification think 文本
6. 追加新的 user(<image>) + assistant(think + answer) 轮次
7. 写出新 json

API 调用只用于：判断新图相关性 + 生成 verification think 文本
"""

import os
import re
import json
import base64
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# ─── 配置 ───────────────────────────────────────────────────────────────
OPENAI_API_KEY  = "REDACTED_API_KEY"
OPENAI_BASE_URL = "https://www.huayanapi.com/v1/"
MODEL_NAME      = "qwen3-vl-235b-a22b-instruct"

INPUT_JSON  = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-new.json"
OUTPUT_JSON = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-verification.json"
IMG_DIRS    = [
    "/mnt/cfs_algo_bj/workspace/shenyucheng/SlideVQA-SFT/img",
    "/mnt/cfs_algo_bj/models/experiments/shenyucheng/SlideVQA-SFT2/img",
]

MAX_WORKERS = 20   # 并行线程数
MAX_RETRY   = 3

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

# ─── 工具函数 ────────────────────────────────────────────────────────────

def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_think(text: str) -> str:
    """提取 <think>...</think> 内容（不含标签）"""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_answer(text: str) -> str:
    """提取 <answer>...</answer> 内容（不含标签）"""
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    return m.group(1).strip() if m else ""


def get_file_prefix(img_path: str) -> str:
    """
    从图片路径中提取文件前缀（去掉页码部分）。
    例: accel-india-ecommerce_95_4.png → accel-india-ecommerce_95
    """
    basename = os.path.splitext(os.path.basename(img_path))[0]
    # 去掉最后的 _数字
    m = re.match(r'^(.+_95)_\d+$', basename)
    return m.group(1) if m else None


def get_used_pages(images: list) -> set:
    """从 images 列表中解析已使用的页码集合"""
    pages = set()
    for img in images:
        basename = os.path.splitext(os.path.basename(img.rstrip('/')))[0]
        m = re.search(r'_95_(\d+)$', basename)
        if m:
            pages.add(int(m.group(1)))
    return pages


def pick_verification_image(images: list) -> str | None:
    """
    从同文件（同前缀）下随机挑一张未使用过的页图片。
    在所有 IMG_DIRS 中搜索，返回完整路径，找不到返回 None。
    """
    # 以第一张真实页面图（非 crop）为基准取前缀
    ref_img = None
    for img in images:
        if img.startswith("/mnt"):
            ref_img = img
            break
    if ref_img is None:
        return None

    prefix = get_file_prefix(ref_img)
    if prefix is None:
        return None

    used_pages = get_used_pages(images)

    # 在所有图片目录中扫描同前缀的未用页
    candidates = []
    for img_dir in IMG_DIRS:
        if not os.path.isdir(img_dir):
            continue
        for fname in os.listdir(img_dir):
            if not fname.endswith(".png"):
                continue
            m = re.match(rf'^{re.escape(prefix)}_(\d+)\.png$', fname)
            if m:
                page = int(m.group(1))
                if page not in used_pages:
                    candidates.append(os.path.join(img_dir, fname))

    if not candidates:
        return None
    return random.choice(candidates)


def extract_question(user_content: str) -> str:
    """从第一条 user 消息中提取 Question: 后面的内容"""
    m = re.search(r'Question:\s*(.+)', user_content)
    return m.group(1).strip() if m else user_content[-200:]


def generate_search_query(question: str) -> str:
    """从问题生成简洁的搜索关键词（规则，无需 API）"""
    # 取问题前 80 个字符作为搜索词，去掉疑问词
    q = re.sub(r'^(how|what|which|when|where|who|why|is|are|does|did)\s+', '', question.lower()).strip()
    return q[:80]


VERIFICATION_PROMPT = """You are analyzing a new retrieved image to verify a previously found answer.

Question: {question}
Previously found answer: {answer}

Please analyze the new image carefully:
1. Is this image relevant to the question above?
2. If relevant, does it support, contradict, or add to the previous answer?

Respond in exactly this format (one paragraph, no extra text):
If RELEVANT: "The new image shows [brief description of what you see]. This [further confirms / adds to / contradicts] the answer that [restate the answer]."
If NOT RELEVANT: "The new image does not contain information relevant to the question. Based on the previous retrieval, the information is already complete."
"""


def call_verification_api(question: str, answer: str, new_img_path: str) -> str:
    """
    调用 qwen3-vl 判断新图是否相关，返回 verification think 文本。
    失败时返回默认 not-relevant 文本。
    """
    default = ("The new image does not contain additional relevant information. "
               "The previous retrieval is already complete.")
    try:
        b64 = encode_image(new_img_path)
        for attempt in range(MAX_RETRY):
            try:
                resp = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"}
                            },
                            {
                                "type": "text",
                                "text": VERIFICATION_PROMPT.format(
                                    question=question,
                                    answer=answer
                                )
                            }
                        ]
                    }],
                    max_tokens=200,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if attempt < MAX_RETRY - 1:
                    time.sleep(5)
    except Exception:
        pass
    return default


def is_relevant(think_text: str) -> bool:
    """从 verification think 文本判断是否相关"""
    lower = think_text.lower()
    return "does not contain" not in lower and "not relevant" not in lower


# ─── 主处理逻辑（单条样本） ────────────────────────────────────────────────

def process_sample(sample: dict) -> dict:
    messages = sample["messages"]
    images   = list(sample["images"])

    # 找第一条 user 消息提取问题
    question = extract_question(messages[0]["content"])

    # 找最后一条 assistant 消息
    last_asst_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx]["role"] == "assistant":
            last_asst_idx = idx
            break
    if last_asst_idx is None:
        return sample  # 异常样本，跳过

    last_content = messages[last_asst_idx]["content"]
    original_think  = extract_think(last_content)
    original_answer = extract_answer(last_content)

    if not original_answer:
        return sample  # 没有 answer，跳过

    # ── Step 1: 改写最后一条 assistant 消息 ──
    # think 末尾加 verification 意图
    new_think = (original_think +
                 " I want to do a verification round, so I will search again.")
    search_query = generate_search_query(question)
    new_last_content = (f"<think>{new_think}</think>\n"
                        f"<search>{search_query}</search>")
    messages[last_asst_idx]["content"] = new_last_content

    # ── Step 2: 挑 verification 图片 ──
    new_img_path = pick_verification_image(images)
    if new_img_path is None:
        # 找不到新图，回滚改动，保留原样本
        messages[last_asst_idx]["content"] = last_content
        return sample

    # ── Step 3: 调用 API 生成 verification think ──
    verification_think = call_verification_api(question, original_answer, new_img_path)
    relevant = is_relevant(verification_think)

    # ── Step 4: 构造 verification 轮的 assistant 回复 ──
    if relevant:
        final_content = (f"<think>{verification_think} "
                         f"The answer is confirmed.</think>\n"
                         f"<answer>{original_answer}</answer>")
    else:
        final_content = (f"<think>{verification_think} "
                         f"The information is already complete, I can now provide the answer.</think>\n"
                         f"<answer>{original_answer}</answer>")

    # ── Step 5: 追加新轮次 ──
    messages.append({"role": "user",    "content": "<image>"})
    messages.append({"role": "assistant", "content": final_content})
    images.append(new_img_path)

    return {"messages": messages, "images": images}


# ─── 入口 ────────────────────────────────────────────────────────────────

CHECKPOINT = OUTPUT_JSON + ".ckpt.jsonl"  # 断点文件，jsonl 格式

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ── 断点续跑：从 checkpoint 文件读已完成条数 ──
    done_count = 0
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    done_count += 1

    todo = data[done_count:]
    print(f"总样本数: {len(data)}，已完成: {done_count}，待处理: {len(todo)}")

    ckpt_f = open(CHECKPOINT, "a", encoding="utf-8")
    lock = __import__("threading").Lock()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(process_sample, sample): (done_count + idx)
                   for idx, sample in enumerate(todo)}
        with tqdm(total=len(todo), desc="处理中") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    print(f"[ERROR] 样本 {idx} 处理失败: {e}")
                    result = data[idx]
                with lock:
                    ckpt_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    ckpt_f.flush()
                pbar.update(1)

    ckpt_f.close()

    # ── 全部完成后，合并 checkpoint → 标准 JSON 数组 ──
    results = []
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"完成，输出至: {OUTPUT_JSON}（共 {len(results)} 条）")


if __name__ == "__main__":
    main()
