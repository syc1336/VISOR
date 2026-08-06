"""
对 cot_v3_results.jsonl 中的 think 进行重写。

目标：history 里第一个 dict（search 前的 think）常常只写
"I need to perform a final verification search..."，缺少对图像的分析。
本脚本调用 VLM，给它看对应图像 + 问题，重写成：
  1. 先描述图像内容
  2. 说明信息完整/不完整
  3. 决定下一步（验证搜索 or 答案）
"""

import json
import base64
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

OPENAI_API_KEY  = "REDACTED_API_KEY"
OPENAI_BASE_URL = "https://www.huayanapi.com/v1/"
VLM_MODEL       = "qwen3-vl-235b-a22b-instruct"

INPUT_FILE  = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/SlideVQA-SFT-v3/results/cot_v3_results_correct.jsonl"
OUTPUT_FILE = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/SlideVQA-SFT-v3/results/cot_v3_results_rewrite.jsonl"

MAX_WORKERS = 20

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


def img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def get_image_size(path: str):
    try:
        with Image.open(path) as img:
            return img.size  # (w, h)
    except Exception:
        return None


def need_rewrite(think: str) -> bool:
    """判断 think 是否属于空洞的验证套话，需要重写。"""
    triggers = [
        "I need to perform a final verification",
        "I must perform a final verification",
        "I will perform a final verification",
        "I need to verify",
        "perform a final verification search",
        "do one final verification search",
        "final verification search using the original question",
    ]
    tl = think.lower()
    return any(t.lower() in tl for t in triggers) and len(think) < 400


REWRITE_SYSTEM = """You are a visual reasoning assistant. You will be shown a slide image and a question.
Your task is to write a concise <think> passage that:
1. Briefly describes the key relevant content visible in the image (text, numbers, charts, etc.)
2. Explains what information this image provides toward answering the question
3. Concludes that you now have sufficient information to answer the question
4. Ends with: "I will do one final verification search to confirm my answer."

Output ONLY the think text (no JSON, no tags, no extra commentary). Be specific and concise."""


def rewrite_think(image_path: str, question: str, answer: str) -> str:
    """调用 VLM 重写 think 文本。"""
    b64 = img_to_b64(image_path)
    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = f"image/{'jpeg' if ext in ('jpg', 'jpeg') else 'png'}"

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": (
            f"Question: {question}\n"
            f"Correct answer: {answer}\n\n"
            "Write the think passage as described. The answer has already been confirmed — "
            "your think should reflect that the image provides sufficient evidence."
        )},
    ]

    resp = client.chat.completions.create(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        max_tokens=512,
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def process_sample(sample: dict) -> dict:
    """处理单条样本，重写需要改写的 think。"""
    history = sample.get("history", [])
    question = sample.get("query", "")
    answer = sample.get("reference_answer", sample.get("response", ""))
    initial_image = sample.get("initial_image", "")

    changed = False

    # history 结构: [think_dict, obs_list, think_dict, obs_list, ...]
    # think_dict: {"think": ..., "search": ...} 或 {"think": ..., "answer": ...}
    # obs_list:   [{"image": ...}, {"text": ...}]
    for i, item in enumerate(history):
        if not isinstance(item, dict):
            continue
        think = item.get("think", "")
        if not need_rewrite(think):
            continue

        # 找这条 think 对应的图像：它是对前一个 obs 的响应
        # obs 在 history[i-1]（如果存在）
        image_path = None
        if i > 0 and isinstance(history[i - 1], list):
            obs = history[i - 1]
            for part in obs:
                if isinstance(part, dict) and "image" in part:
                    image_path = part["image"]
                    break
        elif i == 0:
            # 第一个 think 对应初始图
            image_path = initial_image

        if not image_path or not os.path.exists(image_path):
            continue

        # 确定 next_action 描述（仅用于日志，不传给 VLM）
        if "search" in item:
            next_action = f"search(\"{item['search']}\")"
        elif "answer" in item:
            next_action = f"answer(\"{item['answer']}\")"
        else:
            next_action = "unknown"

        try:
            new_think = rewrite_think(image_path, question, answer)
            if new_think:
                item["think"] = new_think
                changed = True
        except Exception as e:
            print(f"[WARN] rewrite failed for {sample.get('uid')}: {e}")

    return sample


def main():
    # 读取已完成的 uid，支持断点续跑
    done_uids = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        uid = json.loads(line).get("uid")
                        if uid:
                            done_uids.add(uid)
                    except Exception:
                        pass
        print(f"Resuming: {len(done_uids)} samples already done")

    samples = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                s = json.loads(line)
                if s.get("uid") not in done_uids:
                    samples.append(s)

    print(f"Loaded {len(samples)} samples to process")

    import threading
    write_lock = threading.Lock()

    def process_and_save(sample: dict):
        result = process_sample(sample)
        with write_lock:
            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_and_save, s) for s in samples]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rewriting"):
            try:
                fut.result()
            except Exception as e:
                print(f"[ERROR] {e}")

    print(f"Done. Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
