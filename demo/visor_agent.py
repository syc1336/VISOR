import base64
import json
import re
import requests
import math
import os
from io import BytesIO
from PIL import Image
from openai import OpenAI

# ==================== 配置区 ====================
BASE_URL = 'http://0.0.0.0:8001/v1'
SEARCH_URL = 'http://0.0.0.0:8002/search'
MODEL_NAME = 'Qwen/Qwen2.5-VL-7B-Instruct'
MAX_PIXELS = 512 * 28 * 28
MIN_PIXELS = 256 * 28 * 28
MAX_STEPS = 10
PAD_SIZE = 56
CROP_OUTPUT_DIR = "./crop_images"

# ==================== 智能上下文管理器 ====================
class DynamicContextManager:
    def __init__(self, query):
        self.query = query
        # 存储结构：{ "原始图片名": ["思考记录1", "思考记录2"] }
        self.evidence_ledger = {} 
        self.current_root_image = "Initial_Logic" # 追踪当前正在操作的原始图索引

    def set_current_root(self, image_path):
        """当搜索到新图时，更新根索引。bbox操作不触发此函数，从而实现归类。"""
        self.current_root_image = os.path.basename(image_path)
        if self.current_root_image not in self.evidence_ledger:
            self.evidence_ledger[self.current_root_image] = []

    def add_thought_as_evidence(self, text_content):
        """将模型在 <think> 中的思考直接提取并存入当前图的索引下"""
        think_match = re.search(r'<think>(.*?)</think>', text_content, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            if len(thought) > 5:
                # 如果当前没有根图（比如初始推理），则存入初始逻辑
                img_key = self.current_root_image
                if img_key not in self.evidence_ledger:
                    self.evidence_ledger[img_key] = []
                
                # 精简思考内容（截断过长文本以节省Token）
                self.evidence_ledger[img_key].append(thought)

    def get_managed_messages(self, current_messages, max_window=2):
        """构造带有"图像-思考链"结构的精简上下文"""
        ledger_sections = []
        for img, thoughts in self.evidence_ledger.items():
            # 聚合该图片下的所有思考（含bbox后的思考）
            thought_list = "\n".join([f"  - {t}" for t in thoughts])
            ledger_sections.append(f"【Image Source: {img}】\n{thought_list}")
        
        ledger_str = "\n\n".join(ledger_sections) if ledger_sections else "No evidence collected yet."

        system_prompt = (
        f"""
        You are a visual reasoning agent to answer user's question. Follow these rules without exception:

        1. EVERY response MUST begin with exactly one <think> block containing your internal reasoning.
        - Do NOT output anything before <think>.
        - Do NOT skip thinking, even if the answer seems obvious.

        2. AFTER </think>, you must output ONE of the following actions:
        - <search>query</search> — to retrieve relevant images, the file consists of multiple pages, the file name may just be the cover page, you need to pay more attention to the keywords in the question.
        - <bbox>[x1,y1,x2,y2]</bbox> — to zoom into a region for a clearer view (normalized to 0-1000). You can't use bbox on the cropped image.
        - <answer>final answer</answer> —only when you are confident to answer or this is your last response. In your last response, you must use <answer> after <think> </think> as required.

        3. When given an image, analyze it fully in <think>, put any potentially relevant and useful information in your thoughts. Only propose <bbox> if critical details are unclear.

        4. NEVER output multiple actions in one response.
        5. NEVER omit <think>, even for final answers.

        Good examples:
        <think>I need to find browser requirements for Nordic Swan Ecolabel. I will search for the official portal guide.</think><search>Nordic Ecolabelling Portal browser requirements</search>

        <think>The image shows a webpage. The browser recommendation is visible in the top-right corner: "Use Chrome or Edge". No further search needed.</think><answer>Chrome or Edge</answer>.

        "The user's question is \n{self.query}\n\n"

        The COLLECTED EVIDENCE is shown as below:
        """
        )
        
        anchor_text = (
            f"### COLLECTED EVIDENCE\n{ledger_str}\n"
        )
        anchor_msg = {"role": "system", "content": [{"type": "text", "text": system_prompt + anchor_text}]}

        # 滑动窗口：只保留最近的几轮原始视觉交互
        window_size = max_window * 2
        recent_msgs = current_messages[-window_size:] if len(current_messages) > window_size else current_messages

        return [anchor_msg] + recent_msgs

# ==================== 工具函数 ====================
def process_image(img):
    """处理图像，使其适应模型输入尺寸"""
    if not isinstance(img, Image.Image):
        img = Image.open(img)
    
    w, h = img.size
    if w * h > MAX_PIXELS:
        r = math.sqrt(MAX_PIXELS / (w * h))
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    elif w * h < MIN_PIXELS:
        r = math.sqrt(MIN_PIXELS / (w * h))
        img = img.resize((int(w * r), int(h * r)), Image.LANCZOS)
    
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return f"data:image;base64,{base64.b64encode(buf.getvalue()).decode()}"

def search_images(query, search_url):
    try:
        resp = requests.get(search_url, params={"queries": [query]}, timeout=30)
        resp.raise_for_status()
        return [item['image_file'] for item in resp.json()[0]]
    except:
        return []


def save_crop(image, step, index):
    os.makedirs(CROP_OUTPUT_DIR, exist_ok=True)
    crop_path = os.path.join(CROP_OUTPUT_DIR, f"crop_step{step}_{index}.jpg")
    image.save(crop_path, format="JPEG")
    return crop_path

# ==================== 核心推理引擎 ====================
def run_one_question(question, client, search_url, model_name, max_steps):
    ctx = DynamicContextManager(question)
    current_messages = [{"role": "user", "content": [{"type": "text", "text": f"Question: {question}"}]}]
    trace = []
    seen_paths = []
    
    raw_imgs = []   # 原始图像（搜索或裁剪得到的图像）

    for step in range(max_steps):
        # 1. 最后一轮的强制逻辑：如果已经是最后一步且还没出结果，强制要求回答
        is_last_step = (step >= max_steps - 1)
        
        managed_msgs = ctx.get_managed_messages(current_messages)
        # print(managed_msgs)
        
        # 如果是最后一轮，在 context 之后追加一个强力指令
        if is_last_step:
            last_user_prompt = f"I can not retrieval something about the question. This is your last response and you just can based on the COLLECTED EVIDENCE above, provide your final answer for the question: [{question}] inside <answer> </answer> tags after <think> </think>."
    
            # 找到最后一条 role 为 user 的消息并替换其内容
            # 倒序遍历，确保修改的是最后一次用户输入
            for msg in reversed(managed_msgs):
                if msg["role"] == "user":
                    msg["content"] = [{"type": "text", "text": last_user_prompt}]
                    break
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=managed_msgs,
                temperature=0.1,
                max_tokens=2048,
                
            )
        except Exception as e:
            trace.append({"type": "error", "content": str(e)})
            break

        
        text = resp.choices[0].message.content
        if "</think>" in text and "<think>" not in text:
            text = "<think>" + text
        ctx.add_thought_as_evidence(text)
        
        # 记录 trace 和消息历史
        current_messages.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        trace.append({"step": step, "role": "assistant", "content": text})

        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if think_match:
            yield "think", think_match.group(1).strip(), think_match.group(0)

        # 2. 检查是否给出答案
        ans_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if ans_match:
            final_result = ("answer", ans_match.group(1).strip(), ans_match.group(0))
            return final_result, trace, ctx.evidence_ledger

        # 如果已经是最后一步但模型还是没给 <answer>，我们可以手动截取内容或返回错误
        if is_last_step:
            break

        # 3. 处理动作 (search/bbox)
        act_match = re.search(r'<(search|bbox)>(.*?)</\1>', text, re.DOTALL)
        if not act_match:
            current_messages.append({"role": "user", "content": [{"type": "text", "text": "Please continue reasoning or provide an <answer>."}]})
            continue

        action, payload = act_match.group(1), act_match.group(2).strip()
        yield action, payload, act_match.group(0)

        if action == "search":
            paths = search_images(payload, search_url)
            path = next((p for p in paths if p not in seen_paths), None) or (paths[0] if paths else None)
            
            if path:
                ctx.set_current_root(path) 
                seen_paths.append(path)
                raw_img = Image.open(path)
                raw_imgs.append(raw_img)
                
                # 处理图像用于显示
                b64 = process_image(raw_img)
                
                current_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64}},
                        {"type": "text", "text": f"Image loaded, analyze any possible useful information for the question: [{question}] in your think, then continue your action after <think> </think>."}
                    ]
                })
                trace.append({"step": step, "type": "search", "path": path})
                yield "search_image", path, path
            else:
                current_messages.append({"role": "user", "content": [{"type": "text", "text": "No results found."}]})

        elif action == "bbox" and raw_imgs:
            try:
                # 解析bbox坐标
                bbox = json.loads(payload)
                if len(bbox) != 4:
                    raise ValueError("bbox must have 4 values")
                
                x1, y1, x2, y2 = map(float, bbox)
                
                latest_raw = raw_imgs[-1]
                rw, rh = latest_raw.size

                # Map normalized 0-1000 coordinates to the original image, then add pixel padding.
                real = (
                    max((x1 / 1000 * rw) - PAD_SIZE, 0),
                    max((y1 / 1000 * rh) - PAD_SIZE, 0),
                    min((x2 / 1000 * rw) + PAD_SIZE, rw),
                    min((y2 / 1000 * rh) + PAD_SIZE, rh),
                )
                
                # 从原始图像上裁剪
                cropped = latest_raw.crop(real)
                
                raw_imgs.append(cropped)
                
                # 处理裁剪后的图像用于显示
                b64 = process_image(cropped)
                
                current_messages.append({
                    "role": "user", 
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64}},
                        {"type": "text", "text": f"This is the cropped image, analyse it based on the question [{question}], and after <think> </think> you just can use <search> or <answer> this time. If the cropped image is incorrect, please pay attention to previous think."}
                    ]
                })
                trace.append({"step": step, "type": "bbox", "coords": bbox})
                crop_path = save_crop(cropped, step, len(raw_imgs))
                yield "crop_image", crop_path, crop_path

            except json.JSONDecodeError:
                # 如果payload不是JSON格式，尝试解析其他格式
                try:
                    clean_payload = payload.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
                    coords = [float(x) for x in clean_payload.split(',')]
                    if len(coords) == 4:
                        latest_raw = raw_imgs[-1]
                        w, h = latest_raw.size
                        x1, y1, x2, y2 = coords
                        box = (
                            max((x1 / 1000 * w) - PAD_SIZE, 0),
                            max((y1 / 1000 * h) - PAD_SIZE, 0),
                            min((x2 / 1000 * w) + PAD_SIZE, w),
                            min((y2 / 1000 * h) + PAD_SIZE, h),
                        )
                        cropped = latest_raw.crop(box)
                        b64 = process_image(cropped)
                        raw_imgs.append(cropped)
                        
                        current_messages.append({
                            "role": "user", 
                            "content": [
                                {"type": "image_url", "image_url": {"url": b64}},
                                {"type": "text", "text": f"This is the cropped image, analyse it based on the question [{question}], and after <think> </think> you just can use <search> or <answer> this time. If the cropped image is incorrect, please pay attention to previous think and evidence."}
                            ]
                        })
                        trace.append({"step": step, "type": "bbox", "coords": coords})
                        crop_path = save_crop(cropped, step, len(raw_imgs))
                        yield "crop_image", crop_path, crop_path
                    else:
                        raise ValueError("Invalid bbox format")
                except Exception as e:
                    current_messages.append({"role": "user", "content": [{"type": "text", "text": f"BBox error: {str(e)}"}]})
            except Exception as e:
                current_messages.append({"role": "user", "content": [{"type": "text", "text": f"BBox error: {str(e)}"}]})

    # 如果跳出循环仍无答案，返回一个保底提示
    final_result = ("answer", "Failed to get answer within max steps", "")
    return final_result, trace, ctx.evidence_ledger

class VISOR:
    def __init__(
        self,
        base_url=BASE_URL,
        search_url=SEARCH_URL,
        model_name=MODEL_NAME,
        api_key="EMPTY",
        max_steps=MAX_STEPS,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.search_url = search_url
        self.model_name = model_name
        self.max_steps = max_steps
        self.trace = []
        self.evidence_ledger = {}

    def run(self, question):
        final_result, self.trace, self.evidence_ledger = yield from run_one_question(
            question=question,
            client=self.client,
            search_url=self.search_url,
            model_name=self.model_name,
            max_steps=self.max_steps,
        )
        return final_result


def main(question):
    agent = VISOR(
        base_url=BASE_URL,
        search_url=SEARCH_URL,
        model_name=MODEL_NAME,
    )
    generator = agent.run(question)
    try:
        while True:
            action, content, raw_content = next(generator)
            if action in ("search_image", "crop_image"):
                print(content)
            else:
                print(f"[{action}] {content}")
    except StopIteration as result:
        action, answer, _ = result.value
        print(f"[{action}] {answer}")


if __name__ == "__main__":
    main("How much is the Trading Operating Profit in 2011?")
