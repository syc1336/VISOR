import json
import os
import re
import concurrent.futures
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# ================= 配置 =================
# OPENAI_API_KEY = "REDACTED_API_KEY"
# OPENAI_BASE_URL = "https://qianfan.baidubce.com/v2"
OPENAI_API_KEY = "REDACTED_API_KEY"
OPENAI_BASE_URL = "https://www.huayanapi.com/v1/"
MODEL_NAME = "deepseek-v3.2" 

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

def is_answer_correct(query: str, reference: str, response: str) -> bool:
    messages = [
        {"role": "system", "content": "你是一个评判专家。只输出 JSON: {\"correct\": true/false}"},
        {"role": "user", "content": f"问题：{query}\n标准答案：{reference}\n回答：{response}"}
    ]
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.1,
            max_tokens=50,
        )
        content = completion.choices[0].message.content
        # 简单粗暴判断文本内容，增加容错
        return '"correct": true' in content.lower()
    except Exception as e:
        # print(f"API Error: {e}") # 调试用
        return False

def filter_correct_with_llm(input_path: str, output_path: str, max_workers: int = 5):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # 1. 预检查：测试 API
    print("🔍 正在测试 API 连通性...")
    if is_answer_correct("1+1=2?", "正确", "2"):
        print("✅ API 测试通过")
    else:
        print("❌ API 测试失败，请检查 Key 或网络！")
        return

    with input_path.open('r', encoding='utf-8') as f:
        lines = [l for l in f if l.strip()]
    
    os.makedirs(output_path.parent, exist_ok=True)
    correct_count = 0

    print(f"🚀 开始筛选，共有 {len(lines)} 条数据")

    # 2. 使用 'a' 模式追加，确保安全
    with output_path.open('a', encoding='utf-8') as f_out:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 这里的 map 会按顺序返回结果，也可以改为 as_completed
            future_to_line = {executor.submit(is_answer_correct, 
                                            json.loads(line)["query"], 
                                            json.loads(line)["reference_answer"], 
                                            json.loads(line)["response"]): line for line in lines}
            
            for future in tqdm(concurrent.futures.as_completed(future_to_line), total=len(lines)):
                line = future_to_line[future]
                try:
                    is_correct = future.result()
                    if is_correct:
                        f_out.write(line.strip() + '\n')
                        # 3. 强制刷入硬盘：即便断电数据也在
                        f_out.flush() 
                        os.fsync(f_out.fileno()) 
                        correct_count += 1
                except Exception as e:
                    continue

    print(f"\n🎉 筛选完成，保存了 {correct_count} 条数据。")

if __name__ == "__main__":
    INPUT = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/SlideVQA-SFT-v3/results/cot_v3_results.jsonl"
    OUTPUT = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/SlideVQA-SFT-v3/results/cot_v3_results_correct.jsonl"
    
    filter_correct_with_llm(INPUT, OUTPUT, max_workers=20)