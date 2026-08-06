import os
import json
import pandas as pd
from tqdm import tqdm
import io
from PIL import Image

def process_dataset(parquet_dir, image_save_path, json_output_name):
    # 创建图片存放目录
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    all_examples = []
    processed_decks = set()
    
    # 筛选包含 'train' 的 parquet 文件
    files = [f for f in os.listdir(parquet_dir) if f.endswith('.parquet') and 'train' in f.lower()]
    
    for file in files:
        print(f"\n--- 正在读取文件: {file} ---")
        file_path = os.path.join(parquet_dir, file)
        df = pd.read_parquet(file_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file}"):
            deck_name = str(row['deck_name'])
            
            # 1. 保存图片逻辑
            if deck_name not in processed_decks:
                # 检查该 deck 的第一张图是否存在，作为是否已提取过的初步判断
                # 或者更严谨地，在内部循环中逐一判断
                for i in range(1, 21):
                    img_name = f"{deck_name}_{i}.png"
                    save_full_path = os.path.join(image_save_path, img_name)
                    
                    # 【核心修改】：如果本地已存在该图片，直接跳过提取逻辑
                    if os.path.exists(save_full_path):
                        continue
                    
                    col_name = f'page_{i}'
                    img_data = row.get(col_name)
                    
                    if img_data is not None:
                        try:
                            if isinstance(img_data, dict) and 'bytes' in img_data:
                                img_obj = Image.open(io.BytesIO(img_data['bytes']))
                                img_obj.save(save_full_path)
                            elif hasattr(img_data, 'save'):
                                img_data.save(save_full_path)
                            elif isinstance(img_data, bytes):
                                Image.open(io.BytesIO(img_data)).save(save_full_path)
                        except Exception:
                            pass 
                
                processed_decks.add(deck_name)

            # 2. 构造严格格式的 JSON
            # 转换 evidence_pages 为 list
            raw_pages = list(row.get('evidence_pages'))
            # 确保 list 内部全是 Python 原生 int
            clean_pages = [int(p) for p in raw_pages] if isinstance(raw_pages, list) else []
            

            example = {
                "uid": f"{deck_name}_{row['qa_id']}",
                "query": str(row.get('question', '')),
                "reference_answer": str(row.get('answer', '')),
                "meta_info": {
                    "file_name": f"{deck_name}.pdf", # 模拟原格式中的 pdf 后缀
                    "reference_page": clean_pages,
                    "source_type": "slidevqa",
                    "query_type": "slidevqa",
                }
            }
            all_examples.append(example)

    # 3. 输出 JSON
    final_data = {"examples": all_examples}
    with open(json_output_name, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"\n处理完成！共生成 {len(all_examples)} 条数据。")

# 使用方法：
process_dataset('/mnt/cfs_algo_bj/models/experiments/shenyucheng/SlideVQA-SFT2/data', '/mnt/cfs_algo_bj/models/experiments/shenyucheng/SlideVQA-SFT2/img', '/mnt/cfs_algo_bj/models/experiments/shenyucheng/SlideVQA-SFT2/slidevqa_SFT2.json')