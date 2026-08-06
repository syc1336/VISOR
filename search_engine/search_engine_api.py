from fastapi import FastAPI, Query
import uvicorn
from typing import List, Dict, Any  # 导入类型提示，让代码更清晰
from search_engine import SearchEngine
from tqdm import tqdm
import os

# dataset_dir = './search_engine/corpus'
dataset_dir = '/mnt/cfs_algo_bj/workspace/shenyucheng/vidoseek'

# 创建 FastAPI 应用实例
app = FastAPI(
    title="Hybrid Search Engine API",
    description="Provides search functionality using a pre-initialized HybridSearchEngine.",
    version="1.0.0",
)

# 全局变量，用于存储 HybridSearchEngine 实例
search_engine=None
# 在应用启动时初始化 HybridSearchEngine
@app.on_event("startup")
async def startup_event():
    global search_engine
    print("Initializing SearchEngine...")
    search_engine = SearchEngine(dataset_dir, embed_model_name='/mnt/cfs_algo_bj/workspace/shenyucheng/vidore_colqwen2-v1.0')


# 定义搜索 API 端点
@app.get(
    "/search",
    summary="Perform a search query.",
    description="Executes a search using the initialized SearchEngine and returns the results.",
    response_model=List[List[Dict[str, Any]]]  # 定义响应模型，提高 API 文档的清晰度
)
async def search(queries: List[str] = Query(...)):
    """
    执行搜索操作。

    Args:
        query: 搜索查询字符串。

    Returns:
        搜索结果列表。
    """
    results_batch = search_engine.batch_search(queries)
    results_batch = [[dict(idx=idx,image_file=os.path.join(f'{dataset_dir}/img',file)) for idx,file in enumerate(query_results)] for query_results in results_batch]
    return results_batch

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
