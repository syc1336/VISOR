# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import json
import requests
import math
import numpy as np
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor


DEFAULT_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- the query
- a generated answer
- a reference answer

Your task is to evaluate the correctness of the generated answer.

## Query
{query}

## Reference Answer
{reference_answer}

## Generated Answer
{generated_answer}

Your response should be formatted as following:
<judge>True or False</judge>

If the generated answer is correct, please set "judge" to True. Otherwise, please set "judge" to False.

Please note that the generated answer may contain additional information beyond the reference answer.
"""

NO_INFO_SYSTEM_TEMPLATE = """You are an expert evaluation system for a question answering chatbot.

The chatbot was unable to retrieve all relevant information needed to answer the question, \
and you need to judge whether the chatbot appropriately acknowledged this limitation.

## Query
{query}

## Generated Answer
{generated_answer}

Your task: Judge whether the generated answer explicitly expresses that it cannot answer \
due to insufficient information, missing evidence, or uncertainty — rather than attempting \
to guess or fabricate an answer.

Examples of responses that SHOULD be judged True:
- "I don't have enough information to answer this question."
- "Based on the retrieved images, I cannot determine the answer."
- "The relevant page was not found, so I cannot provide a reliable answer."

Examples of responses that SHOULD be judged False:
- Any specific answer (e.g., a number, name, or description) without acknowledging uncertainty.
- Vague answers that don't explicitly admit insufficient information.

Your response should be formatted as following:
<judge>True or False</judge>

Set "judge" to True if the answer appropriately admits insufficient information. \
Otherwise set "judge" to False.
"""

def parse_judge_response(response):
    pattern = r'<judge>(.*?)</judge>'
    match = re.search(pattern, response, re.DOTALL)
    if match:
        try:
            judge_str = match.group(0)
            if 'true' in judge_str or 'True' in judge_str:
                return True
            elif 'false' in judge_str or 'False' in judge_str:
                return False
            else:
                return None
        except Exception as e:
            return None
    else:
        return None

    
def dcg(relevance_scores):
    """
    计算折扣累积增益（DCG）
    :param relevance_scores: 一个列表，表示每个文档的相关性分数
    :return: DCG 值
    """
    dcg_value = 0.0
    for i, relevance in enumerate(relevance_scores, start=1):
        dcg_value += (2 ** relevance - 1) / np.log2(i + 1)
    return dcg_value

def ndcg(sorted_docs, golden_answer_list):
    """
    计算归一化折扣累积增益（NDCG）
    :param sorted_docs: 一个列表，表示已经排好序的文档
    :param golden_answer_list: 一个列表，表示所有相关文档（golden answers）
    :return: NDCG 值
    """
    # 将文档映射为相关性分数（在 golden_answer_list 中的文档为 1，否则为 0）
    relevance_scores = [1 if doc in golden_answer_list else 0 for doc in sorted_docs]
    
    # 计算 DCG
    dcg_value = dcg(relevance_scores)
    
    # 计算 IDCG（理想情况下的 DCG，所有相关文档都排在前面）
    ideal_relevance_scores = [1] * len(golden_answer_list) + [0] * (len(sorted_docs) - len(golden_answer_list))
    idcg_value = dcg(ideal_relevance_scores)
    
    # 防止分母为零
    if idcg_value == 0:
        return 0.0
    
    # 计算 NDCG
    ndcg_value = dcg_value / idcg_value
    return ndcg_value

def get_answer_from_predict_str(text):
    end_tag = '</answer>'
    start_tag = '<answer>'

    end_pos = text.rfind(end_tag)
    if end_pos == -1:
        return None  # 如果没有找到</answer>，返回None

    start_pos = text.rfind(start_tag, 0, end_pos)
    if start_pos == -1:
        return None  # 如果没有找到<answer>，返回None

    start_pos += len(start_tag)  # 跳过<answer>标签
    return text[start_pos:end_pos]


def check_retrieval_completeness(response_str, retrievaled_images, reference_page_list, file_name):
    """
    判断检索图片是否覆盖所有参考页，以及在第几轮首次检索完全。

    原理：
      - 从 response_str 中统计 <search> 标签数量得到总轮次
      - retrievaled_images 是按检索顺序去重追加的扁平列表，每轮固定返回 1 张新图
        因此第 i 张图（0-indexed）对应第 i+1 轮 search
      - crop 图（basename 不符合参考页格式）不计入轮次映射，直接跳过

    Args:
        response_str (str): 模型完整输出的文本，包含 <search>/<answer> 等标签
        retrievaled_images (list[str]): 检索到的图片路径列表（按时间顺序、去重）
        reference_page_list (list[int]): 参考页码列表，如 [3, 7]
        file_name (str): PDF 文件名，如 'abc.pdf'

    Returns:
        tuple:
            is_complete (bool): 是否检索到了所有参考页
            complete_turn (int): 首次检索全的轮次（从 1 开始）；未检索全则为 -1
    """
    file_stem = file_name.split(".pdf")[0]
    reference_set = {f'{file_stem}_{page}' for page in reference_page_list}

    # 将图片路径列表转为 basename（去掉扩展名，与 reference_set 格式一致）
    img_basenames = [os.path.splitext(os.path.basename(img.rstrip('/')))[0] for img in retrievaled_images]

    # 快速判断整体是否完整
    is_complete = reference_set.issubset(set(img_basenames))

    # 统计总检索轮次
    num_turns = len(re.findall(r'<search>.*?</search>', response_str, re.DOTALL))

    if num_turns == 0 or len(img_basenames) == 0:
        return is_complete, (-1 if not is_complete else 1)

    # 每轮 search 固定返回 1 张新图，retrievaled_images 中第 i 张对应第 i+1 轮
    # crop 图（basename 不含 _95_ 页码格式）跳过，不占轮次
    collected = set()
    search_turn = 0  # 当前已消耗的 search 轮次
    for basename in img_basenames:
        is_page_img = re.search(r'_95_\d+$', basename) is not None
        if is_page_img:
            search_turn += 1
        collected.add(basename)
        if reference_set.issubset(collected):
            return True, search_turn

    return is_complete, -1


class RMManager:
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, eval_mode=False, rm_workers_num=3, rm_url="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", rm_key="EMPTY", rm_model_name="qwen-max-latest") -> None:
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.eval_mode = eval_mode
        self.rm_workers_num = rm_workers_num
        self.rm_url = rm_url
        self.rm_key = rm_key
        self.rm_model_name = rm_model_name
    
    def rm_score(self,data_eval_item):
        pay_load = {
            "model": self.rm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": DEFAULT_SYSTEM_TEMPLATE \
                        .replace("{query}", data_eval_item["query"]) \
                        .replace("{reference_answer}", data_eval_item["reference_answer"]) \
                        .replace("{generated_answer}", data_eval_item["generated_answer"])
                }
            ]
        }
        max_retry = 10
        for attempt in range(max_retry):
            try:
                response = requests.post(
                    self.rm_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.rm_key}",
                    },
                    json=pay_load
                )
                response.raise_for_status()
                result = response.json()
                judge_str = parse_judge_response(result['choices'][0]['message']['content'])
                if judge_str is not None:
                    if judge_str:
                        return 1.0
                    else:
                        return 0.0
            except Exception as e:
                pass
            time.sleep(32)
        return 0.0

    def no_info_score(self, data_eval_item):
        """判断检索不全时，模型是否正确表达了信息不足/无法回答。"""
        pay_load = {
            "model": self.rm_model_name,
            "messages": [
                {
                    "role": "user",
                    "content": NO_INFO_SYSTEM_TEMPLATE \
                        .replace("{query}", data_eval_item["query"]) \
                        .replace("{generated_answer}", data_eval_item["generated_answer"])
                }
            ]
        }
        max_retry = 10
        for attempt in range(max_retry):
            try:
                response = requests.post(
                    self.rm_url,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.rm_key}",
                    },
                    json=pay_load
                )
                response.raise_for_status()
                result = response.json()
                judge_str = parse_judge_response(result['choices'][0]['message']['content'])
                if judge_str is not None:
                    return 0.2 if judge_str else 0.0
            except Exception as e:
                pass
            time.sleep(32)
        return 0.0

    
    # 格式验证
    # def verify(self, data):
    #     scores = []
    #     for i in range(len(data)):
    #         data_item = data[i]  # DataProtoItem

    #         prompt_ids = data_item.batch['prompts']

    #         prompt_length = prompt_ids.shape[-1]

    #         valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
    #         valid_prompt_ids = prompt_ids[-valid_prompt_length:]

    #         response_ids = data_item.batch['responses']
    #         valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
    #         valid_response_ids = response_ids[:valid_response_length]

    #         # decode
    #         prompt_str = self.tokenizer.decode(valid_prompt_ids)
    #         response_str = self.tokenizer.decode(valid_response_ids)

    #         ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

    #         data_source = data_item.non_tensor_batch['data_source']

    #         extra_info = data_item.non_tensor_batch.get('extra_info', None)

    #         score = self.compute_score(
    #             data_source=data_source,
    #             solution_str=response_str,
    #             ground_truth=ground_truth,
    #             extra_info=extra_info,
    #         )
    #         scores.append(score)
    #     data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=prompt_ids.device)
    #     return scores

    def verify(self, data):
        data_eval = []
        eval_indices = []  # 记录需要调 API 的样本索引
        scores = [0.0] * len(data)  # 默认全部 0 分

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            generated_answer = get_answer_from_predict_str(self.tokenizer.decode(valid_response_ids))
            if generated_answer is None:
                # 提取不到 answer，直接 0 分，不浪费 API 调用
                continue

            data_eval.append(dict(
                query=extra_info['question'],
                generated_answer=generated_answer,
                reference_answer=data_item.non_tensor_batch['reward_model']['ground_truth']
            ))
            eval_indices.append(i)

        # 只对有有效 answer 的样本并行调用 RM
        if data_eval:
            if self.rm_workers_num > 1:
                with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
                    rm_scores = list(pool.map(self.rm_score, data_eval))
            else:
                rm_scores = [self.rm_score(item) for item in data_eval]

            for idx, score in zip(eval_indices, rm_scores):
                scores[idx] = score

        # 写回 batch，并作为验证指标
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32, device=data.batch['prompts'].device)

        return scores


    def _eval_call(self, data: DataProto):
        """评估模式：只用 rm_score 判断答案对错，返回 0/1，不涉及检索分。"""
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        data_to_rm_eval = []
        per_sample_info = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            response_ids = data_item.batch['responses']
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            # Prefer last_response_str (from reasoning_trace) to avoid truncation issues
            last_response_str = data_item.non_tensor_batch.get('last_response_str', None)
            text_to_parse = last_response_str if last_response_str else response_str
            generated_answer = get_answer_from_predict_str(text_to_parse) or ''

            per_sample_info.append(dict(
                valid_response_length=valid_response_length,
                rm_eval_idx=None,
            ))

            if generated_answer and extra_info is not None:
                eval_item = dict(
                    query=extra_info['question'],
                    reference_answer=ground_truth,
                    generated_answer=generated_answer,
                )
                data_to_rm_eval.append((i, eval_item))
                per_sample_info[-1]['rm_eval_idx'] = len(data_to_rm_eval) - 1

        def _run_parallel(fn, items):
            eval_items = [item for _, item in items]
            if not eval_items:
                return []
            if self.rm_workers_num > 1:
                with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
                    return list(pool.map(fn, eval_items))
            return [fn(item) for item in eval_items]

        rm_results = _run_parallel(self.rm_score, data_to_rm_eval)

        for i, info in enumerate(per_sample_info):
            rm_idx = info['rm_eval_idx']
            score = rm_results[rm_idx] if rm_idx is not None else 0.0
            reward_tensor[i, info['valid_response_length'] - 1] = score

        return reward_tensor

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        if self.eval_mode:
            return self._eval_call(data)

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        # ── Step 1: 收集需要调用 RM API 的样本（仅检索完整的样本才需要 model_eval）──
        # 格式得分已能从 response 直接判断，无需 API；
        # 只有检索完整时答案分才用 RM API 评估。
        data_to_rm_eval = []    # (sample_index, eval_item) 检索完整样本，走 rm_score
        data_to_no_info_eval = []  # (sample_index, eval_item) 检索不全样本，走 no_info_score
        per_sample_info = []    # 每个样本预计算的中间信息

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get('extra_info', None)

            # ── 格式分：compute_score > 0 即为格式正确 ──
            format_ok = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            ) > 0.0
            format_score = 0.0 if format_ok else -1.0
            generated_answer = get_answer_from_predict_str(response_str) or ''

            # ── 检索完整性 ──
            file_name = data_item.non_tensor_batch['file_name']
            reference_page = data_item.non_tensor_batch['reference_page'].tolist()
            retrievaled_images = list(data_item.non_tensor_batch['retrievaled_images'])
            is_complete, complete_turn = check_retrieval_completeness(
                response_str, retrievaled_images, reference_page, file_name
            )

            # 总检索轮次（response 中 <search> 标签数量）
            total_turns = len(re.findall(r'<search>.*?</search>', response_str, re.DOTALL))

            per_sample_info.append(dict(
                prompt_str=prompt_str,
                response_str=response_str,
                ground_truth=ground_truth,
                data_source=data_source,
                extra_info=extra_info,
                valid_response_length=valid_response_length,
                format_score=format_score,
                is_complete=is_complete,
                complete_turn=complete_turn,
                total_turns=total_turns,
                rm_eval_idx=None,       # rm_score 结果索引
                no_info_eval_idx=None,  # no_info_score 结果索引
            ))

            eval_item = dict(
                query=extra_info['question'],
                generated_answer=generated_answer,
            )
            if is_complete and format_ok:
                # 检索完整：用 rm_score 判断答案正确性
                eval_item['reference_answer'] = ground_truth
                data_to_rm_eval.append((i, eval_item))
                per_sample_info[-1]['rm_eval_idx'] = len(data_to_rm_eval) - 1
            elif not is_complete and format_ok:
                # 检索不全：用 no_info_score 判断是否承认信息不足
                data_to_no_info_eval.append((i, eval_item))
                per_sample_info[-1]['no_info_eval_idx'] = len(data_to_no_info_eval) - 1

        # ── Step 2: 并行调用两类 RM API ──
        rm_results = []
        no_info_results = []

        def _run_parallel(fn, items):
            eval_items = [item for _, item in items]
            if not eval_items:
                return []
            if self.rm_workers_num > 1:
                with ThreadPoolExecutor(max_workers=self.rm_workers_num) as pool:
                    return list(pool.map(fn, eval_items))
            else:
                return [fn(item) for item in eval_items]

        rm_results = _run_parallel(self.rm_score, data_to_rm_eval)
        no_info_results = _run_parallel(self.no_info_score, data_to_no_info_eval)

        # ── Step 3: 组合三项分数 ──
        for i, info in enumerate(per_sample_info):
            format_score = info['format_score']
            is_complete = info['is_complete']
            complete_turn = info['complete_turn']
            total_turns = info['total_turns']
            valid_response_length = info['valid_response_length']

            # 格式错误直接 -1，不计算后续分项
            if format_score < 0:
                score = -1.0
                retrieval_score = answer_score = 0.0
            else:
                # ── 检索分 ──
                if not is_complete:
                    retrieval_score = -1.0
                elif complete_turn == -1:
                    # 检索全了但无法定位到具体轮次（图片未出现在 response_str 中），保守给 0 分
                    retrieval_score = 0.0
                else:
                    # extra_turns: 搜全之后又多搜了几轮
                    extra_turns = total_turns - complete_turn
                    if extra_turns == 0:
                        retrieval_score = -0.5   # 搜全后直接答，没做 verification
                    elif extra_turns == 1:
                        retrieval_score = 0.0    # 刚好 1 轮 verification，最优
                    else:
                        # extra_turns ≥ 2：每多一轮额外加 0.1 惩罚，从 -0.2 起
                        retrieval_score = -0.1 * extra_turns

                # ── 答案分 ──
                if is_complete:
                    rm_idx = info['rm_eval_idx']
                    answer_score = rm_results[rm_idx] if rm_idx is not None else 0.0
                else:
                    no_info_idx = info['no_info_eval_idx']
                    answer_score = no_info_results[no_info_idx] if no_info_idx is not None else 0.0

                score = retrieval_score + answer_score
            reward_tensor[i, valid_response_length - 1] = score

            data_source = info['data_source']
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", info['prompt_str'])
                print("[response]", info['response_str'])
                print("[ground_truth]", info['ground_truth'])
                print(f"[score] {score:.2f}  (format={format_score}, retrieval={retrieval_score}, answer={answer_score})")
                print(f"  is_complete={is_complete}, complete_turn={complete_turn}/{total_turns}")

        return reward_tensor