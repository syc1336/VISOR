import json
from datasets import Dataset
import os
import datasets
import argparse
from tqdm import tqdm



# USER_PROMPT = '''Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}'''

USER_PROMPT = '''You are a visual reasoning agent. You will search for images to answer the user's question.

        Rules:
        1. Every response must start with <think> </think> where you reason about what you see and what to do next.
        
        2. After thinking, output exactly one action:
           - <search>query</search> to retrieve images. Each search returns one new image; if you repeat a query, you will get a different image from the same document. Use the original question as your query unless you have a specific reason to change it.
           - <bbox>[x1,y1,x2,y2]</bbox> to zoom into an unclear region (normalized to 0-1000, only on full images).
           - <answer>your answer</answer> once you are ready to give your final answer.
        
        3. Before answering, you must do one final search using the original question to verify your answer. After receiving the new image, give your <answer> immediately — unless the new image provides a directly conflicting answer to the question, in which case search once more and then give your <answer> immediately regardless.

        4. When given an image, analyze it fully in <think> </think> and extract every potentially useful piece of information — your thoughts will be recorded into a COLLECTED EVIDENCE table for later reference, so be as thorough as possible. If the image contains no relevant information, explicitly state that (e.g. "This image does not contain information related to the question.") to avoid polluting the evidence. Only propose <bbox> if critical details are visually unclear.

        A "COLLECTED EVIDENCE" section may appear summarizing findings from previous steps — use it alongside new images to guide your reasoning.


        Good example:

        <think>This image shows the Nordic Ecolabelling Portal login page. The page clearly states that Microsoft Edge or Google Chrome is recommended as the web browser. I have enough information to answer. I will do one final verification search.</think><search>Apply for Nordic Swan Ecolabel license, what is recommended as a web browser?</search>

        <think>This image shows a different section of the portal with no information about browser recommendations. It does not contradict my answer. My answer stands.</think><answer>Microsoft Edge or Google Chrome</answer>

        The user's question is:
        {question}
        '''

# all_examples = [example for example in all_examples if example['query'] not in sft_questions]
def convert_dataset(USER_PROMPT,file_list,file_source_list,output_name,limit=None):
    all_examples = []
    for file_name, source_type in zip(file_list, file_source_list):
        with open(file_name, "r") as f:
            file_data = json.load(f)
            data_list = file_data["examples"]
            for item in data_list:
                item['source'] = source_type
            all_examples.extend(data_list)

    # Limit to first N examples if specified
    if limit is not None and limit > 0:
        all_examples = all_examples[:limit]
        print(f"Limited dataset to first {limit} examples for debugging")

    for example in all_examples:
        if example['source'] == 'vidoseek':
            example['reason_type'] = example['meta_info']['query_type']
            example['content_type'] = example['meta_info']['source_type']
        elif example['source'] == 'slidevqa_test':
            query_type = example['meta_info']['query_type']
            if 'Multi-Hop' in query_type:
                example['reason_type'] = 'MultiHop'
            elif 'Single-Hop' in query_type:
                example['reason_type'] = 'SingleHop'
            if 'Non-Span' in query_type:
                example['content_type'] = 'NonSpan'
            elif 'Single-Span' in query_type:
                example['content_type'] = 'SingleSpan'
            elif 'Multi-Span' in query_type:
                example['content_type'] = 'MultiSpan'
        elif example['source'] == 'mmlongdoc':
            example['content_type'] = '####'.join(example['meta_info']['source_type'])
            example['reason_type'] = example['meta_info']['query_type']
        else:
            example['content_type'] = 'Nan'
            example['reason_type'] = 'Nan'

    dataset = Dataset.from_dict({
        "id": [str(example["uid"]) for example in all_examples],
        "problem": [example["query"] for example in all_examples],
        "prompt": [USER_PROMPT.replace('{question}',example["query"]) for example in all_examples],
        "answer": [example["reference_answer"] for example in all_examples],
        "file_name": [example["meta_info"]["file_name"] for example in all_examples],
        "reference_page": [example["meta_info"]["reference_page"] for example in all_examples],
        "data_source_type": [example["source"] for example in all_examples],
        "query_content_type": [example.get("content_type", "") for example in all_examples],
        "query_reason_type": [example.get("reason_type", "") for example in all_examples]
    })

    def make_map_fn_test(split):
        def process_fn(example, idx):
            prompt = example.pop('prompt')
            answer = example.pop('answer')
            problem = example.pop('problem')
            data_source = example.pop('data_source_type')
            reference_page = example.pop('reference_page')
            file_name = example.pop('file_name')
            query_content_type = example.pop('query_content_type')
            query_reason_type = example.pop('query_reason_type')

            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },

                # ✅ 放到顶层
                "file_name": file_name,
                "reference_page": reference_page,

                "extra_info": {
                    'split': split,
                    'index': idx,
                    'answer': answer,
                    "question": problem,
                    "content_type": query_content_type,
                    "reason_type": query_reason_type
                }
            }
            return data
        return process_fn

    test_dataset = dataset.map(function=make_map_fn_test('test'), with_indices=True, num_proc=8)

    test_dataset.to_parquet(f'./data/{output_name}.parquet')


if __name__ == '__main__':
    # slidevqa train
    convert_dataset(
        USER_PROMPT,
        ['/mnt/cfs_algo_bj/workspace/shenyucheng/SlideVQA-RL/slidevqa_RL.json'],
        ['slidevqa_train'],
        'slidevqa-train',
    )

    # vidoseek test
    convert_dataset(
        USER_PROMPT,
        ['/mnt/cfs_algo_bj/workspace/shenyucheng/vidoseek/vidoseek.json'],
        ['vidoseek'],
        'vidoseek_test',
    )

    # slidevqa test
    convert_dataset(
        USER_PROMPT,
        ['/mnt/cfs_algo_bj/models/experiments/shenyucheng/SlideVQA-test/slidevqa_test.json'],
        ['slidevqa_test'],
        'slidevqa_test',
    )

    # mmlongbench doc test
    convert_dataset(
        USER_PROMPT,
        ['/mnt/cfs_algo_bj/models/experiments/shenyucheng/MMLongBench-Doc/mmlongbench_doc_test.json'],
        ['mmlongdoc'],
        'mmlongdoc_test',
    )