import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Mapping, Any, Dict
import json
from tqdm import tqdm
import torch

from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo, ImageNode
from vl_embedding import VL_Embedding


def nodefile2node(input_file):
    nodes = []
    for doc in json.load(open(input_file, 'r')):
        if doc['class_name'] == 'TextNode' and doc['text'] != '':
            nodes.append(TextNode.from_dict(doc))
        elif doc['class_name'] == 'ImageNode':
            nodes.append(ImageNode.from_dict(doc))
        else:
            continue
    return nodes

class SearchEngine:
    def __init__(self, dataset_dir='search_engine/corpus', node_dir_prefix='colqwen_ingestion',embed_model_name='vidore/colqwen2-v1.0'): # "vidore/colqwen2-v0.1"

        self.workers = 1

        self.dataset_dir = dataset_dir

        self.node_dir = os.path.join(self.dataset_dir, node_dir_prefix)
        self.vector_embed_model = VL_Embedding(model=embed_model_name, mode='image')
        self.query_engine = self.load_query_engine()

    def load_nodes(self):
        files = os.listdir(self.node_dir)
        parsed_files = []
        max_workers = 10
        if max_workers == 1:
            for file in tqdm(files):
                input_file = os.path.join(self.node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    continue
                nodes = nodefile2node(input_file)
                parsed_files.extend(nodes)
        else:
            def parse_file(file,node_dir):
                input_file = os.path.join(node_dir, file)
                suffix = input_file.split('.')[-1]
                if suffix != 'node':
                    return []
                return nodefile2node(input_file)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # results = list(tqdm(executor.map(parse_file, files, self.node_dir), total=len(files)))
                results = list(tqdm(executor.map(parse_file, files, [self.node_dir]*len(files)), total=len(files)))
            # 合并所有线程的结果
            for result in results:
                parsed_files.extend(result)
        return parsed_files
        
    def load_query_engine(self):
        print('Loading nodes...')
        self.nodes = self.load_nodes()
        self.embedding_img = [torch.tensor(node.embedding).view(-1, 128).bfloat16() for node in tqdm(self.nodes, desc="Creating Embeddings")]
        self.embedding_img = [tensor.to(self.vector_embed_model.embed_model.device) for tensor in tqdm(self.embedding_img, desc="Moving to Device")]
        self.image_nums = len(self.embedding_img)

    def load_node_postprocessors(self):
        return []
    def batch_search(self, queries: List[str]):
        batch_queries = self.vector_embed_model.processor.process_queries(queries).to(self.vector_embed_model.embed_model.device)
        with torch.no_grad():
            query_embeddings = self.vector_embed_model.embed_model(**batch_queries)
        scores = self.vector_embed_model.processor.score_multi_vector(query_embeddings, self.embedding_img, batch_size=256, device=self.vector_embed_model.embed_model.device)
        values, indices = torch.topk(scores, k=min(self.image_nums,10), dim=1)
        recall_results = [[self.nodes[idx].metadata['file_name'] for idx in row] for row in indices]
        return recall_results


if __name__ == '__main__':
    search_engine = SearchEngine(dataset_dir='/data1/syc/ViDoSeek',embed_model_name='/data1/syc/vidore_colqwen2-v1.0')
    print(search_engine.batch_search(['o','a']))
    

    