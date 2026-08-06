import torch
import re
import numpy as np
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
from transformers.image_processing_base import BatchFeature
from PIL import Image
from tqdm import tqdm
import json

def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 512 * 512):
    import math
    from io import BytesIO
    from PIL import Image

    if isinstance(image, dict):
        image = Image.open(BytesIO(image['bytes']))
    elif isinstance(image, str):
        image = Image.open(image)


    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image

@dataclass
class GenerationConfig:
    max_turns: int
    max_prompt_length: int
    num_gpus: int
    search_url: str = None
    max_model_len: int = 10240
    image_pad_id: int = 151655
    endoftext_id: int = 151643
    sliding_window_size: int = 2  # keep last N turns of user-assistant dialogue


class LLMGenerationManager:
    def __init__(
        self,
        processor,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
        enable_intent_injection: bool = True,
        intent_injection_template: str = None,
        case_image_dir: str = None,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        self.enable_intent_injection = enable_intent_injection
        self.intent_injection_template = "[{question}]"
        self.case_image_dir = case_image_dir  # Directory for saving cropped images during validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id
        ))

        # Evidence collection: {batch_idx: {image_path: [thoughts]}}
        self.evidence_ledger = {}
        # Counter for unique crop image filenames
        self._crop_counter = 0

    def _extract_and_store_think(self, responses_str: List[str], batch_indices: List[int]):
        """Extract <think>...</think> content and store as evidence."""
        for idx, (resp, batch_idx) in enumerate(zip(responses_str, batch_indices)):
            think_match = re.search(r'<think>(.*?)</think>', resp, re.DOTALL)
            if think_match:
                thought = think_match.group(1).strip()
                if len(thought) > 5:  # Filter out very short thinks
                    # Initialize if batch_idx not in ledger
                    if batch_idx not in self.evidence_ledger:
                        self.evidence_ledger[batch_idx] = {"Initial_Logic": []}

                    # Find current root image (last retrieved image for this batch)
                    if hasattr(self, 'retrievaled_images') and batch_idx < len(self.retrievaled_images):
                        if len(self.retrievaled_images[batch_idx]) > 0:
                            current_root = os.path.basename(self.retrievaled_images[batch_idx][-1])
                        else:
                            current_root = "Initial_Logic"
                    else:
                        current_root = "Initial_Logic"

                    # Store thought under current root
                    if current_root not in self.evidence_ledger[batch_idx]:
                        self.evidence_ledger[batch_idx][current_root] = []
                    self.evidence_ledger[batch_idx][current_root].append(thought)

    def _format_collected_evidence(self, batch_idx: int) -> str:
        """Format collected evidence for a specific batch item."""
        if batch_idx not in self.evidence_ledger or not self.evidence_ledger[batch_idx]:
            return "No evidence collected yet."

        ledger_sections = []
        for img, thoughts in self.evidence_ledger[batch_idx].items():
            thought_list = "\n".join([f"  - {t}" for t in thoughts])
            ledger_sections.append(f"【Image Source: {img}】\n{thought_list}")

        return "\n\n".join(ledger_sections)

    def _build_evidence_obs_ids(self, batch_size: int, active_mask: torch.Tensor) -> torch.Tensor:
        """Build evidence summary as a user message, tokenized for each batch item.
        Only active items get real evidence; inactive items get empty string."""
        evidence_strs = []
        for idx in range(batch_size):
            if active_mask[idx]:
                evidence_str = self._format_collected_evidence(idx)
                evidence_strs.append(
                    f'\n<|im_start|>user\n### COLLECTED EVIDENCE\n{evidence_str}\n<|im_end|>\n<|im_start|>assistant\n'
                )
            else:
                evidence_strs.append('')

        evidence_ids = self.tokenizer(
            evidence_strs,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']
        return evidence_ids

    def _rebuild_with_sliding_window(
        self,
        initial_input_ids: torch.Tensor,
        initial_non_tensor_batch: dict,
        initial_mmd: list,
        initial_mmi: list,
        turn_history: list,
        evidence_ids: torch.Tensor,
        window_size: int,
    ):
        """Rebuild rollings keeping only: initial_prompt + evidence + last window_size turns.

        turn_history: list of dicts, each with keys:
            'response_ids': (batch, resp_len) tensor
            'obs_ids': (batch, obs_len) tensor
            'obs_mmd': list of {'image': [...]} per batch item
            'obs_mmi': list of BatchFeature per batch item
        """
        batch_size = initial_input_ids.shape[0]
        recent_turns = turn_history[-window_size:] if len(turn_history) > window_size else turn_history

        # --- 1. Rebuild token IDs ---
        # Evidence goes right after the initial query, sliding window turns follow
        pieces = [initial_input_ids]
        pieces.append(evidence_ids)
        for turn in recent_turns:
            pieces.append(turn['response_ids'])
            pieces.append(turn['obs_ids'])

        new_input_ids = self.tensor_fn.concatenate_with_padding(pieces)
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        # --- 2. Rebuild multi-modal data (initial + windowed turns only) ---
        rebuilt_mmd = []
        rebuilt_mmi = np.empty(batch_size, dtype=object)
        for idx in range(batch_size):
            # Start with a copy of initial images
            images_list = list(initial_mmd[idx]['image'])
            mmi_item = BatchFeature(dict(initial_mmi[idx]))

            for turn in recent_turns:
                turn_mmd_item = turn['obs_mmd'][idx]
                turn_mmi_item = turn['obs_mmi'][idx]
                if len(turn_mmd_item['image']) > 0 and 'pixel_values' in turn_mmi_item:
                    images_list.extend(turn_mmd_item['image'])
                    if 'pixel_values' in mmi_item:
                        mmi_item['pixel_values'] = torch.cat(
                            (mmi_item['pixel_values'], turn_mmi_item['pixel_values']), dim=0
                        )
                        mmi_item['image_grid_thw'] = torch.cat(
                            (mmi_item['image_grid_thw'], turn_mmi_item['image_grid_thw']), dim=0
                        )
                    else:
                        mmi_item['pixel_values'] = turn_mmi_item['pixel_values']
                        mmi_item['image_grid_thw'] = turn_mmi_item['image_grid_thw']

            rebuilt_mmd.append({'image': images_list})
            rebuilt_mmi[idx] = mmi_item

        # --- 3. Assemble non_tensor_batch ---
        new_non_tensor_batch = {}
        for k, v in initial_non_tensor_batch.items():
            if k in ('multi_modal_data', 'multi_modal_inputs'):
                continue
            new_non_tensor_batch[k] = v.copy() if hasattr(v, 'copy') else v

        new_non_tensor_batch['multi_modal_data'] = np.array(rebuilt_mmd, dtype=object)
        new_non_tensor_batch['multi_modal_inputs'] = rebuilt_mmi

        rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:],
        }, new_non_tensor_batch)

        # After left-truncation, some initial images' tokens may have been cut.
        # Sync multi_modal_data/inputs so the image count matches the truncated input_ids.
        self._sync_multi_modal_after_truncation(rollings)

        return rollings

    def _sync_multi_modal_after_truncation(self, rollings):
        """After left-truncation of input_ids, remove leading images whose tokens
        were truncated so that multi_modal_data/inputs stay aligned with the
        actual image token groups in input_ids."""
        image_pad_id = self.config.image_pad_id
        for idx in range(rollings.batch['input_ids'].shape[0]):
            ids = rollings.batch['input_ids'][idx]
            mask = rollings.batch['attention_mask'][idx]
            # Count image groups (contiguous runs of image_pad_id) in valid tokens
            valid_ids = ids[mask == 1]
            num_groups = 0
            in_group = False
            for t in valid_ids.tolist():
                if t == image_pad_id:
                    if not in_group:
                        num_groups += 1
                        in_group = True
                else:
                    in_group = False

            mmd = rollings.non_tensor_batch['multi_modal_data'][idx]
            mmi = rollings.non_tensor_batch['multi_modal_inputs'][idx]
            num_images = len(mmd['image'])
            if num_images > num_groups:
                # Truncation removed leading images; drop them from multi_modal
                drop = num_images - num_groups
                mmd['image'] = mmd['image'][drop:]
                if 'pixel_values' in mmi and 'image_grid_thw' in mmi:
                    mmi['pixel_values'] = mmi['pixel_values'][drop:]
                    mmi['image_grid_thw'] = mmi['image_grid_thw'][drop:]
            elif num_images < num_groups:
                # More image token groups in input_ids than images in metadata.
                # Drop the excess trailing image token groups from input_ids to
                # keep them consistent with multi_modal metadata.
                print(f"WARNING _sync_multi_modal: idx={idx} num_images({num_images}) < num_groups({num_groups}), "
                      f"stripping excess image token groups from input_ids")
                ids = rollings.batch['input_ids'][idx]
                mask = rollings.batch['attention_mask'][idx]
                groups_to_keep = num_images
                groups_seen = 0
                in_group = False
                strip_from = None
                for pos in range(len(ids)):
                    if mask[pos] == 0:
                        continue
                    if ids[pos] == image_pad_id:
                        if not in_group:
                            groups_seen += 1
                            in_group = True
                            if groups_seen > groups_to_keep:
                                strip_from = pos
                                break
                    else:
                        in_group = False
                if strip_from is not None:
                    # Replace excess image-related tokens with pad
                    in_img_region = False
                    for pos in range(strip_from, len(ids)):
                        if mask[pos] == 0:
                            break
                        tok = ids[pos].item()
                        if tok == image_pad_id:
                            in_img_region = True
                            ids[pos] = self.config.endoftext_id
                            mask[pos] = 0
                        elif tok == 151652 or tok == 151653:
                            # vision_start / vision_end tokens
                            ids[pos] = self.config.endoftext_id
                            mask[pos] = 0
                        elif in_img_region:
                            in_img_region = False

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']
    
    def _postprocess_responses_first(self,batch):
        
        responses_str = self.tokenizer.batch_decode(batch.batch['input_ids'], skip_special_tokens=True)
        responses_str = ["<search>"+item.split('Question: ')[1].split(' \n\nassistant\n')[0]+"</search>" for item in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str
        

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        def extract_tags(text):
            # 定义正则表达式，匹配 <answer>...</answer>、<search>...</search> 和 <think>...</think>
            pattern = r"<(answer|search|think|bbox)>(.*?)</\1>"
            # 使用 findall 方法找到所有匹配的内容
            matches = re.findall(pattern, text, re.DOTALL)
            # 将匹配的内容重新组合成字符串
            result = "\n".join([f"<{tag}>{content}</{tag}>" for tag, content in matches])
            return result

        responses_str = [extract_tags(resp) + self.tokenizer.eos_token for resp in responses_str]

        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _is_verification_turn(self, response_str: str) -> bool:
        """判断上一轮 think 中是否表达了 verification 意图。"""
        think_match = re.search(r'<think>(.*?)</think>', response_str, re.DOTALL)
        if not think_match:
            return False
        think = think_match.group(1).lower()

        # Negative patterns that indicate failure to confirm, not verification intent
        negative_patterns = [
            r"cannot\s+(?:confirm|verify)",
            r"can'?t\s+(?:confirm|verify)",
            r"could\s*n'?t\s+(?:confirm|verify)",
            r"unable\s+to\s+(?:confirm|verify)",
            r"(?:does|do)\s+not\s+(?:confirm|verify)",
            r"didn'?t\s+(?:confirm|verify)",
            r"failed\s+to\s+(?:confirm|verify)",
            r"not\s+(?:confirmed|verified)",
            r"hard\s+to\s+(?:confirm|verify)",
            r"difficult\s+to\s+(?:confirm|verify)",
        ]
        for neg in negative_patterns:
            if re.search(neg, think):
                return False

        # Positive verification intent patterns
        verification_patterns = [
            r'verification',
            r'(?:final\s+)?search\s+to\s+(?:verify|confirm)',
            r'to\s+confirm\s+(?:this|the\s+answer|the\s+information|the\s+result|the\s+observation)',
            r'(?:the\s+)?answer\s+is\s+confirmed',
            r'this\s+confirms\s+(?:the|that|my)',
            r'(?:do|perform)\s+a\s+final\s+(?:search|verification)',
            r'verify\s+(?:this|the\s+answer|the\s+information|the\s+result)',
        ]
        return any(re.search(p, think) for p in verification_patterns)

    def _process_next_obs(self, next_obs: List, rollings, original_questions: List[str] = None, prev_responses_str: List[str] = None) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_str = []
        multi_modal_data = []
        multi_modal_inputs = []
        merge_length = self.processor.image_processor.merge_size**2
        for idx, obs_item in enumerate(next_obs):
            # Get intent injection prompt if enabled
            intent_prompt = ""
            if self.enable_intent_injection and original_questions and idx < len(original_questions):
                intent_prompt = f"{self.intent_injection_template.format(question=original_questions[idx])}"

            # invalid
            if isinstance(obs_item,str):
                next_obs_str.append(obs_item + intent_prompt)
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # invalid
            elif isinstance(obs_item, list) and not isinstance(obs_item[0],dict) and len(self.retrievaled_images[idx]) == 0:
                next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Please try again. \n<|im_end|>\n<|im_start|>assistant\n')
                multi_modal_data.append({'image': []})
                multi_modal_inputs.append(BatchFeature(dict()))
            # crop
            elif isinstance(obs_item,list) and not isinstance(obs_item[0],dict):
                try:
                    latest_image = rollings.non_tensor_batch['multi_modal_data'][idx]['image'][-1]
                    width, height = latest_image.size
                    raw_images_crop = Image.open(self.retrievaled_images[idx][-1])
                    raw_width, raw_height = raw_images_crop.size
                    if self.is_validation:
                        obs_item = [obs_item[0]-28, obs_item[1]-28, obs_item[2]+28, obs_item[3]+28]
                    crop_area = [int(raw_width * obs_item[0] / width), int(raw_height * obs_item[1] / height), int(raw_width * obs_item[2] / width), int(raw_height * obs_item[3] / height)]
                    crop_area = [max(0, crop_area[0]), max(0, crop_area[1]), min(raw_width, crop_area[2]), min(raw_height, crop_area[3])]
                    input_images_list = [raw_images_crop.crop((crop_area[0], crop_area[1], crop_area[2], crop_area[3]))]

                    # Save cropped image during validation
                    if self.is_validation and self.case_image_dir:
                        crop_save_path = os.path.join(self.case_image_dir, f"crop_{idx}_{self._crop_counter}.jpg")
                        input_images_list[0].save(crop_save_path)
                        self.crop_image_paths[idx].append(crop_save_path)
                        self._crop_counter += 1

                    raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                    multi_modal_data.append({'image': raw_images_list})
                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')
                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']
                    obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                    raw_obs_str = f"<|vision_start|>{self.processor.image_token}<|vision_end|>" * len(image_grid_thw)
                    obs_str = '\n<|im_start|>user\n' + obs_str + f'This is the cropped image, analyse it based on the question: {intent_prompt} and after <think> </think> you just can use <search> or <answer> this time. If the cropped image is incorrect, please pay attention to previous think.' + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)
                except Exception as e:
                    next_obs_str.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Please try again.' + '\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
            # ret image
            elif isinstance(obs_item,list) and isinstance(obs_item[0],dict):
                img_file_list = [item['image_file'] for item in obs_item]
                input_images_list = None
                for image_item in img_file_list:
                    if image_item not in self.retrievaled_images[idx]:
                        self.retrievaled_images[idx].append(image_item)
                        input_images_list = [image_item]
                        break

                if input_images_list is None:
                    # All images already retrieved — no new image to show
                    next_obs_str.append('\n<|im_start|>user\nNo more new images can be retrieved. Based on all the evidence you have collected so far, please provide your final answer inside <answer> and </answer> after <think> and </think>. For example, <answer> Beijing </answer>.\n<|im_end|>\n<|im_start|>assistant\n')
                    multi_modal_data.append({'image': []})
                    multi_modal_inputs.append(BatchFeature(dict()))
                else:
                    raw_images_list = [process_image(image, 512*28*28, 256*28*28) for image in input_images_list]

                    multi_modal_data.append({'image': raw_images_list})
                    image_inputs = self.processor.image_processor(raw_images_list, return_tensors='pt')

                    multi_modal_inputs.append(image_inputs)
                    image_grid_thw = image_inputs['image_grid_thw']

                    obs_str = ''.join([f"<|vision_start|>{self.processor.image_token * (image_grid_thw_item.prod() // merge_length)}<|vision_end|>" for image_grid_thw_item in image_grid_thw])
                    prev_resp = prev_responses_str[idx] if prev_responses_str and idx < len(prev_responses_str) else ""
                    if self._is_verification_turn(prev_resp):
                        obs_hint = (
                            'This may be a verification search. Based on all the evidence collected so far, '
                            'if the new image confirms the answer, provide it directly inside <answer> and </answer>. '
                            'Only continue searching if you find a contradiction or missing information.'
                        )
                    else:
                        obs_hint = f'Image loaded, analyze any possible useful information for the question: {intent_prompt} in your think, then continue your action after <think> </think>.'
                    obs_str = '\n<|im_start|>user\n' + obs_str + obs_hint + '<|im_end|>\n<|im_start|>assistant\n'
                    next_obs_str.append(obs_str)
            else:
                raise ValueError('invalid observation')

        next_obs_ids = self.tokenizer(
            next_obs_str,
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,
        )['input_ids']

        return next_obs_ids, next_obs_str, multi_modal_data, multi_modal_inputs
    
    def _concat_multi_modal_data(self, rollings, next_obs_multi_modal_data:list, next_obs_multi_modal_inputs:list):
        if not 'multi_modal_inputs' in rollings.non_tensor_batch.keys():

            rollings.non_tensor_batch['multi_modal_inputs'] = np.empty(len(next_obs_multi_modal_data), dtype=object)
            for idx, item in enumerate(next_obs_multi_modal_inputs):
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = item

            rollings.non_tensor_batch['multi_modal_data'] = np.array(next_obs_multi_modal_data, dtype=object)

        else:

            for idx, multi_modal_data_item in enumerate(next_obs_multi_modal_data):
                if len(multi_modal_data_item['image']) > 0 and 'pixel_values' in next_obs_multi_modal_inputs[idx]:
                    # data
                    rollings.non_tensor_batch['multi_modal_data'][idx]['image'].extend(multi_modal_data_item['image'])
                    if 'pixel_values' in rollings.non_tensor_batch['multi_modal_inputs'][idx]:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'], next_obs_multi_modal_inputs[idx]['pixel_values']),dim=0)
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = torch.cat((rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'], next_obs_multi_modal_inputs[idx]['image_grid_thw']),dim=0)
                    else:
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['pixel_values'] = next_obs_multi_modal_inputs[idx]['pixel_values']
                        rollings.non_tensor_batch['multi_modal_inputs'][idx]['image_grid_thw'] = next_obs_multi_modal_inputs[idx]['image_grid_thw']

        return rollings
        

    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        if next_obs_ids.shape[1] != 0:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses
            ])
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        }, rollings.non_tensor_batch)

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None and next_obs_ids.shape[1] != 0:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
                next_obs_ids
            ], pad_to_left=False)
        else:
            responses = self.tensor_fn.concatenate_with_padding([
                right_side['responses'],
                cur_responses,
            ], pad_to_left=False)
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {'responses': responses[:, :max_len]}


    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        padded_non_tensor_batch = {}

        padded_ids = self.tokenizer(
            ['<|im_start|>user\nHi, who are u?<|im_end|>\n<|im_start|>assistant\n'], 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']
        padded_ids = padded_ids[0]

        pad_input_ids = torch.full_like(active_batch.batch['input_ids'][0], self.config.endoftext_id, dtype=torch.int64) #151643
        pad_input_ids[:len(padded_ids)] = padded_ids
        pad_attention_mask = self.tensor_fn.create_attention_mask(pad_input_ids)
        pad_input_ids = pad_input_ids.unsqueeze(0)
        pad_attention_mask = pad_attention_mask.unsqueeze(0)
        pad_position_ids = self.tensor_fn.create_position_ids(pad_attention_mask)
        
        padded_batch['attention_mask'] = torch.cat([active_batch.batch['attention_mask'], pad_attention_mask.repeat(padding_size, *[1] * (len(active_batch.batch['attention_mask'].shape) - 1))], dim=0)
        padded_batch['input_ids'] = torch.cat([active_batch.batch['input_ids'], pad_input_ids.repeat(padding_size, *[1] * (len(active_batch.batch['input_ids'].shape) - 1))], dim=0)
        padded_batch['position_ids'] = torch.cat([active_batch.batch['position_ids'], pad_position_ids.repeat(padding_size, *[1] * (len(active_batch.batch['position_ids'].shape) - 1))], dim=0)
        

        for k, v in active_batch.non_tensor_batch.items():
            pad_non_tensor_item = np.empty(padding_size, dtype=object)
            if k == 'raw_prompt_ids':
                list_ids = padded_ids.tolist()
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = list_ids
            elif k == 'multi_modal_inputs':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {}
            elif k == 'multi_modal_data':
                for idx in range(padding_size):
                    pad_non_tensor_item[idx] = {'image': []}
            padded_non_tensor_batch[k] = np.concatenate([v, pad_non_tensor_item])
                
        padded_active_batch = DataProto.from_dict(padded_batch, padded_non_tensor_batch)
        
        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)
        
        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def _raw_prompt_ids(self, rollings):
        new_raw_prompt_ids = []
        rollings.batch['input_ids'] = rollings.batch['input_ids'].long()
        raw_next_obs_ids = [ids[mask == 1].tolist() for ids, mask in zip(np.array(rollings.batch['input_ids']),  np.array(rollings.batch['attention_mask']))]
        def replace_consecutive_elements(arr, target):
            result = []
            i = 0
            while i < len(arr):
                if arr[i] == target:
                    result.append(target)
                    while i + 1 < len(arr) and arr[i + 1] == target:
                        i += 1
                else:
                    result.append(arr[i])
                i += 1
            return result
        raw_next_obs_ids = [replace_consecutive_elements(row,self.config.image_pad_id) for row in raw_next_obs_ids] #151655
        raw_next_obs_ids = np.array(raw_next_obs_ids, dtype=object)
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_next_obs_ids
        return rollings

    def deactivate_batch(self, active_mask,rollings):
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']
        max_model_len = self.config.max_model_len
        curr_active_mask = torch.tensor([len(raw_prompt_ids_item) < max_model_len for raw_prompt_ids_item in raw_prompt_ids], dtype=torch.bool)
        active_mask = active_mask * curr_active_mask
        return active_mask

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""

        # original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_left_side = {'input_ids': initial_input_ids}
        original_right_side = {'responses': initial_input_ids[:, []]}

        batch_size = gen_batch.batch['input_ids'].shape[0]
        active_mask = torch.ones(batch_size, dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        raw_prompt_ids = rollings.non_tensor_batch['raw_prompt_ids']

        # Extract original questions if available for intent injection
        original_questions = None
        if self.enable_intent_injection and 'original_question' in rollings.non_tensor_batch:
            original_questions = rollings.non_tensor_batch['original_question'].tolist()

        self.retrievaled_images = [[] for _ in range(batch_size)]
        self.search_counts = [0] * batch_size

        # Initialize evidence ledger for all batch items
        for batch_idx in range(batch_size):
            self.evidence_ledger[batch_idx] = {"Initial_Logic": []}

        # Initialize reasoning traces for validation case recording
        self.reasoning_traces = {i: [] for i in range(batch_size)}
        self.crop_image_paths = {i: [] for i in range(batch_size)}

        # ========== Save initial state for sliding window rebuild ==========
        saved_initial_input_ids = gen_batch.batch['input_ids'].clone()
        saved_initial_non_tensor_batch = {}
        for k, v in gen_batch.non_tensor_batch.items():
            saved_initial_non_tensor_batch[k] = v.copy() if hasattr(v, 'copy') else v

        # Save initial multi-modal data (deep copy images list)
        if 'multi_modal_data' in gen_batch.non_tensor_batch:
            saved_initial_mmd = [{'image': list(item['image'])} for item in gen_batch.non_tensor_batch['multi_modal_data']]
            saved_initial_mmi = []
            for item in gen_batch.non_tensor_batch['multi_modal_inputs']:
                saved_initial_mmi.append(BatchFeature({k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in dict(item).items()}))
        else:
            saved_initial_mmd = [{'image': []} for _ in range(batch_size)]
            saved_initial_mmi = [BatchFeature(dict()) for _ in range(batch_size)]

        # Turn history: stores per-turn response/obs tensors & multi-modal data
        turn_history = []
        window_size = self.config.sliding_window_size

        # Main generation loop
        step = -1
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)
            if not active_mask.sum():
                break

            if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                rollings_active = DataProto.from_dict(
                    tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                    non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                )
            else:
                rollings_active = DataProto.from_dict({
                    k: v[active_mask] for k, v in rollings.batch.items()
                })

            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

            # Extract and store think content for active batch items
            active_indices = torch.where(active_mask)[0].tolist()
            self._extract_and_store_think(responses_str, active_indices)

            print(responses_str[0])

            # Record assistant responses for validation case recording
            if self.is_validation:
                for j, batch_idx in enumerate(active_indices):
                    resp_content = responses_str[j]
                    if resp_content.endswith(self.tokenizer.eos_token):
                        resp_content = resp_content[:-len(self.tokenizer.eos_token)]
                    self.reasoning_traces[batch_idx].append({
                        "step": step,
                        "role": "assistant",
                        "content": resp_content.strip()
                    })

            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            # Save retrieved image counts before execute_predictions for observation tracking
            if self.is_validation:
                prev_ret_images_count = {idx: len(self.retrievaled_images[idx]) for idx in range(len(self.retrievaled_images))}
            # Execute in environment and process observations
            next_obs, dones = self.execute_predictions(responses_str, self.tokenizer.pad_token, active_mask)

            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            next_obs_ids, next_obs_str, next_obs_multi_modal_data, next_obs_multi_modal_inputs = self._process_next_obs(next_obs, rollings, original_questions, prev_responses_str=responses_str)

            # Record observations for validation case recording
            if self.is_validation:
                for idx in range(len(next_obs)):
                    if not active_mask[idx]:
                        continue
                    obs_item = next_obs[idx]
                    if isinstance(obs_item, list) and isinstance(obs_item[0], dict):
                        # Search result - new image retrieved
                        new_images = self.retrievaled_images[idx][prev_ret_images_count[idx]:]
                        self.reasoning_traces[idx].append({
                            "step": step,
                            "role": "observation",
                            "type": "search",
                            "retrieved_images": list(new_images)
                        })
                    elif isinstance(obs_item, list) and not isinstance(obs_item[0], dict):
                        # Bbox crop
                        crop_path = self.crop_image_paths[idx][-1] if self.crop_image_paths[idx] else ''
                        self.reasoning_traces[idx].append({
                            "step": step,
                            "role": "observation",
                            "type": "bbox_crop",
                            "bbox": obs_item,
                            "crop_image_path": crop_path
                        })
                    elif isinstance(obs_item, str) and obs_item == '':
                        # Answer action - done
                        pass
                    else:
                        # Invalid action or other observation
                        self.reasoning_traces[idx].append({
                            "step": step,
                            "role": "observation",
                            "type": "system_message",
                            "content": obs_item[:200] if isinstance(obs_item, str) else str(obs_item)[:200]
                        })

            # Record this turn
            turn_history.append({
                'response_ids': responses_ids,
                'obs_ids': next_obs_ids,
                'obs_mmd': next_obs_multi_modal_data,
                'obs_mmi': next_obs_multi_modal_inputs,
            })

            # ========== Sliding window rebuild ==========
            # Build evidence ids (only for active items, appended at the end)
            evidence_ids = self._build_evidence_obs_ids(batch_size, active_mask)

            rollings = self._rebuild_with_sliding_window(
                saved_initial_input_ids,
                saved_initial_non_tensor_batch,
                saved_initial_mmd,
                saved_initial_mmi,
                turn_history,
                evidence_ids,
                window_size,
            )

            # Keep full trajectory in original_right_side for reward computation
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )


        # final LLM rollout
        if active_mask.sum():
            # Add final prompt for active samples before the last generation
            final_prompt_list = []
            active_indices = torch.where(active_mask)[0].tolist()
            for idx in active_indices:
                question = original_questions[idx] if original_questions and idx < len(original_questions) else "the question"
                # Build intent prompt using the same format as in _process_next_obs
                intent_prompt = ""
                if self.enable_intent_injection and original_questions and idx < len(original_questions):
                    intent_prompt = f"{self.intent_injection_template.format(question=original_questions[idx])}"

                # Include collected evidence in final prompt
                evidence_str = self._format_collected_evidence(idx)
                evidence_section = f"\n\n### COLLECTED EVIDENCE\n{evidence_str}\n"

                final_prompt = f"\n<|im_start|>user\nThis is your last response. Based on the COLLECTED EVIDENCE, provide your final answer for the question: {intent_prompt} inside <answer> </answer> tags after <think> </think>.{evidence_section}\n<|im_end|>\n<|im_start|>assistant\n"
                final_prompt_list.append(final_prompt)

            # Tokenize final prompts
            final_prompt_ids = self.tokenizer(
                final_prompt_list,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids']

            # Pad final_prompt_ids to match the full batch size
            full_final_prompt_ids = torch.zeros((rollings.batch['input_ids'].shape[0], final_prompt_ids.shape[1]), dtype=torch.long)
            full_final_prompt_ids[active_mask] = final_prompt_ids

            # Update rollings with final prompt
            rollings = self._update_rolling_state(
                rollings,
                torch.zeros((rollings.batch['input_ids'].shape[0], 0), dtype=torch.long),
                full_final_prompt_ids
            )
            self._sync_multi_modal_after_truncation(rollings)

            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            rollings = self._raw_prompt_ids(rollings)

            active_mask = self.deactivate_batch(active_mask, rollings)

            if active_mask.sum():

                if 'multi_modal_inputs' in rollings.non_tensor_batch.keys():
                    rollings_active = DataProto.from_dict(
                        tensors={k: v[active_mask] for k, v in rollings.batch.items()},
                        non_tensors={k: v[active_mask] for k, v in rollings.non_tensor_batch.items()}
                    )
                else:
                    rollings_active = DataProto.from_dict({
                        k: v[active_mask] for k, v in rollings.batch.items()
                    })

                gen_output = self._generate_with_gpu_padding(rollings_active)

                meta_info = gen_output.meta_info
                responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])

                # Extract and store think content for active batch items in final round
                active_indices = torch.where(active_mask)[0].tolist()
                self._extract_and_store_think(responses_str, active_indices)

                # Record final rollout assistant responses for validation case recording
                if self.is_validation:
                    for j, batch_idx in enumerate(active_indices):
                        resp_content = responses_str[j]
                        if resp_content.endswith(self.tokenizer.eos_token):
                            resp_content = resp_content[:-len(self.tokenizer.eos_token)]
                        self.reasoning_traces[batch_idx].append({
                            "step": step + 1,
                            "role": "assistant",
                            "content": resp_content.strip(),
                            "is_final": True
                        })

                responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

                # # Execute in environment and process observations
                _, dones = self.execute_predictions(
                    responses_str, self.tokenizer.pad_token, active_mask, do_search=False
                )

                curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
                active_mask = active_mask * curr_active_mask
                active_num_list.append(active_mask.sum().item())

                # Keep full trajectory in original_right_side for reward computation
                original_right_side = self._update_right_side(
                    original_right_side,
                    responses_ids,
                    full_final_prompt_ids
                )

        print("ACTIVE_TRAJ_NUM:", active_num_list)

        # =================== raw prompt ids ===================
        rollings.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids

        # Rebuild full multi_modal_inputs from initial + ALL turns (not just the
        # sliding-window subset), so that image features match the image tokens
        # accumulated in original_right_side which contains ALL turns.
        full_mmd = []
        full_mmi = np.empty(batch_size, dtype=object)
        for idx in range(batch_size):
            images_list = list(saved_initial_mmd[idx]['image'])
            mmi_item = BatchFeature(dict(saved_initial_mmi[idx]))
            for turn in turn_history:
                turn_mmd_item = turn['obs_mmd'][idx]
                turn_mmi_item = turn['obs_mmi'][idx]
                if len(turn_mmd_item['image']) > 0 and 'pixel_values' in turn_mmi_item:
                    images_list.extend(turn_mmd_item['image'])
                    if 'pixel_values' in mmi_item:
                        mmi_item['pixel_values'] = torch.cat(
                            (mmi_item['pixel_values'], turn_mmi_item['pixel_values']), dim=0
                        )
                        mmi_item['image_grid_thw'] = torch.cat(
                            (mmi_item['image_grid_thw'], turn_mmi_item['image_grid_thw']), dim=0
                        )
                    else:
                        mmi_item['pixel_values'] = turn_mmi_item['pixel_values']
                        mmi_item['image_grid_thw'] = turn_mmi_item['image_grid_thw']
            full_mmd.append({'image': images_list})
            full_mmi[idx] = mmi_item

        rollings.non_tensor_batch['multi_modal_data'] = np.array(full_mmd, dtype=object)
        rollings.non_tensor_batch['multi_modal_inputs'] = full_mmi

        if not self.is_validation:
            rollings, original_right_side = self._add_noisy_multi_modal_data(rollings, original_right_side)

        retrievaled_images_array = np.empty(len(self.retrievaled_images), dtype=object)
        for idx in range(len(self.retrievaled_images)):
            retrievaled_images_array[idx] = self.retrievaled_images[idx]
        rollings.non_tensor_batch['retrievaled_images'] = retrievaled_images_array

        return self._compose_final_output(original_left_side, original_right_side, meta_info, rollings)
    
    def _add_noisy_multi_modal_data(self, rollings, original_right_side):
        image_padded = Image.new('RGB', (64, 64), (0, 0, 0))

        image_padded = process_image(image_padded, 256*256, 128*128)
        image_inputs = self.processor.image_processor([image_padded], return_tensors='pt')
        image_grid_thw = image_inputs['image_grid_thw']
        merge_length = self.processor.image_processor.merge_size**2
        padded_str = f"\n<|im_start|>user\n<|vision_start|>{self.processor.image_token * (image_grid_thw.prod() // merge_length)}<|vision_end|><|im_end|>"

        padded_str_list = []
        for idx, multi_modal_item in enumerate(rollings.non_tensor_batch['multi_modal_data']):
            if len(multi_modal_item['image']) == 0:
                padded_str_list.append(padded_str)
                rollings.non_tensor_batch['multi_modal_data'][idx]['image'].append(image_padded)
                rollings.non_tensor_batch['multi_modal_inputs'][idx] = image_inputs
            else:
                padded_str_list.append('')
            
        padded_ids = self.tokenizer(
            padded_str_list, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        original_right_side = self._update_right_side(
            original_right_side,
            padded_ids
        )
        return rollings, original_right_side


    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict,
                            rollings) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output,rollings.non_tensor_batch)
        final_output.meta_info.update(meta_info)
        
        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones = [], []

        bbox_list = [content for action, content in zip(cur_actions, contents) if action == 'bbox']

        if do_search:
            # Collect all search queries and send them in one batch request
            live_queries = []  # (batch_idx, query_string)
            for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
                if not active or action != 'search':
                    continue
                live_queries.append((i, contents[i]))

            # Batch-request all live queries
            if live_queries:
                query_strings = [q for _, q in live_queries]
                raw_results = []
                batch_size_req = 100
                for k in range(0, len(query_strings), batch_size_req):
                    batch_q = query_strings[k:k + batch_size_req]
                    response = requests.get(self.config.search_url, params={"queries": batch_q})
                    raw_results.extend(response.json())
                live_result_map = {bidx: result for (bidx, _), result in zip(live_queries, raw_results)}
            else:
                live_result_map = {}
        else:
            live_result_map = {}

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                elif action == 'search':
                    if do_search:
                        obs_result = live_result_map.get(i, [])
                    else:
                        obs_result = ''
                    self.search_counts[i] += 1
                    next_obs.append(obs_result if do_search else '')
                    dones.append(0)
                elif action == 'bbox':
                    try:
                        bbox_value = json.loads(bbox_list.pop(0))
                        if len(bbox_value) == 4 and bbox_value[0] >= 0 and bbox_value[1] >= 0 and bbox_value[2] >= 0 and bbox_value[3] >= 0:
                            next_obs.append(bbox_value)
                        else:
                            raise ValueError("Invalid bbox value")
                    except:
                        next_obs.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and user will return the searched results. Every time you retrieve an image, you have the option to crop it to obtain a clearer view, the format for coordinates is <bbox>[x1, y1, x2, y2]</bbox>. You can search as many times as your want. If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)
                else:
                    next_obs.append('\n<|im_start|>user\nYour previous action is invalid. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you want to search, you should put the query between <search> and </search>.\nIf you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>. Please try again.\n<|im_end|>\n<|im_start|>assistant\n')
                    dones.append(0)

        return next_obs, dones

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer|bbox)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content)
            
        return actions, contents

