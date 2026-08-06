"""
SFT warm-up v3 —— 基于 SlideVQA-SFT-v3.json（verification 搜索轨迹）

与 v2 的主要差异：
  - 数据格式：messages 含 <image> token + 独立 images 列表
  - 多图：每条样本最多 12 张图，pixel_values 需 concat 而非 stack
  - 冻结视觉编码器，只训练语言模型部分（同 v2）
"""

import torch
import json
import os
import re
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info
from functools import partial

_IMAGE_RE = re.compile(r"<image>")


# ================= 1. 消息格式转换 =================
def convert_messages(messages: list, images: list) -> list:
    """
    将 v3 格式（text 含 <image> token，images 单独列表）转换为
    Qwen2.5-VL 标准格式（content 为 list of dicts）。
    images 列表中的路径按 <image> token 出现顺序一一对应。
    """
    img_iter = iter(images)
    converted = []
    for msg in messages:
        role = msg["role"]
        content_str = msg["content"]
        parts = _IMAGE_RE.split(content_str)
        n_imgs = len(parts) - 1

        content = []
        for i, text_part in enumerate(parts):
            if text_part:
                content.append({"type": "text", "text": text_part})
            if i < n_imgs:
                img_path = next(img_iter)
                content.append({"type": "image", "image": img_path})

        converted.append({"role": role, "content": content})
    return converted


# ================= 2. 数据集定义 =================
class VQADatasetV3(Dataset):
    def __init__(self, json_path: str, processor):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据集文件: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = convert_messages(item["messages"], item["images"])

        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# ================= 3. 多图多模态 collator =================
def multimodal_collator(batch, processor):
    """
    Qwen2.5-VL 多图 collator：
      - input_ids / attention_mask / labels：tokenizer padding
      - pixel_values / image_grid_thw：各样本 concat（dim=0）
    """
    # ── 文本侧 padding ──
    input_ids     = [b["input_ids"]     for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels         = [b["labels"]         for b in batch]

    padded = processor.tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        padding=True,
        return_tensors="pt",
    )

    padded_labels = processor.tokenizer.pad(
        {"input_ids": labels},
        padding=True,
        return_tensors="pt",
    )["input_ids"]
    padded_labels[padded_labels == processor.tokenizer.pad_token_id] = -100

    batch_out = {
        "input_ids":      padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels":         padded_labels,
    }

    # ── 图像侧：concat 而非 stack ──
    skip_keys = set(batch_out.keys())
    for k in batch[0].keys():
        if k in skip_keys:
            continue
        vals = [b[k] for b in batch if k in b]
        if not vals:
            continue
        if torch.is_tensor(vals[0]):
            if k in ("pixel_values", "image_grid_thw"):
                # 不同样本图片数量/patch数不同，沿 dim=0 concat
                batch_out[k] = torch.cat(vals, dim=0)
            else:
                try:
                    batch_out[k] = torch.stack(vals)
                except Exception:
                    batch_out[k] = torch.cat(vals, dim=0)
        else:
            batch_out[k] = vals

    return batch_out


# ================= 4. 训练主函数 =================
def train():
    model_id = "/mnt/cfs_algo_bj/models/experiments/shenyucheng/checkpoints/SFT/1500_slide/qwen2.5_vl_full_final-7B"

    processor = AutoProcessor.from_pretrained(model_id)
    # 限制图像分辨率，降低 pixel_values 显存占用
    processor.image_processor.max_pixels = 768 * 28 * 28
    processor.image_processor.min_pixels = 4 * 28 * 28

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False
    # gradient checkpointing 以激活值换显存
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    # 冻结视觉编码器，只训练语言模型部分
    model.train()
    for name, param in model.named_parameters():
        name_lower = name.lower()
        if (
            "vision" in name_lower
            or "visual" in name_lower
            or "projector" in name_lower
            or "mm_" in name_lower
        ):
            param.requires_grad = False
        else:
            param.requires_grad = True

    train_dataset = VQADatasetV3(
        "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-v3.json",
        processor,
    )

    training_args = TrainingArguments(
        output_dir="./qwen2.5_vl_sftv3_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        bf16=True,
        deepspeed="/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/scripts/ds_config.json",
        num_train_epochs=3,
        learning_rate=1e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=False,
        ddp_find_unused_parameters=False,
    )

    collator = partial(multimodal_collator, processor=processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model("./qwen2.5_vl_sftv3_final")
    processor.save_pretrained("./qwen2.5_vl_sftv3_final")


if __name__ == "__main__":
    train()
