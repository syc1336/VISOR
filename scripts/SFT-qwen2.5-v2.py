"""
SFT warm-up v2 —— 基于 SlideVQA-SFT-v2.json（≥3次搜索轨迹）
数据量：~700条，目标：让模型学会检索≥3次后再回答，为后续RL热身
"""
import torch
import json
import os
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from qwen_vl_utils import process_vision_info
from functools import partial


# ================= 1. 数据集定义 =================
class VQADataset(Dataset):
    def __init__(self, json_path, processor):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"找不到数据集文件: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


# ================= 2. 多模态动态 padding collator =================
def multimodal_collator(batch, processor):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

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
        "input_ids": padded["input_ids"],
        "attention_mask": padded["attention_mask"],
        "labels": padded_labels,
    }

    for k in batch[0].keys():
        if k in batch_out:
            continue
        vals = [b[k] for b in batch]
        if torch.is_tensor(vals[0]):
            batch_out[k] = torch.stack(vals)
        else:
            batch_out[k] = vals

    return batch_out


# ================= 3. 训练主函数 =================
def train():
    model_id = "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/qwen2.5_vl_full_final-SlideVQAsft"

    processor = AutoProcessor.from_pretrained(model_id)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # 只训练语言模型部分，视觉编码器冻结
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

    train_dataset = VQADataset(
        "/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-v2.json",
        processor,
    )

    training_args = TrainingArguments(
        output_dir="./qwen2.5_vl_sftv2_ckpt",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
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
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
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

    trainer.save_model("./qwen2.5_vl_sftv2_final")
    processor.save_pretrained("./qwen2.5_vl_sftv2_final")


if __name__ == "__main__":
    train()
