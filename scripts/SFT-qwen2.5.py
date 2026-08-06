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

        # 预缓存 token ids，避免每条数据重复编码
        self.im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.nl_id = processor.tokenizer.encode("\n", add_special_tokens=False)[0]
        self.assistant_ids = processor.tokenizer.encode("assistant", add_special_tokens=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        messages = item["messages"]

        # 解析图像/视频路径
        image_inputs, video_inputs = process_vision_info(messages)

        # 应用聊天模板
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

        # 移除 batch 维度
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # labels: 只在 assistant 回复部分计算 loss，其余设为 -100
        input_ids = inputs["input_ids"]
        labels = torch.full_like(input_ids, -100)

        im_start_id = self.im_start_id
        im_end_id = self.im_end_id
        nl_id = self.nl_id
        assistant_ids = self.assistant_ids

        ids_list = input_ids.tolist()
        n = len(ids_list)
        i = 0
        while i < n:
            if ids_list[i] == im_start_id:
                # 检查是否是 assistant turn
                j = i + 1
                is_assistant = True
                for k, aid in enumerate(assistant_ids):
                    if j + k >= n or ids_list[j + k] != aid:
                        is_assistant = False
                        break
                if is_assistant:
                    # 跳过 <|im_start|>assistant\n，从内容开始算 loss
                    content_start = j + len(assistant_ids)
                    if content_start < n and ids_list[content_start] == nl_id:
                        content_start += 1
                    # 找到对应的 <|im_end|>
                    content_end = content_start
                    while content_end < n and ids_list[content_end] != im_end_id:
                        content_end += 1
                    # assistant 内容 + <|im_end|> 参与 loss
                    labels[content_start:content_end + 1] = input_ids[content_start:content_end + 1]
                    i = content_end + 1
                    continue
            i += 1

        inputs["labels"] = labels
        return inputs


# ================= 2. 多模态动态 padding collator =================
def multimodal_collator(batch, processor):
    # 文本字段
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]

    # tokenizer 动态 padding
    padded = processor.tokenizer.pad(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
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

    # 处理其余多模态字段（如 pixel_values、image_grid_thw、vision_token_type_ids 等）
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
    model_id = "/mnt/cfs_algo_bj/models/opensource_model/Qwen/Qwen2.5-VL-7B-Instruct"

    # 加载 processor
    processor = AutoProcessor.from_pretrained(model_id)

    # 加载模型（全量微调，不使用 device_map）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    # 确保所有参数参与训练
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

    # 加载数据集
    train_dataset = VQADataset("/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/data/SlideVQA-SFT-verification.json", processor)

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen2.5_vl_full_sft_8gpu",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        bf16=True,
        deepspeed="/mnt/cfs_algo_bj/workspace/shenyucheng/VRAG/scripts/ds_config.json",
        num_train_epochs=3,
        learning_rate=1.0e-5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
    )

    # 构造 collator
    collator = partial(multimodal_collator, processor=processor)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    # 开始训练
    trainer.train()

    # 保存最终模型
    trainer.save_model("./qwen2.5_vl_full_final-SlideVQAsft")
    processor.save_pretrained("./qwen2.5_vl_full_final-SlideVQAsft")


if __name__ == "__main__":
    train()
