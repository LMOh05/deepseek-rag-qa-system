"""
LoRA 微调 Qwen2.5-1.5B-Instruct for RAG 问答
===========================================
这是 Colab Notebook 对应的独立 Python 脚本，方便在 GPU 服务器上直接运行。

运行要求: T4 GPU (16GB) 或更高
预计时间: 下载 5min + 训练 30-45min
"""

import os
import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
)

# ============================================================
# 配置
# ============================================================
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "./qwen25-1.5b-rag-lora"
OUTPUT_MERGED = "./qwen25-1.5b-rag-merged"

LORA_CONFIG = {
    "r": 8,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": TaskType.CAUSAL_LM,
}

TRAINING_CONFIG = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 100,
    "fp16": True,
    "output_dir": OUTPUT_DIR,
}

SYSTEM_PROMPT = """你是一个基于文档知识的问答助手。请严格依据提供的参考资料回答问题。
规则：
1. 只使用参考资料中的信息回答，不要编造
2. 如果资料不足以回答，请明确说"根据现有资料无法回答"
3. 回答末尾标注来源编号，如 [来源:1,2]
4. 回答简洁准确，避免冗余"""


def load_training_data(data_path: str = None):
    """加载训练数据并格式化为 instruction 格式"""
    if data_path is None:
        # 从默认位置加载
        data_path = Path(__file__).parent.parent / "data" / "training_data.json"

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    formatted = []
    for sample in raw["samples"]:
        text = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{sample['instruction']}<|im_end|>
<|im_start|>assistant
{sample['output']}<|im_end|>"""
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def tokenize_function(examples, tokenizer, max_length=1024):
    """分词函数"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def train():
    """主训练流程"""
    print("=" * 60)
    print("LoRA 微调 Qwen2.5-1.5B for RAG 问答")
    print(f"LoRA rank: {LORA_CONFIG['r']}, alpha: {LORA_CONFIG['lora_alpha']}")
    print(f"Epochs: {TRAINING_CONFIG['num_train_epochs']}, LR: {TRAINING_CONFIG['learning_rate']}")
    print("=" * 60)

    # 1. 加载 tokenizer
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载训练数据
    print("\n[2/6] Loading training data...")
    dataset = load_training_data()
    print(f"   Training samples: {len(dataset)}")

    # 3. 分词
    print("\n[3/6] Tokenizing...")
    tokenized = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # 4. 加载基座模型（4-bit 量化以适配 T4 16GB）
    print(f"\n[4/6] Loading base model: {MODEL_NAME}")
    print("   Using 4-bit quantization (NF4)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False  # 训练时关闭 KV cache

    # 5. 配置 LoRA
    print(f"\n[5/6] Configuring LoRA...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 6. 训练
    print(f"\n[6/6] Training...")
    training_args = TrainingArguments(**TRAINING_CONFIG)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 保存 LoRA 权重
    print(f"\nSaving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # 可选：合并权重
    print(f"\nMerging LoRA weights to {OUTPUT_MERGED}...")
    merged = model.merge_and_unload()
    merged.save_pretrained(OUTPUT_MERGED, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_MERGED)

    print("\nDone! LoRA fine-tuning completed successfully.")

    # 训练总结
    print(f"\n输出文件:")
    print(f"  LoRA adapter: {OUTPUT_DIR}/  (约 10-20MB)")
    print(f"  Merged model: {OUTPUT_MERGED}/  (约 3GB)")
    print(f"\n下一步:")
    print(f"  1. 下载 adapter 或 merged model 到本地")
    print(f"  2. 用 notebook 中的 inference cell 测试效果")
    print(f"  3. 对比: 原版 vs LoRA vs DeepSeek API")

    return trainer, model, tokenizer


def test_inference(model, tokenizer, test_question: str, context: str):
    """微调后推理测试"""
    prompt = f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
根据以下参考资料回答问题：

参考资料：
{context}

问题：{test_question}<|im_end|>
<|im_start|>assistant
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
        top_p=0.9,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 只取 assistant 部分
    if "<|im_start|>assistant" in response:
        response = response.split("<|im_start|>assistant")[-1].strip()
    return response


if __name__ == "__main__":
    train()
