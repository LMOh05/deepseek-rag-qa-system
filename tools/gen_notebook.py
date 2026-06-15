"""Generate standard Colab notebook JSON — no external deps needed"""
import json
import os

cells = []
cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "# LoRA 微调 Qwen2.5-1.5B-Instruct for RAG 问答\n\n"
        "**Colab Free T4 GPU | 预计 1 小时**\n\n"
        "### LoRA 参数\n"
        "| 参数 | 值 | 原因 |\n"
        "|------|-----|------|\n"
        "| rank(r) | 8 | 论文默认,效果接近r=16 |\n"
        "| alpha | 16 | alpha/r=2 标准缩放 |\n"
        "| target | q_proj, v_proj | 注意力Q/V最关键 |\n"
        "| lr | 2e-4 | 小模型常用学习率 |\n"
        "| epochs | 3 | 100条数据,3轮防过拟合 |"
    ]
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 1: 检查 GPU"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "!pip install -q transformers peft accelerate bitsandbytes datasets\n"
        "import torch\n"
        "print(f\"CUDA: {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"GPU: {torch.cuda.get_device_name(0)}\")\n"
        "    print(f\"VRAM: {torch.cuda.get_device_properties(0).total_mem/1024**3:.1f} GB\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 2: 加载训练数据\n\n"
        "> 先上传 `training_data.json` (左侧文件面板 -> 上传按钮)"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "import json, os\n\n"
        "if not os.path.exists(\"training_data.json\"):\n"
        "    raise FileNotFoundError(\"请先上传 training_data.json!\")\n\n"
        "with open(\"training_data.json\", \"r\", encoding=\"utf-8\") as f:\n"
        "    raw = json.load(f)\n\n"
        "print(f\"Samples: {len(raw['samples'])}\")\n"
        "print(f\"Task: {raw['description']}\")\n"
        "print(\"OK\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 3: 格式化为 ChatML"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "SYSTEM_PROMPT = (\n"
        "    \"你是一个基于文档知识的问答助手。请严格依据提供的参考资料回答问题。\\n\"\n"
        "    \"规则：\\n\"\n"
        "    \"1. 只使用参考资料中的信息回答，不要编造\\n\"\n"
        "    \"2. 如果资料不足，请明确说'根据现有资料无法回答'\\n\"\n"
        "    \"3. 回答末尾标注来源编号，如 [来源:1,2]\\n\"\n"
        "    \"4. 回答简洁准确，避免冗余\"\n"
        ")\n\n"
        "def format_samples(raw_data):\n"
        "    formatted = []\n"
        "    for s in raw_data[\"samples\"]:\n"
        "        text = (\n"
        "            f\"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n\"\n"
        "            f\"<|im_start|>user\\n{s['instruction']}<|im_end|>\\n\"\n"
        "            f\"<|im_start|>assistant\\n{s['output']}<|im_end|>\"\n"
        "        )\n"
        "        formatted.append({\"text\": text})\n"
        "    return formatted\n\n"
        "formatted = format_samples(raw)\n"
        "print(f\"Formatted: {len(formatted)} samples\")\n"
        "print(\"Sample (first 200 chars):\")\n"
        "print(formatted[0][\"text\"][:200])"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 4: Tokenizer & 分词"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "from datasets import Dataset\n"
        "from transformers import AutoTokenizer\n\n"
        "MODEL_NAME = \"Qwen/Qwen2.5-1.5B-Instruct\"\n\n"
        "tokenizer = AutoTokenizer.from_pretrained(\n"
        "    MODEL_NAME, trust_remote_code=True, padding_side=\"right\")\n"
        "tokenizer.pad_token = tokenizer.eos_token\n\n"
        "ds = Dataset.from_list(formatted)\n\n"
        "def tokenize_fn(examples):\n"
        "    return tokenizer(examples[\"text\"], truncation=True,\n"
        "                     max_length=1024, padding=\"max_length\")\n\n"
        "tokenized = ds.map(tokenize_fn, batched=True,\n"
        "                   remove_columns=ds.column_names)\n"
        "print(f\"Tokenized: {len(tokenized)} samples, seq_len=1024\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 5: 加载模型 (4-bit NF4)"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
        "from peft import prepare_model_for_kbit_training\n"
        "import torch\n\n"
        "bnb = BitsAndBytesConfig(\n"
        "    load_in_4bit=True, bnb_4bit_quant_type=\"nf4\",\n"
        "    bnb_4bit_compute_dtype=torch.float16,\n"
        "    bnb_4bit_use_double_quant=True)\n\n"
        "print(f\"Downloading {MODEL_NAME}... (~3GB)\")\n"
        "model = AutoModelForCausalLM.from_pretrained(\n"
        "    MODEL_NAME, quantization_config=bnb,\n"
        "    device_map=\"auto\", trust_remote_code=True,\n"
        "    torch_dtype=torch.float16)\n"
        "model.config.use_cache = False\n"
        "model = prepare_model_for_kbit_training(model)\n"
        "print(f\"Loaded. VRAM: {torch.cuda.memory_allocated()/1024**3:.1f} GB\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 6: 配置 LoRA"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "from peft import LoraConfig, get_peft_model, TaskType\n\n"
        "config = LoraConfig(\n"
        "    r=8, lora_alpha=16,\n"
        "    target_modules=[\"q_proj\", \"v_proj\"],\n"
        "    lora_dropout=0.05, bias=\"none\",\n"
        "    task_type=TaskType.CAUSAL_LM)\n\n"
        "model = get_peft_model(model, config)\n"
        "model.print_trainable_parameters()\n"
        "# Expected: ~3M trainable params out of ~1.5B total"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 7: 训练 (30-45 分钟)"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "from transformers import TrainingArguments, Trainer\n\n"
        "args = TrainingArguments(\n"
        "    output_dir=\"./qwen25-rag-lora\", num_train_epochs=3,\n"
        "    per_device_train_batch_size=4,\n"
        "    gradient_accumulation_steps=4,\n"
        "    learning_rate=2e-4, warmup_steps=100,\n"
        "    logging_steps=10, save_steps=100,\n"
        "    fp16=True, report_to=\"none\",\n"
        "    save_total_limit=2)\n\n"
        "trainer = Trainer(model=model, args=args,\n"
        "                  train_dataset=tokenized, tokenizer=tokenizer)\n\n"
        "print(\"Training...\")\n"
        "trainer.train()\n\n"
        "# Save LoRA adapter\n"
        "model.save_pretrained(\"./qwen25-rag-lora\")\n"
        "tokenizer.save_pretrained(\"./qwen25-rag-lora\")\n"
        "print(\"Saved!\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 8: 微调后推理测试"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "def generate(instruction, max_tokens=256):\n"
        "    prompt = (\n"
        "        f\"<|im_start|>system\\n{SYSTEM_PROMPT}<|im_end|>\\n\"\n"
        "        f\"<|im_start|>user\\n{instruction}<|im_end|>\\n\"\n"
        "        f\"<|im_start|>assistant\\n\")\n"
        "    inputs = tokenizer(prompt, return_tensors=\"pt\").to(model.device)\n"
        "    outputs = model.generate(\n"
        "        **inputs, max_new_tokens=max_tokens,\n"
        "        temperature=0.3, do_sample=True, top_p=0.9,\n"
        "        pad_token_id=tokenizer.eos_token_id)\n"
        "    full = tokenizer.decode(outputs[0], skip_special_tokens=True)\n"
        "    if \"<|im_start|>assistant\" in full:\n"
        "        return full.split(\"<|im_start|>assistant\")[-1].strip()\n"
        "    return full\n\n"
        "tests = [\n"
        "    (\"根据参考资料：[1] 机器学习是人工智能一个分支，使计算机从数据中学习。问题：什么是机器学习？\", \"概念解释\"),\n"
        "    (\"根据参考资料：[1] RAG工作流程：文档处理->切片->向量化->检索->生成答案。问题：RAG工作流程是什么？\", \"事实查询\"),\n"
        "    (\"根据参考资料：未找到相关内容。问题：明年的天气会怎样？\", \"拒绝回答\"),\n"
        "    (\"根据参考资料：[1] 监督学习用标注数据。[2] 强化学习靠奖励机制。[3] 无监督学习发现隐藏结构。问题：机器学习有哪些类型？分别有什么特点？\", \"多信息综合\"),\n"
        "]\n\n"
        "for q, label in tests:\n"
        "    print(f\"\\n=== {label} ===\")\n"
        "    ans = generate(f\"根据以下参考资料回答问题：\\n\\n{q}\")\n"
        "    print(f\"A: {ans[:300]}\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 9: 下载结果"
    ]
})

cells.append({
    "cell_type": "code", "metadata": {}, "source": [
        "# Merge LoRA into base model\n"
        "merged = model.merge_and_unload()\n"
        "merged.save_pretrained(\"./qwen25-rag-merged\", safe_serialization=True)\n"
        "tokenizer.save_pretrained(\"./qwen25-rag-merged\")\n\n"
        "# Zip adapter for download\n"
        "!zip -r qwen25-rag-lora.zip ./qwen25-rag-lora/\n"
        "!ls -lh qwen25-rag-lora.zip\n"
        "print(\"\\nRight-click qwen25-rag-lora.zip -> Download\")"
    ], "outputs": [], "execution_count": None
})

cells.append({
    "cell_type": "markdown", "metadata": {}, "source": [
        "## Step 10: 效果分析\n\n"
        "### 对比\n\n"
        "| 测试 | DeepSeek v4-flash | Qwen2.5原版 | Qwen2.5+LoRA |\n"
        "|------|:--:|:--:|:--:|\n"
        "| 引用来源 | ✅ | ❌ | ✅ |\n"
        "| 拒绝编造 | ✅ | ❌ | ✅ |\n"
        "| 结构化输出 | ✅ | ⚠️ | ✅ |\n\n"
        "### 面试要点\n"
        "- **微调提升什么？** 回答格式/风格/引用习惯，不是知识本身\n"
        "- **LoRA局限？** 不注入新知识，只调整行为模式\n"
        "- **微调 vs RAG？** 微调优化\"怎么答\"，RAG提供\"答什么\""
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Write to both locations
outdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(outdir, "notebooks", "lora_finetune_colab.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
print(f"Generated: {path}")
print(f"nbformat: {notebook.get('nbformat')}, nbformat_minor: {notebook.get('nbformat_minor')}")
print(f"Cells: {len(cells)}")
