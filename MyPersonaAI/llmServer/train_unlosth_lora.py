import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["DISABLE_TORCH_INDUCTOR"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TRITON_DISABLE"] = "1"
import torch
import torch._dynamo as dynamo
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from unsloth import FastLanguageModel



# train.py

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel
from peft import LoraConfig

# ⚙️ 禁用部分 PyTorch 编译优化，避免错误（特别适配 GTX 1050）
dynamo.config.suppress_errors = True
dynamo.reset()
dynamo.disable()
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["DISABLE_TORCH_COMPILE"] = "1"  # 强制关闭 torch.compile

# ✅ 加载数据集（确保路径正确）
dataset = load_dataset("json", data_files="../data/jsons/articles.jsonl")["train"]

# ✅ 加载 Unsloth 模型（Gemma 3 1B IT 4bit 量化）
MODEL = "unsloth/gemma-3-1b-it"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    #dtype=None,  # ✅ 使用 float32 避免 float16 在 GTX 1050 报错
    dtype=torch.float16, 
)

# ✅ 应用 LoRA 微调配置（Gradient Checkpointing 开启以节省显存）
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
    use_rslora=False,
    loftq_config=None,
    #dtype=None,
    dtype=torch.float16, 
    use_compile=False,   
)

# ✅ 格式化文本，添加 <s> 作为 BOS token（Gemma 风格）
def formatting_func(example):
    return f"<s>{example['text']}</s>"

dataset = dataset.map(lambda x: {"formatted": formatting_func(x)})

# ✅ 设置 tokenizer pad_token
tokenizer.pad_token = tokenizer.eos_token

# ✅ Tokenize 数据集，并添加 labels 字段
def tokenize(example):
    tokenized = tokenizer(
        example["formatted"],
        truncation=True,
        padding="max_length",
        max_length=2048
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

dataset = dataset.map(tokenize, batched=True)

# ✅ 训练参数（适配显存小的显卡）
training_args = TrainingArguments(
    output_dir="./gemma-1b-qlora-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=100,
    save_total_limit=2,
    num_train_epochs=3,
    fp16=True,  # ❗GTX 1050 不支持原生 float16 运算
    bf16=False,
    optim="paged_adamw_8bit",  # 使用 bitsandbytes 优化器
    lr_scheduler_type="cosine",
    learning_rate=2e-4,
    warmup_ratio=0.05,
    report_to="none",
    #use_compile=False,  
)

# ✅ 初始化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)

# ✅ 启动训练
trainer.train()

# ✅ 训练后测试输出（确保模型在正确设备上）
text = tokenizer("量子力学的基本假设包括：", return_tensors="pt").to("cuda")
model.to("cuda")
output = model.generate(**text, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("✅ 训练与测试结束！")
