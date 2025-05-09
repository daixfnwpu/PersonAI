from torch import float16
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel

# ✅ 1. 设置模型和 LoRA 参数
model_id = "facebook/opt-125m"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0,
    bnb_4bit_compute_dtype=float16,  # ✅ 添加这一行
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=2,
    lora_alpha=8,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# ✅ 2. 加载和预处理数据集
# ✅ 2. 加载数据集
dataset = load_dataset("imdb", split="train[:100]")  # 用前100条数据演示

def preprocess(batch):
    prompts = ["Classify sentiment: " + text + "\nAnswer:" for text in batch["text"]]
    inputs = tokenizer(prompts, max_length=256, padding="max_length", truncation=True)

    labels = ["positive" if label == 1 else "negative" for label in batch["label"]]
    label_tokens = tokenizer(labels, max_length=2, padding="max_length", truncation=True)

    inputs["labels"] = label_tokens["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# ✅ 3. 训练参数
training_args = TrainingArguments(
    output_dir="./opt-125m-lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,
    bf16=False,
    gradient_checkpointing=True,
    report_to="none",
    label_names=["labels"]  # ✅ 显式指定标签字段名
)

# ✅ 4. 启动训练
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()

# ✅ 5. 加载训练后的模型并测试
model.save_pretrained("./opt125m_lora_adapter")
print("✅ LoRA adapter 已保存")

# ✅ 6. 加载并测试模型
print("🔄 加载 LoRA adapter 并测试输出")
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./opt125m_lora_adapter")
model.eval()

# ✅ 7. 推理示例
input_text = "Classify sentiment: This movie was absolutely fantastic and inspiring!\nAnswer:"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs,
                         max_new_tokens=50,
                         do_sample=True,
                         temperature=0.7,
                         top_k=50,
                         top_p=0.9,)
print("💬 模型输出:", tokenizer.decode(outputs[0], skip_special_tokens=True))