from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# ✅ 1. 设置模型和 LoRA 参数
model_id = "google/flan-t5-small"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    llm_int8_threshold=6.0
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)
model = get_peft_model(model, lora_config)

# ✅ 2. 加载数据集
dataset = load_dataset("imdb", split="train[:100]")  # 用前100条数据演示

def preprocess(example):
    prompt = "Classify sentiment: " + example["text"]
    inputs = tokenizer(prompt, max_length=256, padding="max_length", truncation=True)
    labels = tokenizer("positive" if example["label"] == 1 else "negative", max_length=2, padding="max_length", truncation=True)
    inputs["labels"] = labels["input_ids"]
    return inputs

tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# ✅ 3. 训练参数
training_args = TrainingArguments(
    output_dir="./flan-t5-small-lora-output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=5,
    save_strategy="epoch",
    fp16=False,  # GTX 1050 不支持 fp16
    bf16=False,
    report_to="none"
)

# ✅ 4. 启动训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

from peft import PeftModel
from transformers import AutoTokenizer

# 重新加载并测试模型
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, "./flan-t5-small-lora-output")

tokenizer = AutoTokenizer.from_pretrained(model_id)
input_text = "Classify sentiment: This movie was absolutely fantastic and inspiring!"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))  # 应输出 positive