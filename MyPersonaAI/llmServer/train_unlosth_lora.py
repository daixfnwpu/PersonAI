from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

dataset = load_dataset("json", data_files="../data/jsons/articles.jsonl")["train"]
#"unsloth/gemma-3-1b-bnb-4bit"
tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it")

def tokenize(example):
    return tokenizer(example["text"])

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])


block_size = 2048

def group_texts(examples):
    concatenated = {k: sum(examples[k], []) for k in examples}
    total_len = len(concatenated["input_ids"])
    total_len = (total_len // block_size) * block_size
    return {
        k: [t[i:i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }

lm_dataset = tokenized.map(group_texts, batched=True)


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-1b-it",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True
)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

training_args = TrainingArguments(
    output_dir="outputs/",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=data_collator,
)

trainer.train()
# 保存模型
model.save_pretrained("finetuned-gemma-1.1b-articles")
tokenizer.save_pretrained("finetuned-gemma-1.1b-articles")
# from unsloth import FastLanguageModel
# from transformers import AutoTokenizer

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/gemma-1.1b-bnb-4bit",
#     max_seq_length=2048,
#     dtype=None,
#     load_in_4bit=True,
# )

# FastLanguageModel.prepare_model_for_training(model)

text = tokenizer("量子力学的基本假设包括：", return_tensors="pt").to("cuda")
output = model.generate(**text, max_new_tokens=100)
print(tokenizer.decode(output[0]))