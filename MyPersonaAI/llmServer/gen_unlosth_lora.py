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
from peft import LoraConfig, PeftModel

# ⚙️ 禁用部分 PyTorch 编译优化，避免错误（特别适配 GTX 1050）
dynamo.config.suppress_errors = True
dynamo.reset()
dynamo.disable()
os.environ["PYTORCH_NVFUSER_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DEBUG"] = "0"
os.environ["DISABLE_TORCH_COMPILE"] = "1"  # 强制关闭 torch.compile

# ✅ 加载 Unsloth 模型（Gemma 3 1B IT 4bit 量化）
MODEL = "unsloth/gemma-3-1b-it"

from transformers import AutoModelForCausalLM, AutoTokenizer


# 先加载基础模型（未包装 LoRA 的原模型）
base_model = FastLanguageModel.from_pretrained(
    model_name=MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
    dtype=torch.float16,
)

# 再加载 LoRA 微调权重，合并到基础模型
model = PeftModel.from_pretrained(base_model, "./gemma-1b-qlora-finetuned")

# 之后 model 就是带有 LoRA 微调效果的完整模型
model.eval()

tokenizer = AutoTokenizer.from_pretrained("./gemma-1b-qlora-finetuned")
# ✅ 训练后测试输出（确保模型在正确设备上）
text = tokenizer("量子力学的基本假设包括：", return_tensors="pt").to("cuda")
model.to("cuda")
output = model.generate(**text, max_new_tokens=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
print("✅ 训练与测试结束！")