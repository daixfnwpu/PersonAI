# from TTS.api import TTS
# from torch.serialization import add_safe_globals
# from TTS.utils.radam import RAdam

# add_safe_globals([RAdam])

# ##tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
# tts = TTS(model_name="tts_models/zh-CN/baker/tacotron2-DDC-GST")

# tts.tts_to_file(text="你好，我是虚拟人", file_path="output.wav")


import os
import torch
import collections
import types
import builtins
import numpy as np
from torch.serialization import add_safe_globals
from TTS.utils.radam import RAdam
import asyncio

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import VitsModel, AutoTokenizer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from TTS.api import TTS
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

####pip install git+https://github.com/coqui-ai/TTS.git  don't use pip to install this tts
###pip install edge-tts



import re


def is_chinese(text):
    # 通过正则表达式匹配中文字符
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def is_english(text):
    # 检查是否包含英文字符（字母和空格）
    return bool(re.search(r'[a-zA-Z]', text))

def detect_language(text):
    if is_chinese(text):
        return "CN"
    elif is_english(text):
        return "EN"
    else:
        return "Unknown"



# 你可以换成别的模型，比如 XTTS v2 等
MODEL_NAME_CN = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
MODEL_NAME_EN ="tts_models/en/ljspeech/tacotron2-DDC"
MODEL_NAME_EN ="facebook/fastspeech2-ljspeech"
MODEL_NAME_EN ="facebook/mms-tts-eng"

## 加载 Common Voice 数据集（选择语言 en）
#dataset = load_dataset("common_voice", "en")
# 1. 加载数据集（假设你使用的是自定义数据集）
# 如果使用 `datasets` 库，可以根据需求自定义数据集
def load_custom_dataset():
    # 加载数据集，这里假设你的数据集已经准备好
   # dataset = load_dataset('common_voice', data_files={'train': 'data/metadata.csv'})
    dataset = load_dataset("common_voice", "en")
    return dataset

# 2. 初始化 TTS 模型（选择适合你的预训练模型）
MODEL_NAME = MODEL_NAME_EN  # 可以根据需求选择其他模型
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = VitsModel.from_pretrained(MODEL_NAME)

# 3. 定义 LoRA 配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵秩（控制微调规模）
    lora_alpha=16,  # 学习率控制
    lora_dropout=0.1,  # LoRA的Dropout率
  #  task_type="text2text-generation",  # 任务类型
)

# 4. 使用 LoRA 获取适配后的模型
model = get_peft_model(model, lora_config)

# 5. 定义数据加载器（假设你有自定义的音频和文本对数据集）
def collate_fn(batch):
    input_ids = tokenizer([item['text'] for item in batch], padding=True, truncation=True, return_tensors="pt").input_ids
    return input_ids

dataset = load_custom_dataset()
train_dataloader = DataLoader(dataset['train'], batch_size=8, collate_fn=collate_fn)

# 6. 训练过程
optimizer = AdamW(model.parameters(), lr=5e-5)

def train_model():
    model.train()
    for epoch in range(10):  # 训练10个Epoch
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            input_ids = batch
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

train_model()

# 7. 保存微调后的模型
model.save_pretrained("fine_tuned_model")
model.eval()
# 8. 使用微调后的模型进行语音合成
def generate_audio_with_finetuned_model(text):
    tts = TTS(model_name="fine_tuned_model", progress_bar=True).to("cuda")
    tts.tts_to_file(text=text, file_path="output.wav")
    print("✅ 合成完成！语音已保存为 output.wav")

# 9. 使用微调后的模型合成语音
if __name__ == "__main__":
    text = "This is a fine-tuned voice synthesis example."
    generate_audio_with_finetuned_model(text)


async def coqui_tts_impl():
# 信任 RAdam 优化器用于反序列化
    add_safe_globals([
        dict,
        RAdam,
        collections.defaultdict,
        collections.OrderedDict,
        types.SimpleNamespace,
        builtins.slice,
        builtins.range,
        builtins.complex,
        builtins.set,
        builtins.frozenset,
        torch.Size,
        torch.device,
        torch.dtype,
        np.float32,
        np.int32,
        np.ndarray
    ])

    from TTS.api import TTS


    # 读取输入文本
    with open("data/article.txt", "r", encoding="utf-8") as f:
        text = f.read().strip()
    

    # 初始化模型（会自动下载模型文件）
    if detect_language(text) == "EN":
        tts = TTS(model_name=MODEL_NAME_EN, progress_bar=True).to("cuda")
    else: 
        tts = TTS(model_name=MODEL_NAME_CN, progress_bar=True).to("cuda")
    # 合成语音并保存
    tts.tts_to_file(text=text, file_path="data/output.wav")

    print("✅ 合成完成！语音已保存为 output.wav")



# # 运行异步任务
# if __name__ == "__main__":
#     asyncio.run(coqui_tts_impl())