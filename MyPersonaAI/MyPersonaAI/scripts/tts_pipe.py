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
import edge_tts
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

    # 你可以换成别的模型，比如 XTTS v2 等
    MODEL_NAME_CN = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
    MODEL_NAME_EN ="tts_models/en/ljspeech/tacotron2-DDC"
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


# 运行异步任务
if __name__ == "__main__":
    asyncio.run(coqui_tts_impl())