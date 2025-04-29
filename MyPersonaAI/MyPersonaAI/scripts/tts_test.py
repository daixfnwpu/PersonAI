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
MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"

# 读取输入文本
with open("data/article.txt", "r", encoding="utf-8") as f:
    text = f.read().strip()

# 初始化模型（会自动下载模型文件）
tts = TTS(model_name=MODEL_NAME, progress_bar=True)

# 合成语音并保存
tts.tts_to_file(text=text, file_path="data/output.wav")

print("✅ 合成完成！语音已保存为 output.wav")
