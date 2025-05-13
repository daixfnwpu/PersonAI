from PIL import Image
import torch
from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

# 加载配置
cfg = Config(parse_args=False)
cfg.model_cfg = 'eval_configs/minigpt4_eval.yaml'
cfg.model_cfg.device_8bit = True  # 如果使用 bitsandbytes 加速

# 注册模型
model = registry.get_model_class(cfg.model.arch).from_config(cfg.model_cfg)
model.eval()
model.cuda()

# 加载图像
image = Image.open("your_image.png").convert("RGB")

# 构建视觉+语言输入
prompt = "请描述这张图片的内容："
res = model.generate(image, prompt)

print("模型输出：", res)
