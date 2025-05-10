# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoTokenizer
# from peft import LoraConfig, get_peft_model
# from datasets import load_dataset
# import torchaudio
# from transformers import Trainer, TrainingArguments

# # Step 1: Load Pretrained Model and Tokenizer
# model_name = "facebook/fastspeech2-ljspeech"
# model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Step 2: Apply LoRA for Fine-Tuning
# lora_config = LoraConfig(
#     r=4,  # LoRA rank
#     lora_alpha=32,  # LoRA alpha parameter
#     lora_dropout=0.1,  # LoRA dropout rate
#     target_modules=["encoder.block", "decoder.block"]  # Targeted modules for LoRA
# )

# # Get LoRA model
# lora_model = get_peft_model(model, lora_config)

# # Step 3: Load and Process Dataset (LJSpeech Example)
# dataset = load_dataset("ljspeech")

# def process_data(batch):
#     audio_path = batch["audio"]["path"]
#     waveform, sample_rate = torchaudio.load(audio_path)
    
#     # Extract Mel Spectrogram
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
    
#     # Tokenize text input
#     text = batch["text"]
#     return {
#         "input_ids": tokenizer(text, return_tensors="pt").input_ids.squeeze(),
#         "mel_spectrogram": mel_spectrogram
#     }

# # Process the dataset
# train_dataset = dataset["train"].map(process_data, remove_columns=["audio", "text"])

# # Step 4: Set Up Training Configuration
# training_args = TrainingArguments(
#     output_dir="./tts_lora_model",  # Output directory for model checkpoints
#     per_device_train_batch_size=4,  # Batch size
#     per_device_eval_batch_size=4,   # Eval batch size
#     logging_dir="./logs",           # Log directory
#     evaluation_strategy="epoch",    # Evaluation strategy
#     save_strategy="epoch",         # Save model checkpoints every epoch
#     num_train_epochs=3,            # Number of training epochs
#     fp16=True,                      # Use mixed precision training
#     logging_steps=100,              # Log every 100 steps
# )

# # Step 5: Create Trainer for Model Training
# trainer = Trainer(
#     model=lora_model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=train_dataset,  # You can replace this with a validation dataset
# )

# # Step 6: Train the Model
# trainer.train()

# # Step 7: Generate Speech from Text Input (Inference)
# lora_model.eval()

# # Sample input text
# input_text = "Hello, this is an example of TTS using LoRA."

# # Tokenize the input text
# input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# # Generate Mel Spectrogram from the input text
# with torch.no_grad():
#     mel_output = lora_model.generate(input_ids)

# # Convert Mel Spectrogram to Audio Waveform (This step requires a Vocoder like WaveGlow or Griffin-Lim)
# # For demonstration purposes, we simply print the Mel spectrogram
# print(mel_output)

from TTS.api import TTS
import soundfile as sf
import numpy as np
import os
from pydub import AudioSegment


MODEL_NAME_CN = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
MODEL_NAME_EN ="tts_models/en/ljspeech/tacotron2-DDC"
# 初始化模型（GPU 加速建议开启）
tts = TTS(model_name=MODEL_NAME_EN, gpu=True)

# 基本参数
speaker_wav_path = "data/speaker.wav"
# 加载 M4A 文件
audio_t = AudioSegment.from_file("data/tongtongen.m4a", format="m4a")

# 将其导出为 WAV 格式
audio_t.export(speaker_wav_path, format="wav")
language = "en"
sample_rate = 24000
pause_sec = 0.5

# 输出目录
os.makedirs("segments", exist_ok=True)

# 读取文本
with open("data/article.txt", "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"读取 {len(lines)} 段文本，准备生成语音...")

# 一次性提取说话人嵌入向量（只做一次）
print("提取说话人嵌入特征...")
#speaker_embedding = (speaker_wav_path)

# 静音段
pause = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)

# 合成音频片段并保存 + 追加到最终结果中
combined_audio = []
subtitles = []

current_time = 0  # 当前时间（秒）
pause_duration = 0.5  # 每段之间插入 0.5 秒停顿
# 遍历合成每段
for i, text in enumerate(lines, 1):
    print(f"[{i}] 合成语音：{text}")
    
    audio = tts.tts(
        text=text,
        speaker_wav=speaker_wav_path,
       # language=language,
    )
    
   # sf.write(f"data/segment_{i}.wav", audio, sample_rate)
# 添加到组合数组
     # 计算每段的持续时间
    duration = len(audio) / sample_rate

    # 生成字幕：时间格式转换为 SRT 所需格式
    start_time = current_time
    end_time = current_time + duration

    start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
    end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"

    subtitles.append(f"{i}")
    subtitles.append(f"{start_time_str} --> {end_time_str}")
    subtitles.append(f"{text}")
    subtitles.append("")  # 空行分隔

    # 更新当前时间
    current_time = end_time + pause_duration  # 下一段开始的时间加上停顿
    combined_audio.append(audio)
    if i < len(lines):
        combined_audio.append(pause)  # 最后一段后无需添加 pause


# 拼接所有音频
final_audio = np.concatenate(combined_audio)

# 保存最终完整音频
final_output_path="data/segments_all.wav"
sf.write(final_output_path, final_audio, sample_rate)

# 保存字幕文件（SRT 格式）
srt_output_path = "data/segments_all.srt"
with open(srt_output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(subtitles))

print(f"\n✅ 所有段落已处理完成，完整语音已保存为：{final_output_path}")
print(f"字幕文件已保存为：{srt_output_path}")


print("全部完成。")
