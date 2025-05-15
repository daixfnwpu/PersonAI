

# -*- coding: utf-8 -*-
'''
@File    :   coquiTTS.py
@Author  :   daixfnwpu 
'''

from ..builder import TTSEngines
from ..engineBase import BaseEngine
from typing import List, Optional
from aiServer.utils import logger
from aiServer.utils import TextMessage, AudioMessage, AudioFormatType
from aiServer.utils.audio import mp3ToWav
from TTS.api import TTS
import soundfile as sf
import numpy as np

__all__ = ["coquiTTSAPI"]

# 你可以换成别的模型，比如 XTTS v2 等
MODEL_NAME_CN = "tts_models/zh-CN/baker/tacotron2-DDC-GST"
MODEL_NAME_EN ="tts_models/en/ljspeech/tacotron2-DDC"
MODEL_NAME_ALL="tts_models/multilingual/multi-dataset/xtts_v2"
MODEL_NAME = MODEL_NAME_ALL
speaker_wav_path = "data/speaker.wav"

@TTSEngines.register("coquiTTSAPI")
class CoquiTTSAPI(BaseEngine):
    
    def setup(self):
        self.language = "en"
        self.sample_rate = 24000
        self.pause_sec = 0.5
        self.pause = np.zeros(int(self.sample_rate * self.pause_sec), dtype=np.float32)
        self.ttsmodel = TTS(model_name=MODEL_NAME_ALL, gpu=True,progress_bar=True)
        return super().setup()
    async def run(self, input: TextMessage, **kwargs) -> Optional[TextMessage]:
        try: 
            combined_audio = []
            subtitles = []
            current_time = 0  # 当前时间（秒）
            pause_duration = 0.5  # 每段之间插入 0.5 秒停顿
            # voice = self.cfg.PER
            # if 'voice' in kwargs and kwargs['voice']:
            #     voice = kwargs['voice']
           # communicate = self.ttsmodel.tts(input) 
            # 遍历合成每段
            lines = input.splitlines()
            communicate = self.ttsmodel.tts(
                    text=input,
                    speaker_wav=speaker_wav_path,
                    language=self.language,
            )
            # for i, text in enumerate(lines, 1):
            #     print(f"[{i}] 合成语音：{text}")
                
            #     # 计算每段的持续时间
            #     duration = len(audio) / self.sample_rate

            #     # 生成字幕：时间格式转换为 SRT 所需格式
            #     start_time = current_time
            #     end_time = current_time + duration

            #     start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
            #     end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"

            #     subtitles.append(f"{i}")
            #     subtitles.append(f"{start_time_str} --> {end_time_str}")
            #     subtitles.append(f"{text}")
            #     subtitles.append("")  # 空行分隔

            #     # 更新当前时间
            #     current_time = end_time + pause_duration  # 下一段开始的时间加上停顿
            #     combined_audio.append(audio)
            #     if i < len(input):
            #         combined_audio.append(self.pause)  # 最后一段后无需添加 pause


            # # 拼接所有音频
            # final_audio = np.concatenate(combined_audio)

            # # 保存最终完整音频
            # final_output_path="data/segments_all.wav"
            data = b''
            async for message in communicate.stream():
                if message["type"] == "audio":
                    data += message["data"]

            # mp3 -> wav
            data = mp3ToWav(data)
            message = AudioMessage(
                data=data, 
                desc=input.data,
                format=AudioFormatType.WAV,
                sampleRate=16000,
                sampleWidth=2,
            )
            return message
        except Exception as e:
            logger.error(f"[TTS] Engine run failed: {e}", exc_info=True)
            return None