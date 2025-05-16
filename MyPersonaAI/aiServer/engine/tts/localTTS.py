# -*- coding: utf-8 -*-
'''
@File    :   localTTS.py
@Author  :   daixfnwpu 
'''

# from ..builder import TTSEngines
# from ..engineBase import BaseEngine
# from typing import List, Optional
# from aiServer.utils import logger
# from aiServer.utils import TextMessage, AudioMessage, AudioFormatType
# from aiServer.utils.audio import mp3ToWav
# from TTS.api import TTS


# __all__ = ["LocalAPI"]

# # 你可以换成别的模型，比如 XTTS v2 等
# MODEL_NAME = "tts_models/zh-CN/baker/tacotron2-DDC-GST"


# @TTSEngines.register("LocalAPI")
# class LocalAPI(BaseEngine):
    
#     def setup(self):
#         self.ttsmodel = TTS(model_name=MODEL_NAME, progress_bar=True)
#         return super().setup()
#     async def run(self, input: TextMessage, **kwargs) -> Optional[TextMessage]:
#         try: 
#             voice = self.cfg.PER
#             if 'voice' in kwargs and kwargs['voice']:
#                 voice = kwargs['voice']
#             communicate = self.ttsmodel.tts(input) 
#             data = b''
#             async for message in communicate.stream():
#                 if message["type"] == "audio":
#                     data += message["data"]
#             # mp3 -> wav
#             data = mp3ToWav(data)
#             message = AudioMessage(
#                 data=data, 
#                 desc=input.data,
#                 format=AudioFormatType.WAV,
#                 sampleRate=16000,
#                 sampleWidth=2,
#             )
#             return message
#         except Exception as e:
#             logger.error(f"[TTS] Engine run failed: {e}", exc_info=True)
#             return None