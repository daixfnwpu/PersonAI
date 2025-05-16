# -*- coding: utf-8 -*-
'''
@File    :   localASR.py
@Author  :   daixfnwpu 
'''

import whisper
import tempfile
import wave

from ..builder import ASREngines
from ..engineBase import BaseEngine
from typing import Optional
from speech_recognition import AudioData
from aiServer.utils import AudioMessage, TextMessage
from aiServer.utils import logger

__all__ = ["LocalWisperAPI"]

@ASREngines.register("LocalWisperAPI")
class LocalWisperAPI(BaseEngine): 
    def setup(self):
        # Load the 'base' Whisper model
        self.model = whisper.load_model("base")

    async def run(self, input: AudioMessage, **kwargs) -> Optional[TextMessage]:
        try:
            # Convert AudioData to a WAV file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_wav:
                with wave.open(temp_wav.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(input.sampleWidth)
                    wf.setframerate(input.sampleRate)
                    wf.writeframes(input.data)
                
                # Transcribe using Whisper
                result = self.model.transcribe(temp_wav.name)
                message = TextMessage(data=result["text"])
                return message

        except Exception as e:
            logger.error(f"[ASR] Engine run failed: {e}")
            return None



