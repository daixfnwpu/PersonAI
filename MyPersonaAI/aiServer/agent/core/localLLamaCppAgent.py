# -*- coding: utf-8 -*-
'''
@File    :   localLLamaCppAgent.py
@Author  :   daixfnwpu 
'''

import asyncio
from llama_cpp import Llama
import torch
from ..builder import AGENTS
from ..agentBase import BaseAgent
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
from aiServer.utils import logger
from aiServer.utils import AudioMessage, TextMessage
from aiServer.engine.engineBase import BaseEngine
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fastapi.responses import JSONResponse

__all__ = ["localLLamaCppAgent"]


@AGENTS.register("localLLamaCppAgent")
class localLLamaCppAgent(BaseAgent):

    def checkKeys(self) -> List[str]:
        return []
    def setup(self):
        try:
        # Load tokenizer and model
            logger.debug(f"LLAMACPP Agent  setup")
            self.llm = Llama(   model_path="models/llama-3.2-1b-instruct-q8_0.gguf",
                                chat_format="llama-3",n_ctx=4092,
                                n_threads=8,      # Adjust based on your CPU cores
                                n_gpu_layers= 16,   # Explicitly disable GPU
                                stream=True,
                                verbose=True)
        except Exception as e:
            raise RuntimeError(f"[LLM] Engine  failed: {e}") 
        return super().setup()
    async def run(
        self, 
        input: Union[TextMessage, AudioMessage], 
        streaming: False,
        **kwargs
    ):##-> AsyncGenerator[TextMessage,None]:
        try: 
            if isinstance(input, AudioMessage):
                raise RuntimeError("localLLamaCppAgent does not support AudioMessage input")
            result = await asyncio.to_thread(self.post, input.data)
            logger.debug(f"result is {result}") 
            yield result
        except Exception as e:
            logger.error(f"[AGENT] Engine run failed: {e}", exc_info=True)
            yield str(e)  # ✅ 确保返回可序列化内容
    def post(self,prompt):
        # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # logger.debug(f"the prompt is {prompt}")
        # input_ids = inputs["input_ids"]
        # attention_mask = inputs["attention_mask"]
       
        messages = []
        messages.append({"role": "user", "content": prompt})
        response_stream = self.llm.create_chat_completion(messages, stream=True)

        reply = ""
        for chunk in response_stream:
            delta = chunk["choices"][0]["delta"]
            content = delta.get("content", "")
            logger.debug(content, end="", flush=True)
            reply += content
        # Decode and print
       # generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
       # logger.debug(f"generated_text is {generated_text}")
        return reply
        #return reply[len(reply)]