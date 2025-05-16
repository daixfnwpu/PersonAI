# -*- coding: utf-8 -*-
'''
@File    :   localChatGPT2Agnet.py
@Author  :   daixfnwpu 
'''

# import asyncio
# import torch
# from ..builder import AGENTS
# from ..agentBase import BaseAgent
# from typing import Any, AsyncGenerator, Dict, List, Optional, Union
# from aiServer.utils import logger
# from aiServer.utils import AudioMessage, TextMessage
# from aiServer.engine.engineBase import BaseEngine
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from fastapi.responses import JSONResponse

# __all__ = ["LocalChatGPT2Agnet"]


# @AGENTS.register("LocalChatGPT2Agnet")
# class LocalChatGPT2Agnet(BaseAgent):

#     def checkKeys(self) -> List[str]:
#         return []
#     def setup(self):
#         try:
#         # Load tokenizer and model
#             logger.debug(f"Gpt2API setup")
#             self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
#             self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
#             #self.tokenizer.pad_token = self.tokenizer.eos_token  # ✅ 解决 padding 问题
#             # Set to eval mode (important for inference)
#             self.model.eval()
#             logger.debug(f"torch cuda is {torch.cuda.is_available()}")
#             # Optional: move to GPU if available
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.model.to(self.device)
#         except Exception as e:
#             raise RuntimeError(f"[LLM] Engine  failed: {e}") 
#         return super().setup()
#     async def run(
#         self, 
#         input: Union[TextMessage, AudioMessage], 
#         streaming: False,
#         **kwargs
#     ):##-> AsyncGenerator[TextMessage,None]:
#         try: 
#             if isinstance(input, AudioMessage):
#                 raise RuntimeError("LocalChatGPT2Agnet does not support AudioMessage input")
#             result = await asyncio.to_thread(self.post, input.data)
#             logger.debug(f"result is {result}") 
#             yield result
#         except Exception as e:
#             logger.error(f"[AGENT] Engine run failed: {e}", exc_info=True)
#             yield str(e)  # ✅ 确保返回可序列化内容
#     def post(self,prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
#         input_ids = inputs["input_ids"]
#         attention_mask = inputs["attention_mask"]
#         # Generate
#         outputs = self.model.generate(
#             input_ids=input_ids,
#             max_length=60,
#             do_sample=True,        # random sampling
#             top_k=50,              # limits token choices to top 50
#             top_p=0.95,            # nucleus sampling
#             temperature=0.8,       # randomness
#             num_return_sequences=1,
#             attention_mask=attention_mask,  # 明确传入 attention_mask
#             pad_token_id=self.tokenizer.eos_token_id  # 显式指定 pad_token_id
#         )

#         # Decode and print
#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         logger.debug(f"generated_text is {generated_text}")
#         return generated_text[len(generated_text):]