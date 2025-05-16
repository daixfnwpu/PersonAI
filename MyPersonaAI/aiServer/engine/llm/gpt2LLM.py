# -*- coding: utf-8 -*-
'''
@File    :   gpt2LLM.py
@Author  :   daixfnwpu 
'''

# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# import torch
# from ..builder import LLMEngines
# from ..engineBase import BaseEngine
# import json
# from typing import List, Optional
# from aiServer.utils import httpxAsyncClient
# from aiServer.utils import TextMessage
# from aiServer.utils import logger
# import asyncio

# __all__ = ["Gpt2API"]

# @LLMEngines.register("Gpt2API")
# class Gpt2API(BaseEngine): 
#     def checkKeys(self) -> List[str]:
#         return ["SK", "MODEL", "LLM_URL"]
        
#     def setup(self):
#         try:
#         # Load tokenizer and model
#             logger.debug(f"Gpt2API setup")
#             self.tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
#             self.model = GPT2LMHeadModel.from_pretrained("distilgpt2")
#             self.tokenizer.pad_token = self.tokenizer.eos_token  # ✅ 解决 padding 问题
#             # Set to eval mode (important for inference)
#             self.model.eval()
#             logger.debug(f"torch cuda is {torch.cuda.is_available()}")
#             # Optional: move to GPU if available
#             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#             self.model.to(self.device)
#         except Exception as e:
#             raise RuntimeError(f"[LLM] Engine  failed: {e}") 
#         return super().setup()
#     async def run(self, input: TextMessage, **kwargs) -> Optional[TextMessage]:
#         try:
#             logger.debug(f"[LLM] Engine run ,the input is :{input.data}")
#             result = await asyncio.to_thread(self.post, input.data) 
#             message = TextMessage(data=result)
#             return message
#         except Exception as e:
#             logger.error(f"[LLM] Engine run failed: {e}", exc_info=True)
#             return None
        
#     def post(self,prompt):
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

#         # Generate
#         outputs = self.model.generate(
#             inputs["input_ids"],
#             max_length=60,
#             do_sample=True,        # random sampling
#             top_k=50,              # limits token choices to top 50
#             top_p=0.95,            # nucleus sampling
#             temperature=0.8,       # randomness
#             num_return_sequences=1
#         )

#         # Decode and print
#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return generated_text[len(prompt):]
      




    