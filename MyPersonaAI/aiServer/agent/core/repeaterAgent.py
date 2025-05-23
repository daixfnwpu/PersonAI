# -*- coding: utf-8 -*-
'''
@File    :   repeaterAgnet.py
@Author  :   一力辉 
'''

from ..builder import AGENTS
from ..agentBase import BaseAgent
from typing import List, Optional, Union
from aiServer.utils import logger
from aiServer.utils import AudioMessage, TextMessage
from aiServer.engine.engineBase import BaseEngine

__all__ = ["Repeater"]


@AGENTS.register("RepeaterAgent")
class RepeaterAgent(BaseAgent):

    def checkKeys(self) -> List[str]:
        return []
    
    async def run(
        self, 
        input: Union[TextMessage, AudioMessage], 
        streaming: False,
        **kwargs
    ):
        try: 
            if isinstance(input, AudioMessage):
                raise RuntimeError("RepeaterAgent does not support AudioMessage input")
            logger.debug(f"RepeaterAgent input data is {input.data}")
            yield input.data
        except Exception as e:
            logger.error(f"[AGENT] Engine run failed: {e}", exc_info=True)
            yield ""