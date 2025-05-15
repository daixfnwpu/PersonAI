# -*- coding: utf-8 -*-
'''
@File    :   __init__.py
@Author  :   一力辉 
'''

from .difyAgent import DifyAgent
from .repeaterAgent import RepeaterAgent
from .fastgptAgent import FastgptAgent
from .openaiAgent import OpenaiAgent
from .localChatGPT2Agnet import LocalChatGPT2Agnet
from .agentFactory import AgentFactory

__all__ = ['AgentFactory']