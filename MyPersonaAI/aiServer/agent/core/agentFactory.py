# -*- coding: utf-8 -*-
'''
@File    :   agentFactory.py
@Author  :   一力辉 
'''

from ..builder import AGENTS
from ..agentBase import BaseAgent
from typing import List
from yacs.config import CfgNode as CN
from aiServer.utils import logger

class AgentFactory():
    """
    Text to Speech Factory
    """
    @staticmethod
    def create(config: CN) -> BaseAgent:
        if config.NAME in AGENTS.list():
            logger.info(f"[AgentFactory] Create instance: {config.NAME}")
            return AGENTS.get(config.NAME)(config)
        else:
            raise RuntimeError(f"[AgentFactory] Please check config {config.NAME}, support AGENT engine: {AGENTS.list()}")
    @staticmethod
    def list() -> List:
        return AGENTS.list()