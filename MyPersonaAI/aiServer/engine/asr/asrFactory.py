# -*- coding: utf-8 -*-
'''
@File    :   asrFactory.py
@Author  :   一力辉 
'''

from ..builder import ASREngines
from ..engineBase import BaseEngine
from typing import List
from yacs.config import CfgNode as CN
from aiServer.utils import logger

__all__ = ["ASRFactory"]

class ASRFactory():
    """
    Automatic Speech Recognition Factory
    """
    @staticmethod
    def create(config: CN) -> BaseEngine:
        logger.info(f"config Name is :{config.NAME}")
        if config.NAME in ASREngines.list():
            logger.info(f"[ASRFactory] Create engine: {config.NAME}")
            return ASREngines.get(config.NAME)(config)
        else:
            raise RuntimeError(f"[ASRFactory] Please check config, support ASR engine: {ASREngines.list()}")
    @staticmethod
    def list() -> List:
        return ASREngines.list()