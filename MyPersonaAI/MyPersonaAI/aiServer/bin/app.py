# -*- coding: utf-8 -*-
'''
@File    :   app.py
@Author  :   一力辉 
'''

import uvicorn
from aiServer.engine import EnginePool
from aiServer.agent import AgentPool
from aiServer.server import app
from aiServer.utils import config

__all__ = ["runServer"]
### need to understand the difference between enginepool and agents;
### why when run agent,then don't run the engine model?
def runServer():
    enginePool = EnginePool()
    enginePool.setup(config.SERVER.ENGINES)
    agentPool = AgentPool()
    agentPool.setup(config.SERVER.AGENTS)
    uvicorn.run(app, host=config.SERVER.IP, port=config.SERVER.PORT, log_level="info")