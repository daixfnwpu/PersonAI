# -*- coding: utf-8 -*-
'''
@File    :   api.py
@Author  :   一力辉 
'''

import asyncio
from .commonApi import router as commonRouter
from .asrApi import router as asrRouter
from .agentApi import router as agentRouter
from .llmApi import router as llmRouter
from .ttsApi import router as ttsRouter
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

__all__ = ["app"]
# Define your background job
background_task = None

async def background_worker():
    while True:
        print("Background task is running...")
        await asyncio.sleep(5)

# Use the new lifespan context
@asynccontextmanager
async def lifespan(app: FastAPI):
    global background_task
    print("Lifespan startup")
    background_task = asyncio.create_task(background_worker())
    yield  # <-- Your app runs between startup and shutdown here
    print("Lifespan shutdown")
    background_task.cancel()
    try:
        await background_task
    except asyncio.CancelledError:
        print("Background task cancelled")

app = FastAPI(
    title="Awesome Digital Human", 
    description="This is a cool set of apis for Awesome Digital Human",
    version="0.0.1",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 路由
app.include_router(commonRouter, prefix="/adh/common", tags=["COMMON"])
app.include_router(asrRouter, prefix="/adh/asr", tags=["ASR"])
app.include_router(llmRouter, prefix="/adh/llm", tags=["LLM"]) #### 目前没有用上。
app.include_router(ttsRouter, prefix="/adh/tts", tags=["TTS"])
app.include_router(agentRouter, prefix="/adh/agent", tags=["AGENT"])