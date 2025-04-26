# MyPersonaAI

This project aims to create a personalized AI model based on LoRA (Low-Rank Adaptation) technique.


该工程主要实现数字：忆我如初、思镜同在、心镜随行；
训练
    ：转换音频为 wav 格式 (16kHz mono)
    ：Voice activity detection
    ：OverlappedSpeechDetection
    ：使用 pyannote 进行说话人分离
    ：Whisper 转录文字
    ：将说话人时间戳匹配文字段
    
    train: 采用lora技术；
    t5，来分析我的情感及归纳总结和分析我的文章；

    gpt： 训练我的语言模型；

    rag：来存储我的数据量，和知识点；

    inference： 使用t5和gpt来实现对话；

    对话的接口：
        1、webUI；
        2、采用音频；
        3、生成视频：

