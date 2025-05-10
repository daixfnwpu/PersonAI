# MyPersonaAI

This project aims to create a personalized AI model based on LoRA (Low-Rank Adaptation) technique.
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes


该工程主要实现数字：数字拟身，忆我如初、思镜同在、心镜随行；
训练
    audio or text process:
        ：转换音频为 wav 格式 (16kHz mono) (已经完成 audio_pipe.py)
        ：Voice activity detection  (已经完成 audio_pipe.py)
        ：OverlappedSpeechDetection (已经完成 audio_pipe.py)
        ：使用 pyannote 进行说话人分离  (已经完成 audio_pipe.py)
        ：Whisper 转录文字  (已经完成 audio_pipe.py)
        ：将说话人时间戳匹配文字段  (已经完成 audio_pipe.py)
    
    train: 采用lora技术；
        t5，来分析我的情感及归纳总结和分析我的文章；(已经完成： train_npl_lora.py)
        tts:用OpenAI TTS 来完成文字转语音 (已经完成： tts_pipe.py),正在完成语音的特殊化，采用lora来实现语音的特殊化。(正在实现中ing,还没有找到可以用lora来fine tine的模型，有可能无法利用lora，只能用原始的增量训练)
        gpt： 训练我的语言模型；(已经完成： train_gpt_lora.py)
    

    

    对话的接口：
        1、webUI；
        2、采用音频；EchoMimic https://github.com/antgroup/echomimic；LivePortrait ; SadTalker
        3、生成视频：EchoMimic https://github.com/antgroup/echomimic； LivePortrait(ditto-talkinghead) ; SadTalker; live2d ; (用 Whisper、Silero-VAD、Wav2Lip 这类模型实时识别音素+口型);
        4、live2d来训练我的动作和嘴型；(采用live2d移植，live2d已经跑通)

    rag：来存储我的数据量，和知识点；

    inference： 使用t5和gpt来实现对话；
