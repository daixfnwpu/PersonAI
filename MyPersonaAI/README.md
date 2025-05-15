# MyPersonaAI




This project aims to create a personalized AI model based on LoRA (Low-Rank Adaptation) technique.
pip uninstall torch torchvision torchaudio -y

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate peft bitsandbytes

The Web server and the AiServer use the : https://github.com/wan-h/awesome-digital-human-live2d ,ths;
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
        tts:用OpenAI TTS 来完成文字转语音 (已经完成： tts_pipe.py,并完成语音的clone: fast_tts_clone.py)
        gpt： 训练我的语言模型；(已经完成： train_gpt_lora.py)
    

    image gen ai :
        minigpt-4 : 用这个模型来解释图片的内容，包括视频的内容；（需要直接写脚本来完成推理);
            如果是要解析视频，需要将视频离散化,减少资源消耗; 

    对话的接口：
        1、webUI；
        2、采用音频；EchoMimic https://github.com/antgroup/echomimic；LivePortrait ; SadTalker
        3、生成视频：EchoMimic https://github.com/antgroup/echomimic； 
            (1),LivePortrait(ditto-talkinghead,https://github.com/antgroup/ditto-talkinghead?tab=readme-ov-file) ;
                https://colab.research.google.com/drive/1KxUMcXmonzuTTsWJ9msj43owLk7_1a78 
            (2),SadTalker; 
            (3),live2d ; (用 Whisper、Silero-VAD、Wav2Lip 这类模型实时识别音素+口型);
        4、live2d来训练我的动作和嘴型；(采用live2d移植，live2d已经跑通)

  *  rag：来存储我的数据量，和知识点；(还没有实现,model采用opt-150 model)

    inference： 使用t5和gpt来实现对话；(opt-150 model)

  *  UI:  将这里面的整个代码，转移到： live2d中的digitalhummer 中去； 



