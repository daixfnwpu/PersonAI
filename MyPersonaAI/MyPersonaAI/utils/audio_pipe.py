import torch
import whisper
from pyannote.audio import Pipeline
import ffmpeg
import os
import torch
print(torch.version.cuda)             # 显示 CUDA 版本（如果支持）
print(torch.cuda.is_available())      # True 表示可以用 GPU

ffmpeg_path = "D:/Program Files (x86)/ffmpeg/bin/ffmpeg.exe"  # 改成你自己的路径
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["SPEECHBRAIN_LOCAL_FILE_STRATEGY"] = "copy"  # SpeechBrain 改为复制文件
# Step 1: 转换音频为 wav 格式 (16kHz mono)
def convert_audio(input_path, output_path="converted.wav"):
    ffmpeg.input(input_path).output(
        output_path, ar='16000', ac=1, format='wav'
    ).overwrite_output().run()
    return output_path

# Step 2: 使用 pyannote 进行说话人分离
def diarize_audio(wav_path, hf_token):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
      #  cache_dir="./models/pyannote"
    )
    pipeline.to(torch.device("cuda"))
    diarization = pipeline(wav_path)
    return diarization

# Step 3: Whisper 转录文字
def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # or "medium" / "large"
    result = model.transcribe(audio_path)
    return result["segments"]

# Step 4: 将说话人时间戳匹配文字段
def align_speaker_segments(diarization, whisper_segments):
    output = []
    for segment in whisper_segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"]

        speaker = "Unknown"
        for turn in diarization.itertracks(yield_label=True):
            turn_start, turn_end = turn[0].start, turn[0].end
            if turn_start <= start <= turn_end:
                speaker = turn[2]
                break

        output.append(f"{speaker}: {text.strip()}")
    return output

# 主流程
from dotenv import load_dotenv
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
original_audio = "data/testaudio.mp3"

from pyannote.audio import Model
modelpyannote = Model.from_pretrained(
  "pyannote/segmentation-3.0", 
  use_auth_token=hf_token)

### Voice activity detection
from pyannote.audio.pipelines import VoiceActivityDetection
pipeline = VoiceActivityDetection(segmentation=modelpyannote)
HYPER_PARAMETERS = {
  # remove speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
vad = pipeline(original_audio)


from pyannote.audio.pipelines import OverlappedSpeechDetection
pipeline = OverlappedSpeechDetection(segmentation=modelpyannote)
HYPER_PARAMETERS = {
  # remove overlapped speech regions shorter than that many seconds.
  "min_duration_on": 0.0,
  # fill non-overlapped speech regions shorter than that many seconds.
  "min_duration_off": 0.0
}
pipeline.instantiate(HYPER_PARAMETERS)
osd = pipeline(original_audio)

# 查看结果
print("VAD:", vad)
print("OSD:", osd)

converted = convert_audio(original_audio)
diarization_result = diarize_audio(converted, hf_token)
whisper_result = transcribe_audio(converted)
final_output = align_speaker_segments(diarization_result, whisper_result)

for line in final_output:
    print(line)
