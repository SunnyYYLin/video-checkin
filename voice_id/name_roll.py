import edge_tts
import asyncio
from pydub import AudioSegment
from io import BytesIO

VOICE_SOURCE = 'zh-CN-XiaoxiaoNeural'

async def async_call_name(name: str, voice: str= VOICE_SOURCE) -> AudioSegment:
    # 使用 edge_tts 异步生成音频数据
    communicator = edge_tts.Communicate(name, voice)
    audio_data = BytesIO()
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio": # 只处理音频数据
            audio_data.write(chunk["data"]) # 写入音频数据
    audio_data.seek(0)
    return AudioSegment.from_file(audio_data, format="mp3")

def call_name(name: str, voice: str=VOICE_SOURCE) -> AudioSegment:
    return asyncio.run(async_call_name(name, voice))

async def async_call_roll(names: list[str], interval: float=1.5, voice: str=VOICE_SOURCE) -> AudioSegment:
    tasks = [asyncio.to_thread(call_name, name, voice) for name in names]
    vocal_segs = await asyncio.gather(*tasks)
    silence = AudioSegment.silent(duration=interval * 1000)  # interval 是秒，转换为毫秒
    segments = [seg for vocal_seg in vocal_segs for seg in (vocal_seg, silence)]
    return sum(segments)

def call_roll(names: list[str], interval: float=1.5, voice: str=VOICE_SOURCE) -> AudioSegment:
    return asyncio.run(async_call_roll(names, interval, voice))

if __name__ == "__main__":
    names = ["张博为", "舒子骏", "葛奇昱", "林佳豪", "刘鑫宇"]
    roll_call_audio = call_roll(names)
    roll_call_audio.export("roll_call.mp3", format="mp3")
