import asyncio
from pathlib import Path
import edge_tts

VOICE_SOURCE = 'zh-CN-XiaoxiaoNeural'
RATE = 24000  # 默认采样率

async def async_call_name_with_file(name: str, voice: str= VOICE_SOURCE) -> Path:
    communicate = edge_tts.Communicate(name, voice)
    cache_dir = Path(f"audio_cache")
    if not cache_dir.exists():
        cache_dir.mkdir()
    await communicate.save(cache_dir/f"{name}.mp3")
    return cache_dir/f"{name}.mp3"

def call_name_with_file(name: str, voice: str=VOICE_SOURCE) -> Path:
    return asyncio.run(async_call_name_with_file(name, voice))

if __name__ == "__main__":
    names = ["张博为", "舒子骏", "葛奇昱", "林佳豪", "刘鑫宇"]
    
