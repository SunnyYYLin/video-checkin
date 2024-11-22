from dataclasses import dataclass
import os

@dataclass
class Config:
    voice_source: str = 'zh-CN-XiaoxiaoNeural'
    interval: float = 1.5
    names_file: str = os.path.join("data", "names.txt")