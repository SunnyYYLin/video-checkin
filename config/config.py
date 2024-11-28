from dataclasses import dataclass
import os

@dataclass
class Config:
    voice_source: str = 'zh-CN-XiaoxiaoNeural'
    interval: float = 1.5
    names_file: str = os.path.join("data", "names.txt")
    face_threshold: float = 0.7
    voice_threshold: float = 0.7
    face_threshold: float = 0.4
    face_count_threshold: float = 12
    face_width_threshold: float  = 20
    face_height_threshold: float =20
    device: str ='cuda'