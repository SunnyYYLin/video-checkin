from dataclasses import dataclass

@dataclass
class Config:
    face_threshold: float = 0.7
    voice_threshold: float = 0.7
    face_threshold: float = 0.4
    face_count_threshold: float = 12
    face_width_threshold: float  = 20
    face_height_threshold: float =20
    device: str ='cuda'
    voice_round_threshold: float = 0.3
    voice_video_threshold: float = 0.2