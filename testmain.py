import cv2
import moviepy as mp
from pydub import AudioSegment
from PIL import Image
import os
from io import BytesIO

# Step 1: Extract frames from the video
def extract_frames(video_file):
    # 将文件对象保存到临时文件
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    cap = cv2.VideoCapture(temp_video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frames.append(pil_image)
    cap.release()
    cv2.destroyAllWindows()
    os.remove(temp_video_path)
    return frames

# Step 2: Extract and split audio from the video
def extract_and_split_audio(video_file, segment_length_ms):
    # 将文件对象保存到临时文件
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())
    
    # 提取音频
    video = mp.VideoFileClip(temp_video_path)
    audio_path = "temp_audio.wav"  # 临时文件
    video.audio.write_audiofile(audio_path)
    
    # 分割音频文件
    audio = AudioSegment.from_file(audio_path)
    audio_length_ms = len(audio)
    audio_frames = [audio[i:i + segment_length_ms] for i in range(0, audio_length_ms, segment_length_ms)]
    
    video.close()  # 确保关闭 VideoFileClip 对象
    os.remove(temp_video_path)
    os.remove(audio_path)
    return audio_frames

# Usage
with open("test.mp4", "rb") as video_file:
    segment_length_ms = 1000  # 每段1秒
    frames = extract_frames(video_file)
    video_file.seek(0)  # 重置文件指针
    audio_frames = extract_and_split_audio(video_file, segment_length_ms)
    print(type(frames), type(audio_frames))