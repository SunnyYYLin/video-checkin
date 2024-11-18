# main.py

import gradio as gr
import numpy as np
import torch
import moviepy.editor as mp
import librosa
from PIL import Image
from audio_recognition import VoiceID  
from face_recognition import FaceID  
from database_module import Database  

# 读取视频文件并将其分解为图像和音频序列
def split_video(video_file):
    # 读视频
    video = mp.VideoFileClip(video_file)
    # 获取图像序列
    frames = []
    for frame in video.iter_frames():
        frames.append(frame)
    # 获取音频序列
    audio = video.audio.to_soundarray()
    return audio,frames

# 处理音频数据
def process_audio_data(audio_frames, voice_id):
    audio_feature = None
    for audio_frame in audio_frames:
        # 将每帧声音流转换为 torch.Tensor
        audio_tensor = torch.tensor(audio_frame)
        voice_id.add_frames([audio_tensor])

        # 判断一轮问答是否结束
        if voice_id.is_round_end():
            audio_feature = voice_id.extract_round_features()
            # 清除已处理的音频帧
            del audio_frames[:audio_frames.index(audio_frame) + 1]
            break  # 假设我们在一轮问答结束后停止处理
    return audio_feature

# 处理图像数据
def process_image_data(image_frames, face_id, face_features):
    # 设置提取人脸特征的间隔帧数
    frame_interval = 5  # 每隔5帧提取一次人脸特征
    # 初始化计数器
    frame_count = 0

    for image_frame in image_frames:
    # 每隔 frame_interval 帧提取一次人脸特征
        if frame_count % frame_interval == 0:
            # 提取人脸特征
            face_feature = face_id.extract_features(image_frame)
            # 检测并添加新的人脸特征
            face_id.is_new_features([face_feature], face_features)
    
        # 增加计数器
        frame_count += 1

# 识别
def recognize(image_frames, audio_frames,voice_id, face_id, database):
    # 存储特征向量
    face_features = []
    text_list = []

    # 先进行人脸识别
    process_image_data(image_frames, face_id, face_features)
    
    while True:
        # 检查输入数据是否为空
        if not audio_frames:
            break
    
        audio_feature = None
        for audio_frame in audio_frames:
            # 将每帧声音流转换为 torch.Tensor
            audio_tensor = torch.tensor(audio_frame)
            voice_id.add_frames([audio_tensor])

            # 判断一轮问答是否结束
            if voice_id.is_round_end():
                audio_feature = voice_id.extract_round_features()
                # 清除已处理的音频帧
                del audio_frames[:audio_frames.index(audio_frame) + 1]
                break  # 假设我们在一轮问答结束后停止处理    

        # 将特征向量传输给识别系统
        if audio_feature is not None:
            result = database.compare_feature_vectors(face_features, audio_feature)
            # 将识别结果添加到文本列表
            text_list.append(result)
    
    return text_list

#主函数
def main(video_file, image, audio, tag):
    # 创建 VoiceID 实例
    voice_config = {"param1": "value1", "param2": "value2"}  #参数配置,后续添加
    voice_id = VoiceID(voice_config)

    # 创建 FaceID 实例
    face_config = {"param1": "value1", "param2": "value2"}  
    face_id = FaceID(face_config)

    # 创建 Database 实例
    db_config = {"param1": "value1", "param2": "value2"}  
    database = Database(db_config)

    # 从输入中获取视频文件并进行检测
    if video_file is not None:
        test_list = []

        #获取音频序列和图像序列
        audio_frames, image_frames = split_video(video_file)

        #识别
        test_list = recognize(image_frames, audio_frames, voice_id, face_id, database)

        return test_list
    
    #读入样本数据
    elif image is not None and audio is not None and tag is not None:
        #将NujmPy数组转换为PIL图像
        pil_image = Image.fromarray(image)
        # 提取人脸特征
        face_feature = face_id.extract_features(pil_image)

        # 将声音数据转换为音频序列
        # 以及采样率的元组 (audio_data, sample_rate)
        audio_data, sample_rate = audio
        # 将音频数据转换为音频序列
        # 这里假设每个音频帧的长度为 2048
        frame_length = 2048
        hop_length = 512
        audio_frames = librosa.util.frame(audio_data, frame_length=frame_length, hop_length=hop_length)
        for audio_frame in audio_frames:
            # 将每帧声音流转换为 torch.Tensor
            audio_tensor = torch.tensor(audio_frame)
            voice_id.add_frames([audio_tensor])
        audio_feature = voice_id.extract_round_features()

        # 将样本的人脸特征、声音特征和标签存储到数据库
        database.store_combined_feature(tag, face_feature, audio_feature)

        #训练KNN模型
        database.train_knn()

        return ["Sample input successfully received"]
