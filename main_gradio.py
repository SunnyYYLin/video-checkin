import gradio as gr
import time
import numpy as np
import torch
import moviepy as mp
import librosa
from PIL import Image
from voice_id import VoiceID  
from face_id import FaceID  
<<<<<<< HEAD
from database import Database
from config import Config
=======
from database import Database  
>>>>>>> 549ab913cd9e08b3e23dcd34562d79480a9dc8a1

img_list = []
aud_list = []
name_list = []


# def main(video=None, image=None, audio=None, tag=None):
#     if image is not None and audio is not None and isinstance(tag, str) and tag.strip():
#         img_list.append(image)
#         aud_list.append(audio)
#         name_list.append(tag)
#         print(f"Image: {len(img_list)}, Audio: {len(aud_list)}, Tag: {len(name_list)}")
#         time.sleep(5)
#         return {"result": True}
#     if video is not None:
#         time.sleep(5)
#         return {"name1": True, "name2": False, "name3": True}
#     else:
#         return None

def waiting_local():
    return {
        wait_local: gr.Textbox(label="检测中", visible=True)
    }
    
def sample(image, audio, text):
    if image is None or audio is None or not isinstance(text, str) or not text.strip():
        return "样本输入失败！请确保所有内容都已填写并重新输入！"
    
    try:
        result = main(image=image, audio=audio, tag=text)
    except ValueError as e:
        return f"样本输入失败！发生错误：{str(e)}"
    
    if result is None:
        return "样本输入失败！请重新输入！"
    elif result.get("result", False):
        return "样本输入成功！"
    else:
        return "样本输入失败！请重新输入！"

def check_local(video):
    namedict = main(video=video)
    if namedict is None: 
        return {
            wait_local: gr.Textbox(label="计算中", visible=False),
            output_check_local_true: gr.Textbox(value="无效输入，请重新输入！", label="无效输入", visible=True),
            output_check_local_false: gr.Textbox(value="", visible=False),
            check_local_btn: gr.Button(visible=True),
        }
    true_names = [name for name, value in namedict.items() if value]
    false_names = [name for name, value in namedict.items() if not value]
    return {
        wait_local: gr.Textbox(label="计算中", visible=False),
        output_check_local_true: gr.Textbox(value="\n".join(true_names), label="到教室的人", visible=True),
        output_check_local_false: gr.Textbox(value="\n".join(false_names), label="缺勤的人", visible=True),
        check_local_btn: gr.Button(visible=False),
    }

with gr.Blocks(fill_height=True) as demo:
    
    gr.Markdown(
        """
        <h1 style="text-align: center;">点到器</h1>
        <p style="text-align: center;">这是一个结合人脸识别与声纹识别的点到器,可以上传视频来检测到教室的人。</p>
        """
    )
    
    with gr.Tab("输入样本"):
        with gr.Column():
            with gr.Row(equal_height=True):
                img = gr.Image(label="请输入人脸")
                aud = gr.Audio(label="请输入人声")
            name = gr.Textbox(label="请输入人名")
            output_sample = gr.Textbox(label="运行结果")
            sample_btn = gr.Button("输入样本")
            
    with gr.Tab("视频检测：本地视频版"):
        with gr.Column():
            input_check = gr.Video(label="输入待检测的本地视频")
            check_local_btn = gr.Button("开始检测")
            wait_local = gr.Textbox(label="计算中", visible=False)
            output_check_local_true = gr.Textbox(label="检测结果：到教室的人", visible=False)
            output_check_local_false = gr.Textbox(label="检测结果：缺勤的人", visible=False)

    sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
    check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])

    
<<<<<<< HEAD

=======
>>>>>>> 549ab913cd9e08b3e23dcd34562d79480a9dc8a1


# main.py



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
    text_list = database.recognize_faces(face_features)
    
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
            result = database.recognize_voice(audio_feature)
            # 将识别结果添加到文本列表
            if result not in text_list:
                text_list.append(result)
    
    return text_list

#主函数
def main(video_file=None, image=None, audio=None, tag=None):
    global img_list, aud_list, name_list
    # 创建 VoiceID 实例
    # voice_config = {"param1": "value1", "param2": "value2"}  #参数配置,后续添加
    config = Config()
    voice_id = VoiceID(config)

    # 创建 FaceID 实例
    face_config = {"device": "cuda"}  
    face_id = FaceID(face_config)

    # 创建 Database 实例
    # db_config = {}  
    database = Database()

    # 从输入中获取视频文件并进行检测
    if video_file is not None:
        test_list = []

        #获取音频序列和图像序列
        audio_frames, image_frames = split_video(video_file)

        #识别
        test_list = recognize(image_frames, audio_frames, voice_id, face_id, database)
        
        dict = {}
        
        for name in test_list:
            if name in name_list:
                dict.update({name: True})
            else:
                dict.update({name: False})

        #return test_list
        return dict
    
    #读入样本数据
    elif image is not None and audio is not None and tag is not None:
        
        # img_list.append(image)
        # aud_list.append(audio)
        name_list.append(tag)
        #将NujmPy数组转换为PIL图像
        
        pil_image = Image.fromarray(image)
        # 提取人脸特征
        face_feature = face_id.extract_features(pil_image)

        # 提取声音特征
        audio_feature = voice_id.extract_label_features(audio)

        print("正在存储学生特征...")
        # 将样本的人脸特征、声音特征和标签存储到数据库
        database.store_feature(tag, face_feature, audio_feature)
        print("学生特征存储完成！")

        num_epochs = 10

        print("正在训练脸部识别孪生网络模型...")
        database.train_face_siamese_model(num_epochs)
        print("语音识别模型训练完成！")

        print("正在训练声音识别孪生网络模型...")
        database.train_voice_siamese_model(num_epochs)
        print("声音识别模型训练完成！")

        #return ["Sample input successfully received"]
        return {"result": True}
    
    else:
        return None

demo.launch()
