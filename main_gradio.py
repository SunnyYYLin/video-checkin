import gradio as gr
import time
import numpy as np
import torch
import moviepy as mp
import os
from PIL import Image
from voice_id import VoiceID  
from face_id import FaceID  
from database import Database
from config import Config

img_list = []
aud_list = []
name_list = []

config = Config()
voice_id = VoiceID(config)

# 创建 FaceID 实例
face_config = {"device": "cuda"}  
face_id = FaceID(**face_config)

# 创建 Database 实例
# db_config = {}  
database = Database()

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
        result = handle_inputs(mode="sample", image=image, audio=audio, tag=text )
    except ValueError as e:
        return f"样本输入失败！发生错误：{str(e)}"
    
    if result is None:
        return "样本输入失败！请重新输入！"
    elif result.get("result", False):
        return "样本输入成功！"
    else:
        return "样本输入失败！请重新输入！"

def check_local(video):
    namedict = handle_inputs(mode="check", video_file=video)
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
        check_local_btn: gr.Button(visible=True),
    }
    
def train():
    yield "训练中......"
    handle_inputs(mode="train")
    yield "训练完成！" 

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
            train_status = gr.Textbox(label="训练状态")
            sample_btn = gr.Button("输入样本")
            begin_btn = gr.Button("开始训练") 
            
    with gr.Tab("视频检测：本地视频版"):
        with gr.Column():
            input_check = gr.Video(label="输入待检测的本地视频", format='mp4')
            check_local_btn = gr.Button("开始检测")
            wait_local = gr.Textbox(label="计算中", visible=False)
            output_check_local_true = gr.Textbox(label="检测结果：到教室的人", visible=False)
            output_check_local_false = gr.Textbox(label="检测结果：缺勤的人", visible=False)

    sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
    begin_btn.click(train, inputs=None, outputs=train_status ) 
    check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])

#主函数

def handle_inputs(mode: str, video_file:str =None, 
                  image: np.ndarray=None, 
                  audio: tuple[int, np.ndarray]=None, 
                  tag: str=None):
    
    match mode:
        case "sample":
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
            return {"result": True}

        case "train":
            num_epochs = 10

            print("正在训练脸部识别孪生网络模型...")
            database.train_face_siamese_model(num_epochs)
            print("语音识别模型训练完成！")

            print("正在训练声音识别孪生网络模型...")
            database.train_voice_siamese_model(num_epochs)
            print("声音识别模型训练完成！")    
            # print("正在备份可识别同学信息...")
            # database.save_feature_db(filename="feature_db.pt")
            # print("可识别同学信息备份完成！")

            #return ["Sample input successfully received"]
            return {"result": True}
        
        case "check":
            test_list = []
            face_features = []
            #获取音频序列和图像序列
            print(video_file)
            video = mp.VideoFileClip(video_file)
            rate = video.audio.fps
            audio = video.audio.to_soundarray(fps=rate)
            images = [Image.fromarray(img_array) for img_array in
                       video.iter_frames(fps=video.fps, dtype='uint8')]

            #提取图像、音频特征向量
            face_features = face_id.get_features_list(images)
            audio_features = voice_id.extract_clip_features((rate, audio.T))

            #识别人脸和声音
            text_list = database.recognize_faces(face_features)
            for audio_feature in audio_features:
                result = database.recognize_voice(audio_feature)
                # 将识别结果添加到文本列表
                if result not in text_list:
                    text_list.append(result)

            dict = {}
            print(test_list)
            # 检查识别结果是否在名单中
            for name in name_list:
                if name in text_list:
                    dict.update({name: True})
                else:
                    dict.update({name: False})

            return dict

        case _:
            raise ValueError("Invalid mode! Please provide a valid mode.")

demo.launch()
