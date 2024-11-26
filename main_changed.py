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
    
#感觉这里得你来部署，传什么参数给谁，然后主函数添加判断即可
    
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
            print("正在备份可识别同学信息...")
            database.save_feature_db(filename="feature_db.pt")
            print("可识别同学信息备份完成！")

            #return ["Sample input successfully received"]
            return {"result": True}
        
        case "check":
            text_list = []
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
            if not name_list:
                name_list=database.get_all_names()

            # 检查识别结果是否在名单中
            for name in name_list:
                if name in text_list:
                    dict.update({name: True})
                else:
                    dict.update({name: False})

            return dict

        case _:
            raise ValueError("Invalid mode! Please provide a valid mode.")
        

def audio_done(txt):
    if txt == "done":
        return True
    elif txt == "True":
        return False

#这里将控制移到最后，方便与主函数进行交互，并添加了实时检测的逻辑

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
            
    # with gr.Tab("视频检测：实时检测版"):
    #     with gr.Column():
    #         with gr.Row():
    #             audio_stream = gr.Audio(sources=["microphone"], type="numpy", label="请打开摄像头")
    #             image_stream = gr.Image(sources=["webcam"], type="numpy", label="请打开麦克风")
    #         begin_check_btn = gr.Button("开始检测")
    #         stream_output = gr.Textbox(label="检测结果")
    #         play_signal = gr.State("start_check")# 不变的开始播放状态
    #         play_done = gr.State("done")# 不变的播放完成状态
    #         play_state = gr.State("empty")# 变化的调度状态
    #         play_audio = gr.Audio(label="播放学生名")

    # sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
    # begin_btn.click(train, inputs=None, outputs=train_status ) 
    # check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])
    
    
    # begin_check_btn.click(handle_inputs, inputs=play_signal, outputs=play_audio)
    # play_audio.stop(handle_inputs, inputs=play_done, outputs=play_state)
    # if play_state.value == "begin_check":
    #     audio_stream.stream(handle_inputs, 
    #                     inputs=[gr.State("aud_stream"), audio_stream],
    #                     outputs=[stream_output, play_state], time_limit=3, stream_every=0.3)
    #     image_stream.stream(handle_inputs, 
    #                     inputs=[gr.State("img_stream"), image_stream],
    #                     outputs=[stream_output, play_state], time_limit=0.001, stream_every=0.001)
    # play_state.change(handle_inputs, inputs=[play_signal, play_state], outputs=play_audio)#这里需要主函数判断一下play_state是否为"check_done"，如果是，则返回音频，否则返回空值，但不知道空值会发生什么
    
    
    with gr.Tab("视频检测：实时检测版"):
        with gr.Column():
            with gr.Row():
                audio_stream = gr.Audio(sources=["microphone"], type="numpy", label="请打开摄像头")
                image_stream = gr.Image(sources=["webcam"], type="numpy", label="请打开麦克风")
            begin_check_btn = gr.Button("开始检测")
            stream_output = gr.Textbox(label="检测结果")
            start_signal = gr.State("start_check")# 不变的开始播放状态
            play_done = gr.State("done")# 不变的播放完成状态
            play_state = gr.State("False")# 变化的调度状态
            play_audio = gr.Audio(label="播放学生名")

    sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
    begin_btn.click(train, inputs=None, outputs=train_status ) 
    check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])
    
    
    begin_check_btn.click(handle_inputs, inputs=start_signal, outputs=play_audio)
    play_audio.stop(handle_inputs, inputs=play_done, outputs=None)
    play_audio.stop(audio_done, inputs=play_done, outputs=play_state)
    play_audio.play(audio_done, inputs=play_state, outputs=play_state)#这里的假定与下相同
    #True代表已经播完音频可以开始流式了
    if play_state.value == "True":
        audio_stream.stream(handle_inputs, 
                        inputs=[gr.State("aud_stream"), audio_stream],
                        outputs=[stream_output, play_audio], time_limit=3, stream_every=0.3)  #设想的是如果play_audio为空则维持流式，否则播放音频
        image_stream.stream(handle_inputs, 
                        inputs=[gr.State("img_stream"), image_stream],
                        outputs=[stream_output, play_audio], time_limit=0.001, stream_every=0.001)
    #play_state.change(handle_inputs, inputs=[play_signal, play_state], outputs=play_audio)#这里需要主函数判断一下play_state是否为"check_done"，如果是，则返回音频，否则返回空值，但不知道空值会发生什么
    
    
    
    
#首先点击按钮，我将play_signal传给主函数，并接受音频
#当音频播放完之后，我将播放完的信号play_done传给主函数，主函数返回我play_state="begin_check"
#我检测到begin_check后将音频和视频流传给主函数进行人名检测，返回stream_output与新的play_state="check_done"
#我检测play_state的变化，一旦改变，就传递播放信号play_signal给主函数，主函数传给我音频




#主函数先生成一段音频，当我按下按钮，即检测btn.click, 我将一个状态“开始播放”主函数，主函数返回给我音频输出
#我检测音频是否播完，audio.stop(),我将状态“播放完了”传给主函数，主函数告诉我状态“开始检测”——这里想让我造一个函数，然后我启动流式
#我打开流式音频主函数操作，完了给我一个音频，在来一个“开始播放”状态，我播放音频
#当主函数检测到人名，返回给我人名字典与一个状态“检测完毕”，我用if语句，输出人名，并将该状态改变

demo.launch()