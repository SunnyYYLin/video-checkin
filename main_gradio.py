import gradio as gr
import time
import numpy as np
import torch
import moviepy as mp
import os
import base64
import edge_tts
from typing import List
from pydub import AudioSegment
from io import BytesIO
import asyncio
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
from voice_id import VoiceID, call_roll, call_name 
from face_id import FaceID  
from database import Database
from config import Config

VOICE_SOURCE = 'zh-CN-XiaoxiaoNeural'

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
    global name_list
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
            # print(video_file)
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

async def async_call_name(name: str, voice: str= VOICE_SOURCE) -> AudioSegment:
    # 使用 edge_tts 异步生成音频数据
    communicator = edge_tts.Communicate(name, voice)
    audio_data = BytesIO()
    async for chunk in communicator.stream():
        if chunk.get("type") == "audio": # 只处理音频数据
            audio_data.write(chunk["data"]) # 写入音频数据
    audio_data.seek(0)

    # 使用 pydub 处理音频数据
    audio_segment = AudioSegment.from_file(audio_data, format="mp3")
    # 将 AudioSegment 导出为字节流
    audio_bytes_io = BytesIO()
    audio_segment.export(audio_bytes_io, format="mp3")
    audio_bytes = audio_bytes_io.getvalue()
    # # 转换为字节流
    # output = BytesIO()
    # audio_segment.export(output, format="mp3")
    # output.seek(0)
    # # 将音频数据编码为 base64 字符串
    # audio_base64 = base64.b64encode(output.read()).decode('utf-8')

    # # 生成 HTML 音频标签，包含自动播放属性
    # audio_html = f'<audio controls autoplay><source src="data:audio/mp3;base64,{audio_base64}" type="audio/mpeg"></audio>'
    # return audio_html  
    return audio_bytes

async def handle_stream(mode, input_data):
    global PROCESS_POOL
    global arrived
    global name_list
    global sign
    global num
    global vad_tasks
    global voice_id_tasks
    global database_tasks
    global face_id_tasks
    global next_call
    match mode:
        case "aud_stream":
            voice_id.add_chunk((input_data[0],input_data[1].T))
            # 提交任务到进程池
            future = PROCESS_POOL.apply_async(voice_id.is_round_end)
            vad_tasks.append(future)
            for task in vad_tasks:
                if task.ready() and task.get():

                    num+=1
                    if num<len(name_list):
                        this_call = next_call
                        next_call = await async_call_name(name_list[num])

                    future_voice_id = PROCESS_POOL.apply_async(voice_id.extract_round_features)
                    voice_id_tasks.append(future_voice_id)
                    vad_tasks.remove(task)
                    break

            for task in voice_id_tasks:
                if task.ready():
                    audio_feature = task.get()
                    future_database = PROCESS_POOL.apply_async(database.recognize_voice, args=(audio_feature,))
                    database_tasks.append(future_database)
                    voice_id_tasks.remove(task)
                    break

        case "img_stream":
            pil_image = Image.fromarray(input_data)
            future_face_id = PROCESS_POOL.apply_async(face_id.extract_features, args=([pil_image],))
            face_id_tasks.append(future_face_id)
            for task in face_id_tasks:
                if task.ready():
                    face_features = task.get()
                    future_database = PROCESS_POOL.apply_async(database.recognize_faces, args=(face_features,))
                    database_tasks.append(future_database)
                    face_id_tasks.remove(task)
                    break

        case "start_check":
            this_call = await async_call_name(name_list[num])
            next_call = await async_call_name(name_list[num+1])
            return this_call
        
        case _:
            raise ValueError("Invalid mode! Please provide a valid mode.")

    for task in database_tasks:
        if task.ready():
            if isinstance(task.get(), str): 
                arrived.add(task.get())

            elif isinstance(task.get(), list):
                arrived.update(task.get())

            database_tasks.remove(task)
            break
    
    return f"arrived: {arrived - {None}}\n", this_call


def audio_done(txt):
    if txt == "done":
        return True
    elif txt == "True":
        return False

if __name__=="__main__": 
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

    arrived: set[str] = set()  # 记录已经到达
    num=0
    sign=True
    vad_tasks: List[asyncio.Task] = []
    voice_id_tasks: List[asyncio.Task] = []
    database_tasks: List[asyncio.Task] = []
    face_id_tasks: List[asyncio.Task] = []

    if not name_list:
        name_list=database.get_all_names()

    next_call = None
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
        #             audio_stream = gr.Audio(sources=["microphone"], type="numpy", label="请打开麦克风")
        #             image_stream = gr.Image(sources=["webcam"], type="numpy", label="请打开摄像头")
        #         stream_text_output = gr.Textbox(label="检测结果")
        #         stream_html_output = gr.HTML(label="点名")

        # sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
        # begin_btn.click(train, inputs=None, outputs=train_status ) 
        # check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])
        # audio_stream.stream(process, 
        #                     inputs=[gr.State("aud_stream"), audio_stream],
        #                     outputs=[stream_text_output,stream_html_output], time_limit=10, stream_every=0.3)
        # image_stream.stream(process, 
        #                     inputs=[gr.State("img_stream"), image_stream],
        #                     outputs=[stream_text_output,stream_html_output], time_limit=10, stream_every=0.5)

        with gr.Tab("视频检测：实时检测版"):
            with gr.Column():
                with gr.Row():
                    audio_stream = gr.Audio(sources=["microphone"], type="numpy", label="请打开麦克风")
                    image_stream = gr.Image(sources=["webcam"], type="numpy", label="请打开摄像头")
                begin_check_btn = gr.Button("开始检测")
                stream_output = gr.Textbox(label="检测结果")
                start_signal = gr.State("start_check")# 不变的开始播放状态
                play_done = gr.State("done")# 不变的播放完成状态
                play_state = gr.State("False")# 变化的调度状态
                play_audio = gr.Audio(label="播放学生名", streaming=True)

        sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
        begin_btn.click(train, inputs=None, outputs=train_status ) 
        check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])
        
        
        begin_check_btn.click(handle_stream, inputs=start_signal, outputs=play_audio)
        play_audio.stop(audio_done, inputs=play_done, outputs=play_state)
        #True代表已经播完音频可以开始流式了
        async def start_streams(play_state_value):
            if play_state_value == "True":
                audio_stream.stream(
                    handle_stream,
                    inputs=[gr.State("aud_stream"), audio_stream],
                    outputs=[stream_output, play_audio],
                    time_limit=3,
                    stream_every=0.3
                )
                image_stream.stream(
                    handle_stream,
                    inputs=[gr.State("img_stream"), image_stream],
                    outputs=[stream_output, play_audio],
                    time_limit=0.001,
                    stream_every=0.001
                )

        play_state.change(
            start_streams,
            inputs=play_state,
            outputs=None
        )

    # 全局进程池
    PROCESS_POOL = multiprocessing.Pool(processes=4)  # 设置进程池的大小，例如 2 个进程
    demo.launch()

