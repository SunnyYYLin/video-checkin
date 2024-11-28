import gradio as gr
import numpy as np
import moviepy as mp
from PIL import Image
from voice_id import VoiceID
from face_id import FaceID  
from database import Database
from config import Config

VOICE_SOURCE = 'zh-CN-XiaoxiaoNeural'

def waiting_local():
    return {
        wait_local: gr.Textbox(label="检测中", visible=True)
    }
    
def sample(image, audio, text):
    if (image is None and audio is None) or not isinstance(text, str) or not text.strip():
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
    
#主函数
def handle_inputs(mode: str, video_file:str =None, 
                  image: np.ndarray=None, 
                  audio: tuple[int, np.ndarray]=None, 
                  tag: str=None):
    match mode:
        case "sample":
            #将NujmPy数组转换为PIL图像
            pil_image = Image.fromarray(image)
            # 提取人脸特征
            face_feature = face_id.extract_features(pil_image, mode='enter').squeeze(0) # (dim_face, )
            # 提取声音特征
            audio_feature = voice_id.extract_label_features(audio) # (dim_audio, )

            print("正在存储学生特征...")
            database.add(tag, face_feature, audio_feature)
            database.save()
            print("学生特征存储完成！")
            return {"result": True}

        case "train":
            database.train_both(num_epochs=10, batch_size=8)
            database.save()
            return {"result": True}
        
        case "check":
            arrived_list = []
            face_features = []
            #获取音频序列和图像序列
            # print(video_file)
            video = mp.VideoFileClip(video_file)
            rate = video.audio.fps
            audio = video.audio.to_soundarray(fps=rate)
            images = [Image.fromarray(img_array) for img_array in
                       video.iter_frames(fps=video.fps, dtype='uint8')]
            images = images[::30]

            #提取图像、音频特征向量
            print("正在提取人脸特征...")
            face_features = face_id.get_features(images) # (batch_size, dim_face)
            print("人脸特征提取完成！")
            print("正在提取声音特征...")
            audio_features = voice_id.extract_clip_features((rate, audio.T)) # (batch_size, dim_audio)
            print("声音特征提取完成！")

            #识别人脸和声音
            print("正在识别人脸...")
            arrived_list = database.recognize_faces(face_features)
            print("人脸识别完成！")
            print("正在识别声音...")
            arrived_list += database.recognize_voices(audio_features)
            print("声音识别完成！")
            arrived_list = list(set(arrived_list))
            return {name: (name in arrived_list) for name in database.name_list}

        case _:
            raise ValueError("Invalid mode! Please provide a valid mode.")

async def handle_stream(mode, input_data=None):
    global arrived
    global stream_name_list
    global sign
    global num
    global vad_tasks
    global voice_id_tasks
    global database_tasks
    global face_id_tasks
    global has_started
    global pool
    global next_call

    this_call = None
    audio_result = None
    face_results = []
    
    match mode:
        case "aud_stream":
            audio = (input_data[0], input_data[1].T.astype(np.float32)/2**15)
            voice_id.add_chunk(audio)
            if voice_id.is_round_end():
                print("正在提取声纹特征...")
                audio_feature = voice_id.extract_round_features() # (dim_audio, )
                print("声纹特征提取完成！")
                print(audio_feature)
                audio_result = database.recognize_voice(audio_feature)

        case "img_stream":
            pil_image = Image.fromarray(input_data)
            print("正在提取人脸特征...")
            face_features = face_id.extract_features(pil_image, mode='checkin') # (batch_size, dim_face)
            print(f"人脸特征提取完成: {len(face_features)}")
            face_results = database.recognize_faces(face_features)
        
        case _:
            raise ValueError("Invalid mode! Please provide a valid mode.")

    arrived.add(audio_result)
    arrived.update(face_results)
    
    return f"arrived: {arrived - {None}}\n"
    
def audio_playback_finished():
    global play_state
    play_state.value = 'idle'

if __name__=="__main__": 
    img_list = []
    aud_list = []
    name_list = []
    stream_name_list = []

    config = Config()
    voice_id = VoiceID(config)
    face_id = FaceID(config)
    database = Database(config)

    arrived: set[str] = set()  # 记录已经到达
    num=0
    sign=True

    if not stream_name_list:
        stream_name_list=database.name_list

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
                
        sample_btn.click(sample, inputs=[img,aud,name], outputs=output_sample)
        begin_btn.click(train, inputs=None, outputs=train_status ) 
                
        with gr.Tab("视频检测：本地视频版"):
            with gr.Column():
                input_check = gr.Video(label="输入待检测的本地视频", format='mp4')
                check_local_btn = gr.Button("开始检测")
                wait_local = gr.Textbox(label="计算中", visible=False)
                output_check_local_true = gr.Textbox(label="检测结果：到教室的人", visible=False)
                output_check_local_false = gr.Textbox(label="检测结果：缺勤的人", visible=False)

        check_local_btn.click(waiting_local,inputs=None,outputs=wait_local).\
            then(check_local, inputs=input_check, outputs=[wait_local,output_check_local_true, output_check_local_false, check_local_btn])
            
        with gr.Tab("视频检测：实时检测版"):
            with gr.Column():
                with gr.Row():
                    audio_stream = gr.Audio(sources=["microphone"], type="numpy", label="请打开麦克风")
                    image_stream = gr.Image(sources=["webcam"], type="numpy", label="请打开摄像头")
                stream_output = gr.Textbox(label="检测结果")
        
        audio_stream.stream(
                handle_stream,
                inputs=[gr.State("aud_stream"), audio_stream],
                outputs=[stream_output],
                time_limit=2,
                stream_every=0.25,
                show_progress="full"
            )
        image_stream.stream(
                handle_stream,
                inputs=[gr.State("img_stream"), image_stream],
                outputs=[stream_output],
                time_limit=0.1,
                stream_every=0.1,
                show_progress="full"
            )
        
    demo.launch()