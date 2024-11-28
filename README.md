SunnyYYLin: 写出这个人脸-声纹联合签到系统的readme

Workspace: # 人脸-声纹联合签到系统

## 项目简介

本项目是一个基于人脸识别和声纹识别的课堂考勤系统，旨在通过结合人脸和声纹双重识别技术，实现高效、准确的课堂考勤。系统支持本地视频检测和实时检测两种模式，并提供了用户友好的Web UI界面。

## 目录结构

```
.
├── __pycache__/
├── .gitignore
├── audio/
├── checkpoints/
│   ├── face_siamese_model.pt
│   ├── students.pt
│   └── voice_siamese_model.pt
├── config/
│   ├── __init__.py
│   ├── __pycache__/
│   └── config.py
├── data/
│   ├── case_videos/
│   ├── kws/
│   └── pos/
├── database/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── database.py
│   ├── dataset.py
│   └── siamese_net.py
├── docs/
│   ├── begin.md
│   ├── faceid_report/
│   ├── voice_id.md
│   └── web_ui.md
├── environment.yml
├── face_id/
│   ├── __init__.py
│   ├── __pycache__/
│   └── face_recognition.py
├── LICENSE
├── main.py
├── name_roll.py
├── README.md
├── voice_id/
│   ├── __init__.py
│   ├── __pycache__/
│   ├── README.md
│   ├── utils.py
│   └── voice_id.py
```

## 安装与配置

### 环境依赖

请确保已安装以下依赖项：

- Python 3.12
- PyTorch
- torchaudio
- facenet-pytorch
- speechbrain
- Gradio
- moviepy
- PIL
- numpy
- scipy
- matplotlib

可以通过以下命令安装依赖：

```sh
conda env create -f environment.yml
conda activate checkin
```

### 配置文件

在 

config

 目录下的 

config.py

 文件中配置系统参数。

## 使用说明

### 运行主程序

在终端中运行以下命令启动系统：

```sh
python main.py
```

### Web UI

系统提供了一个基于 Gradio 的 Web UI，用户可以通过浏览器访问并使用以下功能：

1. **输入样本**：用户可以输入待检测人员的照片、声音和姓名，系统将样本信息存储到数据库中。
2. **本地视频检测**：用户可以上传本地视频，系统将检测视频中的到场人员和缺勤人员。
3. **实时检测**：用户可以通过摄像头和麦克风进行实时检测，系统将实时输出检测结果。

### 样本输入

在 Web UI 的“输入样本”标签页中，上传人脸图像和语音音频，并输入姓名，点击“输入样本”按钮。

### 训练模型

在 Web UI 的“输入样本”标签页中，点击“开始训练”按钮，系统将训练人脸和声纹识别模型。

### 本地视频检测

在 Web UI 的“视频检测：本地视频版”标签页中，上传待检测的视频文件，点击“开始检测”按钮，系统将输出到场人员和缺勤人员名单。

### 实时检测

在 Web UI 的“视频检测：实时检测版”标签页中，打开摄像头和麦克风，点击“开始检测”按钮，系统将实时输出检测结果。

## 主要功能

### 人脸识别

使用 `facenet-pytorch` 库中的 InceptionResnetV1 模型进行人脸特征提取，并通过孪生网络进行人脸识别。

### 声纹识别

使用 

speechbrain

 库中的 ECAPA-TDNN 模型进行声纹特征提取，并通过孪生网络进行声纹识别。

### 数据库管理

系统使用 

torch

 库管理学生特征数据库，支持添加、存储和加载学生特征。

## 贡献

欢迎对本项目进行贡献！如果您有任何建议或发现了问题，请提交 issue 或 pull request。

## 许可证

本项目基于 [GNU General Public License v3.0](LICENSE) 许可证开源。

SunnyYYLin: 请写进README.md

GitHub Copilot: # 人脸-声纹联合签到系统

## 项目简介

本项目是一个基于人脸识别和声纹识别的课堂考勤系统，旨在通过结合人脸和声纹双重识别技术，实现高效、准确的课堂考勤。系统支持本地视频检测和实时检测两种模式，并提供了用户友好的Web UI界面。

## 目录结构

```
.
├── checkpoints/
│   ├── face_siamese_model.pt
│   ├── students.pt
│   └── voice_siamese_model.pt
├── config/
│   ├── __init__.py
│   ├── __pycache__/
│   └── config.py
├── database/
│   ├── __init__.py
│   ├── database.py
│   ├── dataset.py
│   └── siamese_net.py
├── docs/
├── environment.yml
├── face_id/
│   ├── __init__.py
│   └── face_recognition.py
├── LICENSE
├── main.py
├── name_roll.py
├── README.md
├── voice_id/
│   ├── __init__.py
│   ├── utils.py
│   └── voice_id.py
```

## 安装与配置

### 环境依赖

可以通过以下命令安装依赖：

```sh
conda env create -f environment.yml
conda activate checkin
```

也可以使用：

```sh
conda create -n checkin python=3.12
conda activate checkin
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
pip install facenet-pytorch speechbrain gradio pydub moviepy
```

### 配置文件

在 

config

 目录下的 

config.py

 文件中配置系统参数。

## 使用说明

### 运行主程序

在终端中运行以下命令启动系统：

```sh
python main.py
```

### Web UI

系统提供了一个基于 Gradio 的 Web UI，用户可以通过浏览器访问并使用以下功能：

1. **输入样本**：用户可以输入待检测人员的照片、声音和姓名，系统将样本信息存储到数据库中。
2. **本地视频检测**：用户可以上传本地视频，系统将检测视频中的到场人员和缺勤人员。
3. **实时检测**：用户可以通过摄像头和麦克风进行实时检测，系统将实时输出检测结果。

### 样本输入

在 Web UI 的“输入样本”标签页中，上传人脸图像和语音音频，并输入姓名，点击“输入样本”按钮。

### 训练模型

在 Web UI 的“输入样本”标签页中，点击“开始训练”按钮，系统将训练人脸和声纹识别模型。

### 本地视频检测

在 Web UI 的“视频检测：本地视频版”标签页中，上传待检测的视频文件，点击“开始检测”按钮，系统将输出到场人员和缺勤人员名单。

### 实时检测

在 Web UI 的“视频检测：实时检测版”标签页中，打开摄像头和麦克风，点击“开始检测”按钮，系统将实时输出检测结果。

## 主要功能

### 人脸识别

使用 `facenet-pytorch` 库中的 InceptionResnetV1 模型进行人脸特征提取，并通过孪生网络进行人脸识别。

### 声纹识别

使用 `speechbrain` 库中的 ECAPA-TDNN 模型进行声纹特征提取，并通过孪生网络进行声纹识别。

### 数据库管理

系统使用 `torch` 库管理学生特征数据库，支持添加、存储和加载学生特征。

## 贡献

欢迎对本项目进行贡献！如果您有任何建议或发现了问题，请提交 issue 或 pull request。

## 许可证

本项目基于 [GNU General Public License v3.0](LICENSE) 许可证开源。