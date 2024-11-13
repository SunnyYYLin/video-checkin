from facenet_pytorch import InceptionResnetV1,MTCNN
import torch
from PIL import Image
import numpy as np

class FaceID:
    def __init__(self, threshold=0.6,device='cuda'):
        #初始化模型，设置相似度阈值
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.threshold = threshold
        self.mtcnn = MTCNN(keep_all=True,device=self.device)

    def extract_feature(self, image: Image.Image):
        """
        提取图像中的每个检测到的人脸的特征向量。
        参数:
            image: 包含人脸的PIL图像对象。
        返回:
            特征向量列表，每个特征向量代表图像中的一个人脸。
        """
        face_positions,_ = self.detect_faces(image)  #得到人脸的位置
        faces = []
        for (x1,y1,x2,y2) in face_positions:
            face = image.crop((x1,y1,x2,y2))    #将人脸位置转化为人脸PIL图像对象
            faces.append(face)
        features = []
        for face in faces:
            face_tensor = self.preprocess(face)  #将每个人脸转换为张量
            with torch.no_grad():
                feature_vector = self.model(face_tensor).numpy()
            features.append(feature_vector)
        return features

    def existed_features(self, feature_vectors, existing_features_list):
        """
        检测每个特征向量是否已存在于现有特征列表中，若不存在则加入该列表。
        参数:
            feature_vectors: 当前提取到的人脸特征向量列表。
            existing_features_list: 外部传入的特征向量列表。
        """
        for feature in feature_vectors:
            if not any(self.is_similar(feature, existing_feature) for existing_feature in existing_features_list):
                existing_features_list.append(feature)

    def is_similar(self, feature1, feature2):
        """
        判断两个特征向量是否相似。
        参数:
            feature1, feature2: 两个人脸特征向量。
        返回:
            布尔值，指示两个特征向量是否相似。
        """
        distance = np.linalg.norm(feature1 - feature2)
        return distance < self.threshold

    def preprocess(self, face_image):
        #预处理人脸图像并转换为模型所需的输入格式
        face_image = face_image.resize((160, 160))
        face_tensor = torch.tensor(np.array(face_image) / 255).permute(2, 0, 1).unsqueeze(0).float()
        return face_tensor

    def detect_faces(self, image):
        faces, probs = self.mtcnn.detect(image)
        return faces,probs