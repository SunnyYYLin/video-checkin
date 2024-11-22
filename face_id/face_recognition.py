from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
from PIL import Image
import numpy as np

class FaceID:
    def __init__(self, threshold=0.4, count_threshold = 12, width_threshold = 20, height_threshold =20, device='cuda'):
        # 初始化模型，设置相似度阈值
        self.device = torch.device(device)
        self.model = InceptionResnetV1(pretrained='vggface2').eval()
        self.threshold = threshold
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.positions = [] #用来记录各个人脸框的区域
        self.count = [] # 用来记录每个区域连续出现的次数
        self.count_threshold = count_threshold # 连续出现次数的阈值
        self.width_threshold = width_threshold
        self.height_threshold = height_threshold

    def extract_features(self, image: Image.Image, mode='enter'):
        """
        提取图像中的每个检测到的人脸的特征向量。
        参数:
            image: 包含人脸的PIL图像对象。
        返回:
            特征向量列表，每个特征向量代表图像中的一个人脸。
        """
        # 转换图像格式为模型输入的张量格式
        face_positions, _ = self.detect_faces(image)  # 检测并裁剪所有人脸，返回人脸图像列表
        if mode == 'checkin':
            self.faces_count(face_positions)
        faces = []
        for (x1, y1, x2, y2) in face_positions:
            if mode == 'checkin':
                width = x2-x1
                height = y2-y1
                region = self.is_in_faces((x1,y1,x2,y2))
                if self.count[self.positions.index(region)] >= self.count_threshold and width>=self.width_threshold and height >= self.height_threshold: #只有满足三种要求的人脸才被认为是有效人脸
                    face = image.crop((x1, y1, x2, y2))
                    faces.append(face)
            elif mode == 'enter':
                    face = image.crop((x1, y1, x2, y2))
                    faces.append(face)
        features = []
        for face in faces:
            face_tensor = self.preprocess(face)  # 将每个人脸转换为张量
            with torch.no_grad():
                feature_vector = self.model(face_tensor).numpy().squeeze()
            features.append(feature_vector)
        if mode == 'enter':
            return features[0]
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
        使用余弦相似度判断两个特征向量是否相似。
        """
        # 将特征向量展平为一维
        feature1 = feature1.flatten()
        feature2 = feature2.flatten()
        
        # 归一化
        feature1_norm = feature1 / np.linalg.norm(feature1)
        feature2_norm = feature2 / np.linalg.norm(feature2)
        
        # 计算余弦相似度
        similarity = np.dot(feature1_norm, feature2_norm)
        return similarity > (1 - self.threshold)  # 阈值调整为相似度

    def is_in_face(self, face_position, region):
        if(region == None):
            return False
        else:
            x1,y1,x2,y2 = face_position
            rx1,ry1,rx2,ry2 = region
            return x1>=rx1 and y1>=ry1 and x2<=rx2 and y2<=ry2

    def is_in_faces(self, detect_face_position):
        if self.positions == []:
            return False
        else:
            for region in self.positions:
                if self.is_in_face(detect_face_position,region):
                    return region
                else:
                    continue
            return False

    def faces_count(self, detect_face_positions):
        count_copy = np.copy(self.count)
        for detect_face_position in detect_face_positions:
            result = self.is_in_faces(detect_face_position)
            if result:
                self.count[self.positions.index(result)] += 1
            else:
                region = self.cap_region(detect_face_position)
                self.positions.append(region)
                self.count.append(1)
        for i,old_count in enumerate(count_copy):
            if old_count == self.count[i]:
                self.count[i] = 0

    def preprocess(self, face_image):
        # 预处理人脸图像并转换为模型所需的输入格式
        face_image = face_image.resize((160, 160))
        face_tensor = torch.tensor(np.array(face_image) / 255).permute(2, 0, 1).unsqueeze(0).float()
        return face_tensor

    def detect_faces(self, image):
        faces, probs = self.mtcnn.detect(image)
        return faces, probs

    def cap_region(self, position):
        x1,y1,x2,y2 = position
        width = x2-x1
        height = y2-y1
        return (x1-width/2,y1-height/2,x2+width/2,y2+height/2)
