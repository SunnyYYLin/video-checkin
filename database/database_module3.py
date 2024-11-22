'''
刘鑫宇：识别 2024/11/14 ？
主要任务是特征存储和点名阶段特征拼接识别
代码最后有主程序参考
第三版：没有拼接脸部特征向量和声音特征向量，分开识别；
       脸部特征向量分类识别选择SVM支持向量机
       声音特征向量分类识别仍然取KNN，方法还可以变动
'''
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

class FeatureEntry:
    def __init__(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor):
        self.name = name
        self.face_feature_vector = face_feature_vector
        self.voice_feature_vector = voice_feature_vector

class Database:
    def __init__(self, n_neighbors=1):
        self.feature_db = []  # 用于存储包含面部和语音特征向量的条目
        self.face_svm = SVC(kernel='linear')  # 面部SVM模型
        self.voice_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')  # 语音KNN模型
        self.face_scaler = StandardScaler()  # 用于面部特征的标准化器

    def store_feature(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:
        """
        将一个学生的面部和语音特征存储到数据库。
        """
        # 存储面部特征和语音特征
        self.feature_db.append(FeatureEntry(name, face_feature_vector, voice_feature_vector))
        print(f"成功存储 {name} 的面部和语音特征向量。")

    def train_face_svm(self):
        """
        训练面部特征识别的SVM模型。
        """
        if len(self.feature_db) == 0:
            print("数据库为空，无法训练面部SVM模型。")
            return
    
        # 提取数据库中所有面部特征向量和对应的姓名
        face_features = np.array([entry.face_feature_vector.numpy() for entry in self.feature_db])
        names = [entry.name for entry in self.feature_db]
        
        # 标准化面部特征
        self.face_scaler.fit(face_features)  # 在训练集上训练标准化器
        face_features_scaled = self.face_scaler.transform(face_features)  # 对特征进行标准化
        
        # 训练SVM模型
        self.face_svm.fit(face_features_scaled, names)
        print("面部识别SVM模型已成功训练。")

    def train_voice_knn(self):
        """
        训练语音特征识别的KNN模型。
        """
        if len(self.feature_db) == 0:
            print("数据库为空，无法训练声音KNN模型。")
            return
        
        # 提取数据库中所有语音特征向量和对应的姓名
        voice_features = np.array([entry.voice_feature_vector.numpy() for entry in self.feature_db])
        names = [entry.name for entry in self.feature_db]
        
        # 使用这些数据训练语音KNN模型
        self.voice_knn.fit(voice_features, names)
        print("语音识别KNN模型已成功训练。")

    def compare_face_feature_vector(self, face_feature_vector: torch.Tensor) -> str:
        """
        使用训练好的 SVM 模型对单个面部特征向量进行识别，返回匹配的学生姓名。
        :param face_feature_vector: 面部特征向量
        :return: 匹配的学生姓名
        """
        # 使用训练好的标准化器对新的面部特征进行标准化
        face_feature_scaled = self.face_scaler.transform(face_feature_vector.numpy().reshape(1, -1))
        
        # 使用 SVM 模型预测匹配的学生姓名
        predicted_name = self.face_svm.predict(face_feature_scaled)
        return predicted_name[0]  # 返回匹配的学生姓名

    def compare_face_feature_vectors(self, face_features: list[torch.Tensor]) -> list[str]:
        """
        批量处理面部特征向量列表，返回所有匹配的学生姓名。
        :param face_features: 面部特征向量列表
        :return: 匹配的学生姓名列表
        """
        matched_names = []
        for face_feature in face_features:
            # 对每个面部特征向量进行识别
            matched_name = self.compare_face_feature_vector(face_feature)
            matched_names.append(matched_name)
        
        return matched_names

    def compare_voice_feature_vector(self, voice_feature_vector: torch.Tensor) -> str:
        """
        使用训练好的 KNN 模型对语音特征向量进行识别，返回匹配的学生姓名。
        :param voice_feature_vector: 声音特征向量
        :return: 匹配的学生姓名
        """
        # 使用KNN模型直接进行预测
        predicted_name = self.voice_knn.predict(voice_feature_vector.numpy().reshape(1, -1))
        return predicted_name[0]  # 返回匹配的学生姓名