'''
刘鑫宇：识别 2024/11/14 18:24
主要任务是特征存储和点名阶段特征拼接识别
代码最后有主程序参考
第二版：没有拼接脸部特征向量和声音特征向量，分开识别，仍然选择knn
'''
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class FeatureEntry:
    def __init__(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor):
        self.name = name
        self.face_feature_vector = face_feature_vector
        self.voice_feature_vector = voice_feature_vector

class Database:
    def __init__(self, n_neighbors=1):
        self.feature_db = []  # 用于存储包含面部和语音特征向量的条目
        self.face_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')  # 面部KNN模型
        self.voice_knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')  # 语音KNN模型

    def store_feature(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:
        """
        将一个学生的面部和语音特征存储到数据库。
        """
        # 存储面部特征和语音特征
        self.feature_db.append(FeatureEntry(name, face_feature_vector, voice_feature_vector))
        print(f"成功存储 {name} 的面部和语音特征向量。")

    def train_face_knn(self):
        """
        训练面部特征识别的KNN模型。
        """
        # 提取数据库中所有面部特征向量和对应的姓名
        face_features = np.array([entry.face_feature_vector.numpy() for entry in self.feature_db])
        names = [entry.name for entry in self.feature_db]
        
        # 使用这些数据训练面部KNN模型
        self.face_knn.fit(face_features, names)
        print("面部识别KNN模型已成功训练。")

    def train_voice_knn(self):
        """
        训练语音特征识别的KNN模型。
        """
        # 提取数据库中所有语音特征向量和对应的姓名
        voice_features = np.array([entry.voice_feature_vector.numpy() for entry in self.feature_db])
        names = [entry.name for entry in self.feature_db]
        
        # 使用这些数据训练语音KNN模型
        self.voice_knn.fit(voice_features, names)
        print("语音识别KNN模型已成功训练。")

    def compare_face_feature_vector(self, face_feature_vector: torch.Tensor) -> str:
        """
        使用训练好的 KNN 模型对单个面部特征向量进行识别，返回匹配的学生姓名。
        :param face_feature_vector: 面部特征向量
        :return: 匹配的学生姓名
        """
        # 查找最近的邻居及其距离
        distances, indices = self.face_knn.kneighbors(face_feature_vector.numpy().reshape(1, -1))
        
        # 最近邻的名字
        predicted_name = self.face_knn.classes_[indices[0][0]]

        return predicted_name  # 返回匹配的学生姓名

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
        # 查找最近的邻居及其距离
        distances, indices = self.voice_knn.kneighbors(voice_feature_vector.numpy().reshape(1, -1))
        
        # 最近邻的名字
        predicted_name = self.voice_knn.classes_[indices[0][0]]

        return predicted_name  # 返回匹配的学生姓名



'''主程序 调用思路
给sjz的参考，以下所有内容均不属于我database_module.py文件，而是在主程序中使用
if __name__ == "__main__":
    # 创建数据库对象
    db = Database(n_neighbors=3)

    # 存储面部和语音特征向量：输入同一个人的姓名、面部特征向量和语音特征向量
    face_feature_a = torch.rand(512)  # 示例面部特征
    voice_feature_a = torch.rand(512)  # 示例语音特征
    db.store_feature('liuxinyu', face_feature_a, voice_feature_a)

    face_feature_b = torch.rand(512)  # 示例面部特征
    voice_feature_b = torch.rand(512)  # 示例语音特征
    db.store_feature('another_name', face_feature_b, voice_feature_b)

    # 在全部特征向量储存完毕后，训练KNN模型
    db.train_face_knn()
    db.train_voice_knn()

    # 面部识别：输入的face_feature是list[torch.Tensor]，列表中每个人脸的特征向量均为512维列向量
    face_features_test = [torch.rand(512), torch.rand(512)]  # 示例面部特征列表，例子仅有两个元素
    matched_face_names = db.compare_face_feature_vectors(face_features_test)  # 输出基于face_feature列表所识别的所有人名

    # 语音识别：输入的voice_feature是512维torch.Tensor
    voice_feature_test = torch.rand(512)  # 示例语音特征
    matched_voice_name = db.compare_voice_feature_vector(voice_feature_test)  # 输出基于voice_feature所识别的单个人名
'''