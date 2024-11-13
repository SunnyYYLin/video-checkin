'''
刘鑫宇：识别 2024/11/12
主要任务是特征存储和点名阶段特征拼接识别
代码最后有主程序参考
'''
import torch
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class FeatureEntry:
    def __init__(self, name: str, feature_vector: torch.Tensor):
        self.name = name
        self.feature_vector = feature_vector

class Database:
    def __init__(self, n_neighbors=1):
        self.feature_db = []  # 用于存储拼接后的特征向量
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')  # KNN模型

    def store_combined_feature(self, name: str, face_feature: torch.Tensor, voice_feature: torch.Tensor) -> None:
        # 拼接特征向量
        combined_feature = torch.cat((face_feature, voice_feature), dim=-1)
        
        # 存储拼接特征向量到数据库
        self.feature_db.append(FeatureEntry(name=name, feature_vector=combined_feature))
        print(f"成功存储 {name} 的拼接特征向量。")

    def train_knn(self):
        # 提取数据库中所有特征向量和对应的姓名
        feature_vectors = np.array([entry.feature_vector.numpy() for entry in self.feature_db])
        names = [entry.name for entry in self.feature_db]
        
        # 使用这些数据训练KNN模型
        self.knn.fit(feature_vectors, names)
        print("KNN 模型已成功训练。")

    def compare_feature_vector(self, face_feature: torch.Tensor, voice_feature: torch.Tensor) -> str:
        """
        使用已训练的KNN算法比对特征向量并返回匹配的学生姓名，同时计算欧氏距离。

        这个函数是单独对一个拼接特征向量进行分类，如果您的数据是
        face_features: list[torch.Tensor], voice_feature: torch.Tensor，
        那么不用使用这个函数

        :param face_feature: 面部特征向量
        :param voice_feature: 声音特征向量
        :return: 匹配的学生姓名和欧氏距离
        """
        # 拼接面部特征向量和声音特征向量
        combined_feature = torch.cat((face_feature, voice_feature), dim=-1)

        # 查找最近的邻居及其距离
        distances, indices = self.knn.kneighbors(combined_feature.numpy().reshape(1, -1))
        
        # 最近邻的名字
        predicted_name = self.knn.classes_[indices[0][0]]
        
        # 计算欧氏距离
        distance = distances[0][0]

        return predicted_name, distance  # 返回匹配的学生姓名和欧氏距离

    def compare_feature_vectors(self, face_features: list[torch.Tensor], voice_feature: torch.Tensor) -> str:
        """
        批量处理面部特征向量列表与一个声音特征向量的匹配，返回欧氏距离最小的匹配学生姓名。
        :param face_features: 面部特征向量列表
        :param voice_feature: 声音特征向量
        :return: 匹配的学生姓名
        """
        # 将声音特征与每个面部特征进行拼接
        combined_features = [torch.cat((face_feature, voice_feature), dim=-1) for face_feature in face_features]
    
        # 将拼接后的特征向量转换为 NumPy 数组
        combined_features = np.array([feature.numpy().reshape(1, -1) for feature in combined_features])

        # 计算所有拼接后的特征的欧氏距离
        distances, indices = self.knn.kneighbors(combined_features)
    
        # 找到距离最小的索引
        min_distance_index = np.argmin(distances)  # 在所有查询样本的距离中找到最小值
        min_distance = distances.flatten()[min_distance_index]  # 获取最小距离
        closest_index = indices.flatten()[min_distance_index]  # 获取对应的最近邻索引
    
        # 使用 KNN 的索引获取匹配的学生姓名
        matched_name = self.knn.classes_[closest_index]  # 获取匹配的学生姓名

        return matched_name


'''主程序 调用思路
给sjz的参考，以下所有内容均不属于我database_module.py文件，而是在主程序中使用

if __name__ == "__main__":
    # 存储数据到数据库，下面仅仅是数据例子
    注意：zbw给出的face_feature是[torch.Tensor]，列表中每个人脸的特征向量均为512维列向量
    face_feature_a = torch.rand(512)  # 示例数据（仅是例子）
    voice_feature_a = torch.rand(128？)  # 示例数据（得问ljh，我不知道他voice_feature特征向量的维度）
    face_feature_b = torch.rand(512)  # 示例数据
    voice_feature_b = torch.rand(128)  # 示例数据

    # 创建数据库对象
    db = Database(n_neighbors=3)

    # 存储学生的特征向量
    db.store_combined_feature('liuxinyu', face_feature_a, voice_feature_a)
    db.store_combined_feature('another_name', face_feature_b, voice_feature_b)

    # 在全部特征向量储存完毕后，显式训练 KNN 模型
    db.train_knn()

    # 假设您有一些测试样本：face_features: list[torch.Tensor], voice_feature: torch.Tensor
    face_features_test = [torch.rand(512), torch.rand(512)]  # 示例面部特征列表
    voice_feature_test = torch.rand(128)  # 示例声音特征向量

    # 使用训练好的 KNN 模型来预测测试样本的姓名
    matched_name = db.compare_feature_vectors(face_features_test, voice_feature_test)
    print(f"预测的学生姓名：{matched_name}")
'''