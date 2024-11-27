'''
识别 2024/11/21 18：38
主要任务是特征存储和点名阶段特征拼接识别
第六版：没有拼接脸部特征向量和声音特征向量，分开识别；
       尝试应用孪生网络处理面对512维向量的few-shot问题
    2024/11/22 19:06 增加存储功能，可以存储两个模型的权重文件和读入db.feature[list]，后者需要增加一个接口
    2024/11/23 14.14 增加全班名单提取函数
database_module6.py
class Database:
Database(config)
- def __init__(self) -> None
- save_feature_db(self, filename="feature_db.pt"):将数据库的学生特征存储到 .pt 文件中。
- get_all_names(self, filename="feature_db.pt") -> list[str]:检测指定的 .pt 文件，提取其中的所有学生姓名。
- store_feature(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:输入人名、脸特征向量、声音特征向量，存储到数据库
- train_face_siamese_model(self, num_epochs: int = 10) -> None:训练面部特征识别的孪生网络模型
- train_voice_siamese_model(self, num_epochs: int = 10) -> None:训练语音特征识别的孪生网络模型
- recognize_faces(self, face_features: list[torch.Tensor]) -> list[str]:批量处理面部特征向量，返回所有匹配学生姓名
- recognize_voice(self, input_voice_vector: torch.Tensor) -> str:使用训练好的语音孪生网络模型进行语音识别并返回声音最相似同学

不维护到场同学名单，仅根据当前输入的特征向量来输出识别到的同学
存储并提取全班同学信息、存储并应用两个模型的权重
主程序调用思路在代码文件最后
'''
from itertools import combinations
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from functools import lru_cache
import os
from pathlib import Path
from .siamese_net import SiameseNetwork, ContrastiveLoss, train
from .dataset import FeaturePairDataset, FeatureEntry, Student

class Database:
    def __init__(self) -> None:
        """
        数据库类，用于存储学生的面部特征和语音特征，并提供训练和识别功能。
        """
        self.students: dict[str, Student] = {}  # 学生特征数据库
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load features from checkpoints
        if Path("checkpoints/students.pt").exists():
            self.students = torch.load("checkpoints/students.pt", weights_only=False)
            print("存在students.pt：成功加载学生特征数据库！")
            
        for student in self.students.values():
            student.face_features = [feature.to(self.device) for feature in student.face_features]
            student.voice_features = [feature.to(self.device) for feature in student.voice_features]
            if student.face_prototype is not None:
                student.face_prototype = student.face_prototype.to(self.device)
            if student.voice_prototype is not None:
                student.voice_prototype = student.voice_prototype.to(self.device)

        # Initialize siamese models
        self.face_siamese_model = SiameseNetwork(input_dim=512, output_dim=64)  # 面部特征孪生网络模型
        self.voice_siamese_model = SiameseNetwork(input_dim=192, output_dim=64)  # 语音特征孪生网络模型
        if Path("checkpoints/face_siamese_model.pt").exists():
            self.face_siamese_model.load_state_dict(torch.load("checkpoints/face_siamese_model.pt", weights_only=True))
            print("存在face_siamese_model.pt：成功加载面部识别模型权重！可以不训练，直接进行面部识别")
        if Path("checkpoints/voice_siamese_model.pt").exists():
            self.voice_siamese_model.load_state_dict(torch.load("checkpoints/voice_siamese_model.pt", weights_only=True))
            print("存在voice_siamese_model.pt：成功加载语音识别模型权重！可以不训练，直接进行语音识别")
        self.face_siamese_model.to(self.device)
        self.voice_siamese_model.to(self.device)
        
        self.face_threshold = self.auto_threshold(attr="face_features")
        self.voice_threshold = self.auto_threshold(attr="voice_features")
    
    def add(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:
        """
        添加学生信息到数据库。
        
        :param name: 学生姓名。
        :param face_feature_vector: 学生面部特征向量。
        :param voice_feature_vector: 学生语音特征向量。
        """
        face_feature_vector = face_feature_vector.to(self.device)
        voice_feature_vector = voice_feature_vector.to(self.device)
        if name in self.students:
            self.students[name].face_features.append(face_feature_vector)
            self.students[name].voice_features.append(voice_feature_vector)
        else:
            self.students[name] = Student(name, [face_feature_vector], [voice_feature_vector])
    
    def save(self):
        if not Path("checkpoints").exists():
            Path("checkpoints").mkdir()
        torch.save(self.students, "checkpoints/students.pt")
        torch.save(self.face_siamese_model.state_dict(), "checkpoints/face_siamese_model.pt")
        torch.save(self.voice_siamese_model.state_dict(), "checkpoints/voice_siamese_model.pt")
        print("成功保存学生特征数据库、面部识别模型权重、语音识别模型权重！")
        
    @property
    def name_list(self) -> list[str]:
        return list(self.students.keys())
    
    def train_both(self, num_epochs: int = 32, batch_size=16) -> None:
        """
        训练两个孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        self.train_face_siamese_model(num_epochs, batch_size)
        self.train_voice_siamese_model(num_epochs, batch_size)

    def train_face_siamese_model(self, num_epochs: int = 32, batch_size=16) -> None:
        """
        训练面部特征识别的孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        dataset = FeaturePairDataset.from_students(self.students, 'face_features')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = ContrastiveLoss()
        print("开始训练面部识别模型...")
        train(self.face_siamese_model, dataloader, criterion, num_epochs, self.device)
        print("面部识别模型训练完成！")
        
        # get face prototype
        for student in self.students.values():
            features = torch.stack(student.face_features)
            student.face_prototype = self.face_siamese_model.forward_one(features).mean(dim=0)
        self.face_threshold = self.auto_threshold(attr="face_features")

    def train_voice_siamese_model(self, num_epochs: int = 32, batch_size=16) -> None:
        """
        训练语音特征识别的孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        dataset = FeaturePairDataset.from_students(self.students, 'voice_features')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = ContrastiveLoss()
        print("开始训练语音识别模型...")
        train(self.voice_siamese_model, dataloader, criterion, num_epochs, self.device)
        print("语音识别模型训练完成！")
        
        # get voice prototype
        for student in self.students.values():
            features = torch.stack(student.voice_features)
            student.voice_prototype = self.voice_siamese_model.forward_one(features).mean(dim=0)
        self.voice_threshold = self.auto_threshold(attr="voice_features")

    @torch.no_grad()
    def recognize_face(self, face_feature_vector: torch.Tensor, threshold: float=None) -> str|None:
        """
        识别输入的面部特征向量并返回相应的姓名。
        
        :param face_feature_vector: 输入的面部特征向量。
        :return: 识别到的姓名。
        """
        if face_feature_vector is None:
            return None
        threshold = self.face_threshold if threshold is None else threshold
    
        # 提取所有 prototype 和对应的姓名
        prototypes = []
        names = []
        for name, student in self.students.items():
            if student.face_prototype is not None:
                prototypes.append(student.face_prototype)
                names.append(name)
        
        if not prototypes:
            return None
        
        face_feature_vector = face_feature_vector.unsqueeze(0).to(self.device)
        face_feature_vector = self.face_siamese_model.forward_one(face_feature_vector)

        # 将所有 prototype 拼接成矩阵
        prototype_matrix = torch.stack(prototypes)  # Shape: (N, D)
        
        # 计算输入向量与所有 prototype 的余弦相似度
        similarities = F.cosine_similarity(face_feature_vector, prototype_matrix)  # Shape: (N,)

        # 找到相似度最高的 prototype
        max_similarity, idx = torch.max(similarities, dim=0) # Shape: (1,)
        print(f"相似度分别为：{similarities.tolist()}")

        # 检查是否超过阈值
        return names[idx] if max_similarity < threshold else None
    
    @torch.no_grad()
    def recognize_faces(self, face_features: list[torch.Tensor], threshold: float=None) -> list[str]:
        """
        批量处理面部特征向量列表，返回所有匹配的学生姓名。
        :param face_features: 面部特征向量列表
        :return: 匹配的学生姓名列表
        """
        if len(face_features) == 0:
            return []
        
        threshold = self.face_threshold if threshold is None else threshold
        
        # 提取所有 prototype 和对应的姓名
        prototypes = []
        names = []
        for name, student in self.students.items():
            if student.face_prototype is not None:
                prototypes.append(student.face_prototype)
                names.append(name)
        
        if not prototypes:
            return []

        # 将所有 prototype 拼接成矩阵
        prototype_matrix = torch.stack(prototypes) # (num_prototypes, out_dim)
        
        # 将输入向量拼接成矩阵
        face_feature_matrix = torch.stack(face_features)  # (batch_size, in_dim)
        face_feature_matrix = face_feature_matrix.to(self.device)
        face_feature_matrix = self.face_siamese_model.forward_one(face_feature_matrix) # (batch_size, out_dim)
        
        # 计算所有输入向量与所有 prototype 的余弦相似度 (batch_size, num_prototypes)
        similarities = F.cosine_similarity(face_feature_matrix.unsqueeze(1), prototype_matrix.unsqueeze(0), dim=2)  # Shape: (M, N)
        
        # 找到每个输入向量的最高相似度及对应的索引
        max_similarities, indices = torch.max(similarities, dim=1)  # (batch_size,)
        print(f"相似度分别为：{similarities}")

        # 根据阈值判断并返回对应的姓名
        recognized_names = [names[idx] if max_similarity < threshold else None
                            for max_similarity, idx in zip(max_similarities, indices)]
        recognized_names = set(recognized_names)
        recognized_names.remove(None)
        
        return list(recognized_names)

    @torch.no_grad()
    def recognize_voice(self, voice_feature_vector: torch.Tensor, threshold: float=None) -> str|None:
        """
        识别输入的语音特征向量并返回相应的姓名。
        
        :param voice_feature_vector: 输入的语音特征向量。
        :return: 识别到的姓名。
        """
        if voice_feature_vector is None:
            return None
        
        threshold = self.voice_threshold if threshold is None else threshold
        
        # 提取所有 prototype 和对应的姓名
        prototypes = []
        names = []
        for name, student in self.students.items():
            if student.voice_prototype is not None:
                prototypes.append(student.voice_prototype)
                names.append(name)
        
        if not prototypes:
            return None

        voice_feature_vector = voice_feature_vector.unsqueeze(0).to(self.device)
        voice_feature_vector = self.voice_siamese_model.forward_one(voice_feature_vector) # Shape: (1, D)
        
        # 将所有 prototype 拼接成矩阵
        prototype_matrix = torch.stack(prototypes)  # Shape: (N, D)
        
        # 计算输入向量与所有 prototype 的余弦相似度
        similarities = F.cosine_similarity(voice_feature_vector, prototype_matrix)  # Shape: (N,)
        
        # 找到相似度最高的 prototype
        max_similarity, idx = torch.max(similarities, dim=0) # Shape: (1,)
        print(f"相似度分别为：{similarities.tolist()}")
        
        return names[idx] if max_similarity < threshold else None
    
    @torch.no_grad()
    def recognize_voices(self, voice_features: torch.Tensor, threshold: float=None) -> list[str]:
        """
        批量处理语音特征向量列表，返回所有匹配的学生姓名。
        
        :param voice_features: 语音特征向量列表。
        :return: 匹配的学生姓名列表。
        """
        if len(voice_features) == 0:
            return []
        
        threshold = self.voice_threshold if threshold is None else threshold
        
        # 提取所有 prototype 和对应的姓名
        prototypes = []
        names = []
        for name, student in self.students.items():
            if student.voice_prototype is not None:
                prototypes.append(student.voice_prototype)
                names.append(name)
        
        if not prototypes:
            return []

        # 将所有 prototype 拼接成矩阵
        prototype_matrix = torch.stack(prototypes) # (num_prototypes, out_dim)
        
        # 将输入向量拼接成矩阵
        voice_features = voice_features.to(self.device) # (batch_size, in_dim)
        voice_features = self.voice_siamese_model.forward_one(voice_features) # (batch_size, out_dim)
        
        # 计算所有输入向量与所有 prototype 的余弦相似度 (batch_size, num_prototypes)
        similarities = F.cosine_similarity(voice_features.unsqueeze(1), prototype_matrix.unsqueeze(0), dim=2)
        
        # 找到每个输入向量的最高相似度及对应的索引
        max_similarities, indices = torch.max(similarities, dim=1) # (batch_size,)
        print(f"相似度分别为：{similarities}")
        
        # 根据阈值判断并返回对应的姓名
        recognized_names = [names[idx] if max_similarity < threshold else None
                            for max_similarity, idx in zip(max_similarities, indices)]
        recognized_names = set(recognized_names)
        recognized_names.remove(None)
        
        return list(recognized_names)
    
    @torch.no_grad()
    def auto_threshold(self, attr: str, alpha=0.3) -> float:
        """
        自动调整阈值，结合类内和类间距离动态计算。
        
        :param attr: 特征类型（'face_features' 或 'voice_features'）。
        :param alpha: 类内和类间距离的权重（0-1），默认 0.3。
        :return: 自动计算的阈值。
        """
        assert attr in ['face_features', 'voice_features'], "Invalid attribute"

        # 提取所有学生的 prototype
        prototypes = {name: getattr(student, f"{attr[:-9]}_prototype") for name, student in self.students.items()}
        prototypes = {name: proto for name, proto in prototypes.items() if proto is not None}
        
        assert len(prototypes) > 1, "Not enough prototypes for threshold calculation"

        # 计算类间距离
        inter_distances = []
        for (name1, proto1), (name2, proto2) in combinations(prototypes.items(), 2):
            inter_distance = F.cosine_similarity(proto1.unsqueeze(0), proto2.unsqueeze(0)).item()
            inter_distances.append(inter_distance)
        
        max_inter_distance = max(inter_distances)

        # 动态计算阈值
        threshold = alpha * max_inter_distance
        print(f"计算出的 {attr} 阈值: {threshold:.4f}")
        return threshold
        
# def main():
#     # 初始化数据库
#     db = Database()
    
#     # 模拟学生特征存储
#     print("正在存储学生特征...")

#     student_data = [
#         ("Alice", torch.rand(512),torch.rand(192)),
#         ("Bob", torch.rand(512),torch.rand(192)),
#         ("Charlie", torch.rand(512),torch.rand(192)),
#     ]
    
#     for name, face_vector, voice_vector in student_data:
#         db.store_feature(name, face_vector, voice_vector)
#     print("学生特征存储完成！")

#     # 训练脸部识别孪生网络模型
#     print("正在训练脸部识别孪生网络模型...")
#     db.train_face_siamese_model(num_epochs=5)
#     print("脸部识别模型训练完成！")

#     # 训练语音识别孪生网络模型
#     print("正在训练语音识别孪生网络模型...")
#     db.train_voice_siamese_model(num_epochs=5)
#     print("语音识别模型训练完成！")

#     # 备份同学姓名、两类特征向量
#     print("正在备份可识别同学信息...")
#     db.save_feature_db(filename="feature_db.pt")
#     print("可识别同学信息备份完成！")
    
#     # 提取所有学生姓名
#     print("检测并提取 feature_db.pt 中的学生姓名...")
#     student_names = db.get_all_names()
#     if student_names:
#         print("提取的学生姓名列表：", student_names)
#     else:
#         print("未找到学生姓名或 feature_db.pt 文件无效。")
        
#     # 脸部识别
#     print("开始批量面部识别...")
#     test_face_vectors = [torch.rand(512) for _ in range(5)]
#     recognized_faces = db.recognize_faces(test_face_vectors)
#     print(f"批量面部识别结果：{recognized_faces}")

#     # 语音识别
#     print("开始语音识别测试...")
#     test_voice_vector = torch.rand(192)
#     recognized_voice = db.recognize_voice(test_voice_vector)
#     print(f"语音识别结果：{recognized_voice}")

if __name__ == "__main__":
    # main()
    pass