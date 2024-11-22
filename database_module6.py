'''
刘鑫宇：识别 2024/11/21 18：38
主要任务是特征存储和点名阶段特征拼接识别
第六版：没有拼接脸部特征向量和声音特征向量，分开识别；
       尝试应用孪生网络处理面对512维向量的few-shot问题
database_module6.py
class Database:
Database(config)
- def __init__(self) -> None
- store_feature(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:输入人名、脸特征向量、声音特征向量，存储到数据库
- train_face_siamese_model(self, num_epochs: int = 10) -> None:训练面部特征识别的孪生网络模型
- train_voice_siamese_model(self, num_epochs: int = 10) -> None:训练语音特征识别的孪生网络模型
- recognize_faces(self, face_features: list[torch.Tensor]) -> list[str]:批量处理面部特征向量，返回所有匹配学生姓名
- recognize_voice(self, input_voice_vector: torch.Tensor) -> str:使用训练好的语音孪生网络模型进行语音识别并返回声音最相似同学

不维护到场同学名单，仅根据当前输入的特征向量来输出识别到的同学
主程序调用思路在代码文件最后
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    孪生网络模型，用于处理不同的特征向量（面部特征或语音特征）。
    """
    def __init__(self, input_dim: int):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # 第一层全连接层
        self.fc2 = nn.Linear(1024, 512)       # 第二层全连接层，输出512维嵌入特征

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        对一个输入进行前向传播。
        
        :param x: 输入张量（面部或语音特征向量）。
        :return: 处理后的输出张量（嵌入特征）。
        """
        x = F.relu(self.fc1(x))  # 第一个全连接层 + ReLU 激活
        x = self.fc2(x)          # 第二个全连接层
        return x

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> tuple:
        """
        对两个输入进行前向传播，返回它们的嵌入特征。
        
        :param input1: 第一个输入张量（面部或语音特征向量）。
        :param input2: 第二个输入张量（面部或语音特征向量）。
        :return: 包含两个输出嵌入的元组。
        """
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0) -> None:
        """
        对比损失函数，用于孪生网络训练。
        
        :param margin: 用于区分相似和不同样本的距离阈值。
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1: torch.Tensor, output2: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        计算两个输出特征嵌入之间的对比损失。
        
        :param output1: 第一个输出张量（特征嵌入）。
        :param output2: 第二个输出张量（特征嵌入）。
        :param label: 二进制标签（1 表示相同，0 表示不同）。
        :return: 计算得到的对比损失。
        """
        euclidean_distance = F.pairwise_distance(output1, output2)  # 计算欧氏距离
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) + 
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss


class FeatureEntry:
    def __init__(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor):
        """
        特征条目类，存储学生的姓名、面部特征向量和语音特征向量。
        
        :param name: 学生的姓名。
        :param face_feature_vector: 学生的面部特征向量。
        :param voice_feature_vector: 学生的语音特征向量。
        """
        self.name = name
        self.face_feature_vector = face_feature_vector
        self.voice_feature_vector = voice_feature_vector


class FeaturePairDataset(Dataset):
    def __init__(self, feature_db: list[FeatureEntry], feature_type: str) -> None:
        """
        特征配对数据集类，用于训练孪生网络。
        
        :param feature_db: 存储所有学生面部或语音特征的数据库。
        :param feature_type: 用于区分面部特征或语音特征（'face' 或 'voice'）。
        """
        self.feature_db = feature_db
        self.feature_type = feature_type
        self.pairs = self.create_pairs()  # 创建特征对

    def create_pairs(self) -> list[tuple]:
        """
        创建特征对，用于孪生网络训练。
        
        :return: 特征对列表，包含特征向量对和标签。
        """
        pairs = []
        for i in range(len(self.feature_db)):
            for j in range(i + 1, len(self.feature_db)):
                if self.feature_type == 'face':
                    pairs.append((self.feature_db[i].face_feature_vector, self.feature_db[j].face_feature_vector, 1))  # 相同
                    pairs.append((self.feature_db[i].face_feature_vector, self.feature_db[j].face_feature_vector, 0))  # 不同
                elif self.feature_type == 'voice':
                    pairs.append((self.feature_db[i].voice_feature_vector, self.feature_db[j].voice_feature_vector, 1))  # 相同
                    pairs.append((self.feature_db[i].voice_feature_vector, self.feature_db[j].voice_feature_vector, 0))  # 不同
        return pairs

    def __len__(self) -> int:
        """
        返回数据集中配对的数量。
        
        :return: 数据集中配对的数量。
        """
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple:
        """
        获取指定索引处的特征对和标签。
        
        :param idx: 特征对的索引。
        :return: 包含两个输入特征向量和对应标签的元组。
        """
        input1, input2, label = self.pairs[idx]
        return input1, input2, label


class Database:
    def __init__(self) -> None:
        """
        数据库类，用于存储学生的面部特征和语音特征，并提供训练和识别功能。
        """
        self.feature_db = []  # 存储特征的条目
        self.face_siamese_model = SiameseNetwork(input_dim=512)  # 面部特征孪生网络模型
        self.voice_siamese_model = SiameseNetwork(input_dim=192)  # 语音特征孪生网络模型
        self.face_optimizer = optim.Adam(self.face_siamese_model.parameters(), lr=0.001)  # 面部模型优化器
        self.voice_optimizer = optim.Adam(self.voice_siamese_model.parameters(), lr=0.001)  # 语音模型优化器
        self.loss_fn = ContrastiveLoss(margin=1.0)  # 对比损失函数

    def store_feature(self, name: str, face_feature_vector: torch.Tensor, voice_feature_vector: torch.Tensor) -> None:
        """
        存储学生的面部特征和语音特征。
        
        :param name: 学生的姓名。
        :param face_feature_vector: 学生的面部特征向量。
        :param voice_feature_vector: 学生的语音特征向量。
        """
        self.feature_db.append(FeatureEntry(name, face_feature_vector, voice_feature_vector))

    def train_face_siamese_model(self, num_epochs: int = 10) -> None:
        """
        训练面部特征识别的孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        dataset = FeaturePairDataset(self.feature_db, feature_type='face')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            self.face_siamese_model.train()
            total_loss = 0
            for input1, input2, label in dataloader:
                self.face_optimizer.zero_grad()
                output1, output2 = self.face_siamese_model(input1, input2)  # 前向传播
                loss = self.loss_fn(output1, output2, label)  # 计算损失
                loss.backward()  # 反向传播
                self.face_optimizer.step()  # 更新权重
                total_loss += loss.item()

            print(f"面部识别训练 Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}")

    def train_voice_siamese_model(self, num_epochs: int = 10) -> None:
        """
        训练语音特征识别的孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        dataset = FeaturePairDataset(self.feature_db, feature_type='voice')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            self.voice_siamese_model.train()
            total_loss = 0
            for input1, input2, label in dataloader:
                self.voice_optimizer.zero_grad()
                output1, output2 = self.voice_siamese_model(input1, input2)  # 前向传播
                loss = self.loss_fn(output1, output2, label)  # 计算损失
                loss.backward()  # 反向传播
                self.voice_optimizer.step()  # 更新权重
                total_loss += loss.item()

            print(f"语音识别训练 Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader)}")

    def recognize_face(self, face_feature_vector: torch.Tensor, threshold=0.5) -> str:
        """
        识别输入的面部特征向量并返回相应的姓名。
        
        :param face_feature_vector: 输入的面部特征向量。
        :return: 识别到的姓名。
        """
        self.face_siamese_model.eval()
        min_distance = float('inf')
        recognized_name = "未知"
        
        for entry in self.feature_db:
            stored_face_vector = entry.face_feature_vector
            output1, output2 = self.face_siamese_model(face_feature_vector, stored_face_vector)
            distance = F.pairwise_distance(output1, output2).item()
            if distance < min_distance:
                min_distance = distance
                recognized_name = entry.name
        
        return recognized_name
        '''
        if min_distance < threshold:  # 如果最小距离小于阈值，认为匹配成功
            return recognized_name
        else:
            return None  # 如果没有找到匹配的学生
        '''
    
    def recognize_faces(self, face_features: list[torch.Tensor]) -> list[str]:
        """
        批量处理面部特征向量列表，返回所有匹配的学生姓名。
        :param face_features: 面部特征向量列表
        :return: 匹配的学生姓名列表
        """
        matched_names = []
        for face_feature in face_features:
            # 对每个面部特征向量进行识别
            matched_name = self.recognize_face(face_feature)
            matched_names.append(matched_name)
        
        return matched_names

    def recognize_voice(self, voice_feature_vector: torch.Tensor, threshold=0.5) -> str:
        """
        识别输入的语音特征向量并返回相应的姓名。
        
        :param voice_feature_vector: 输入的语音特征向量。
        :return: 识别到的姓名。
        """
        self.voice_siamese_model.eval()
        min_distance = float('inf')
        recognized_name = "未知"
        
        for entry in self.feature_db:
            stored_voice_vector = entry.voice_feature_vector
            output1, output2 = self.voice_siamese_model(voice_feature_vector, stored_voice_vector)
            distance = F.pairwise_distance(output1, output2).item()
            if distance < min_distance:
                min_distance = distance
                recognized_name = entry.name
        
        return recognized_name
        '''
        if min_distance < threshold:  # 如果最小距离小于阈值，认为匹配成功
            return recognized_name
        else:
            return None  # 如果没有找到匹配的学生
        '''
'''
主程序调用思路：此代码仅为随机生成数据的例子，可以结合上面代码使用
def main():
    # 初始化数据库
    db = Database()

    # 模拟学生特征存储
    print("正在存储学生特征...")

    student_data = [
        ("Alice", torch.rand(512),torch.rand(192)),
        ("Bob", torch.rand(512),torch.rand(192)),
        ("Charlie", torch.rand(512),torch.rand(192)),
    ]
    
    for name, face_vector, voice_vector in student_data:
        db.store_feature(name, face_vector, voice_vector)
    print("学生特征存储完成！")

    # 训练脸部识别孪生网络模型
    print("正在训练脸部识别孪生网络模型...")
    db.train_face_siamese_model(num_epochs=5)
    print("脸部识别模型训练完成！")

    # 训练语音识别孪生网络模型
    print("正在训练语音识别孪生网络模型...")
    db.train_voice_siamese_model(num_epochs=5)
    print("语音识别模型训练完成！")

    # 脸部识别
    print("开始批量面部识别...")
    test_face_vectors = [torch.rand(512) for _ in range(5)]
    recognized_faces = db.recognize_faces(test_face_vectors)
    print(f"批量面部识别结果：{recognized_faces}")

    # 语音识别
    print("开始语音识别测试...")
    test_voice_vector = torch.rand(192)
    recognized_voice = db.recognize_voice(test_voice_vector)
    print(f"语音识别结果：{recognized_voice}")

if __name__ == "__main__":
    main()
'''
