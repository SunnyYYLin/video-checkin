import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class SiameseNetwork(nn.Module):
    """
    孪生网络模型，处理192维语音特征向量。
    """
    def __init__(self, input_dim=192):
        super(SiameseNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)  # 第一层全连接层
        self.fc2 = nn.Linear(1024, 512)       # 第二层全连接层，输出512维嵌入特征

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        对一个输入进行前向传播。
        
        :param x: 输入张量（192维的特征向量）。
        :return: 处理后的输出张量（512维的嵌入特征）。
        """
        x = F.relu(self.fc1(x))  # 第一个全连接层 + ReLU 激活
        x = self.fc2(x)          # 第二个全连接层
        return x

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> tuple:
        """
        对两个输入进行前向传播，返回它们的嵌入特征。
        
        :param input1: 第一个输入张量（192维的特征向量）。
        :param input2: 第二个输入张量（192维的特征向量）。
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
    def __init__(self, name: str, voice_feature_vector: torch.Tensor):
        """
        特征条目类，存储学生的姓名和语音特征向量。
        
        :param name: 学生的姓名。
        :param voice_feature_vector: 学生的语音特征向量。
        """
        self.name = name
        self.voice_feature_vector = voice_feature_vector

class FeaturePairDataset(Dataset):
    def __init__(self, feature_db: list[FeatureEntry]) -> None:
        """
        特征配对数据集类，用于训练孪生网络。
        
        :param feature_db: 存储所有学生语音特征的数据库。
        """
        self.feature_db = feature_db
        self.pairs = self.create_pairs()  # 创建特征对

    def create_pairs(self) -> list[tuple]:
        """
        创建特征对，用于孪生网络训练。
        
        :return: 特征对列表，包含特征向量对和标签。
        """
        pairs = []
        for i in range(len(self.feature_db)):
            for j in range(i + 1, len(self.feature_db)):
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
        数据库类，用于存储学生的语音特征，并提供训练和识别功能。
        """
        self.feature_db = []  # 存储语音特征的条目
        self.voice_siamese_model = SiameseNetwork()  # 语音孪生网络模型
        self.voice_optimizer = optim.Adam(self.voice_siamese_model.parameters(), lr=0.001)  # 语音模型优化器
        self.loss_fn = ContrastiveLoss(margin=1.0)  # 对比损失函数

    def store_feature(self, name: str, voice_feature_vector: torch.Tensor) -> None:
        """
        存储学生的语音特征。
        
        :param name: 学生的姓名。
        :param voice_feature_vector: 学生的语音特征向量。
        """
        self.feature_db.append(FeatureEntry(name, voice_feature_vector))
        # print(f"成功存储 {name} 的语音特征向量。")

    def train_voice_siamese_model(self, num_epochs: int = 10) -> None:
        """
        训练语音特征识别的孪生网络模型。
        
        :param num_epochs: 训练的轮数。
        """
        dataset = FeaturePairDataset(self.feature_db)
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

    def recognize_voice(self, input_voice_vector: torch.Tensor, threshold=0.5) -> str:
        """
        使用训练好的语音孪生网络模型进行语音识别。
        :param input_voice_vector: 输入的语音特征向量。
        :param threshold: 判定是否相似的阈值。
        :return: 如果识别成功，返回识别到的学生姓名，否则返回 None。
        """
        self.voice_siamese_model.eval()  # 设置模型为评估模式
        min_distance = float('inf')
        recognized_name = None

        for entry in self.feature_db:  # 遍历数据库中的所有特征条目
            output = self.voice_siamese_model(input_voice_vector, entry.voice_feature_vector)  # 计算欧氏距离
            distance = F.pairwise_distance(output[0], output[1]).item()

            if distance < min_distance:
                min_distance = distance
                recognized_name = entry.name
        
        return recognized_name

def main():
    # 初始化数据库
    db = Database()

    # 模拟学生特征存储
    print("正在存储学生特征...")

    student_data = [
        ("Alice", torch.rand(192)),
        ("Bob", torch.rand(192)),
        ("Charlie", torch.rand(192)),
    ]

    for name, voice_vector in student_data:
        db.store_feature(name, voice_vector)
    print("学生特征存储完成！")

    # 训练语音识别孪生网络模型
    print("正在训练语音识别孪生网络模型...")
    db.train_voice_siamese_model(num_epochs=5)
    print("语音识别模型训练完成！")

    # 语音识别
    print("开始语音识别测试...")
    test_voice_vector = torch.rand(192)
    recognized_voice = db.recognize_voice(test_voice_vector)
    print(f"语音识别结果：{recognized_voice}")

if __name__ == "__main__":
    main()