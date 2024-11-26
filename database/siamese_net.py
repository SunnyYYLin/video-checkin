import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

class SiameseNetwork(nn.Module):
    """
    孪生网络模型，用于处理不同的特征向量（面部特征或语音特征）。
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.3) -> None:
        super(SiameseNetwork, self).__init__()
        hidden_dim = (input_dim + output_dim) // 2
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层全连接层
        self.dropout = nn.Dropout(dropout)                 # Dropout 层
        self.fc2 = nn.Linear(hidden_dim, output_dim)       # 第二层全连接层

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """
        对一个输入进行前向传播。
        
        :param x: 输入张量（面部或语音特征向量）。
        :return: 处理后的输出张量（嵌入特征）。
        """
        x = F.relu(self.fc1(x))  # 第一个全连接层 + ReLU 激活
        x = self.dropout(x)       # Dropout 层
        x = self.fc2(x)          # 第二个全连接层
        x = F.normalize(x, p=2, dim=1)  # L2 归一化
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
    def __init__(self, margin: float = 0.5) -> None:
        """
        对比损失函数，用于孪生网络训练。
        
        :param margin: 用于区分相似和不同样本的相似度阈值。
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
        cosine_similarity = F.cosine_similarity(output1, output2, dim=1) # (batch_size,)
        loss = label * torch.pow(torch.clamp(self.margin - cosine_similarity, min=0.0), 2) + \
                (1 - label) * torch.pow(cosine_similarity, 2) # (batch_size,)
        return loss.mean()
    
def train(model: nn.Module,
          train_loader: DataLoader, 
          criterion: nn.Module = ContrastiveLoss(),
          num_epochs: int = 32,
          device=torch.device("cuda" if torch.cuda.is_available() else "cpu")) -> None:
    """
    训练孪生网络模型。
    
    :param model: 孪生网络模型。
    :param criterion: 损失函数。
    :param train_loader: 训练数据加载器。
    :param num_epochs: 训练轮数。
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), desc=f"Epoch {epoch + 1}"):
            inputs1, inputs2, labels = data
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs1, outputs2 = model(inputs1, inputs2)
            loss = criterion(outputs1, outputs2, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            positive_distances = F.cosine_similarity(outputs1, outputs2)[labels == 1]
            if len(positive_distances) > 0:  # 防止没有正样本时报错
                criterion.margin = max(0.5, positive_distances.mean().item()*1.2)
            
        print(f"Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}")