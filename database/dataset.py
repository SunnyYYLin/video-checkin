from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset

@dataclass
class FeatureEntry:
    """
    特征条目类，存储学生的姓名和特征向量。
    """
    name: str
    feature: torch.Tensor
        
@dataclass
class Student:
    """
    存储学生的姓名、面部特征向量和语音特征向量。
    
    :param name: 学生的姓名。
    :param face_feature: 学生的面部特征向量。
    :param voice_feature: 学生的语音特征向量。
    """
    name: str
    face_features: list[torch.Tensor] = field(default_factory=list)
    voice_features: list[torch.Tensor] = field(default_factory=list)
    face_prototype: torch.Tensor = None
    voice_prototype: torch.Tensor = None

class FeaturePairDataset(Dataset):
    def __init__(self, feature_entries: list[FeatureEntry]) -> None:
        """
        特征配对数据集类，用于训练孪生网络。
        
        :param feature_entries: 存储所有学生面部或语音特征
        """
        self.feature_entries = feature_entries

    def __len__(self) -> int:
        """
        返回数据集中配对的数量。
        
        :return: 数据集中配对的数量。
        """
        return len(self.feature_entries) * (len(self.feature_entries) + 1) // 2

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        获取指定索引处的特征对和标签。
        
        :param idx: 特征对的索引。
        :return: 包含两个输入特征向量和对应标签的元组。
        """
        assert 0 <= idx < len(self), "Index out of range"
        
        # 通过字典序公式动态解算 (idx1, idx2)
        idx1 = int((1 + (1 + 8 * idx)**0.5) // 2)
        idx2 = idx - idx1 * (idx1 - 1) // 2 + idx1
        
        feature1 = self.feature_entries[idx1].feature
        feature2 = self.feature_entries[idx2].feature
        label = int(self.feature_entries[idx1].name == self.feature_entries[idx2].name)
        return feature1, feature2, label
    
    @staticmethod
    def from_students(students: list[Student]|dict[Student], attr: str) -> "FeaturePairDataset":
        """
        从学生列表/字典中构建特征配对数据集。
        
        :param students: 学生列表/字典。
        :param attr: 特征类型（'face_features' 或 'voice_features'）。
        :return: 构建的特征配对数据集。
        """
        assert attr in ['face_features', 'voice_features'], "Invalid attribute"
        
        feature_entries = []
        for student in students.values() if isinstance(students, dict) else students:
            features = getattr(student, attr, [])
            feature_entries.extend([FeatureEntry(student.name, feature) for feature in features])
        return FeaturePairDataset(feature_entries)