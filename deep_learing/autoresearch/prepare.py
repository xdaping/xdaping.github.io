#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据准备、日志配置、模型评估等工具函数
"""

import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

def setup_logger(log_path):
    """设置日志系统"""
    logger = logging.getLogger('Training')
    logger.setLevel(logging.DEBUG)

    # 文件处理器
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ---------------------------------------------------------------------------
# 数据集
# ---------------------------------------------------------------------------

class TrainingDataset(Dataset):
    """自定义数据集类"""

    def __init__(self, file_path):
        """
        初始化数据集

        Args:
            file_path: 数据文件路径
        """
        self.data = []
        self.labels = []

        with open(file_path, 'r') as f:
            lines = f.readlines()
            # 跳过表头
            for line in lines[1:]:
                if line.strip():
                    parts = line.strip().split(',')
                    features = [float(x) for x in parts[:-1]]
                    label = int(parts[-1])
                    self.data.append(features)
                    self.labels.append(label)

        self.data = np.array(self.data, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_datasets(train_file, test_file, batch_size=64):
    """
    加载训练和测试数据集

    Args:
        train_file: 训练数据文件路径
        test_file: 测试数据文件路径
        batch_size: 批处理大小

    Returns:
        train_loader, test_loader: 数据加载器
        train_dataset, test_dataset: 数据集
    """
    train_dataset = TrainingDataset(train_file)
    test_dataset = TrainingDataset(test_file)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset, test_dataset


# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, criterion, device, logger, phase='Test'):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    logger.info(f"{phase} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    return avg_loss, accuracy

