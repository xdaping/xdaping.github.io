#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PyTorch深度学习模型训练脚本
- 三层全连接神经网络
- 训练3个epoch
- 输出训练过程到log文件

使用: python train.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from prepare import setup_logger, load_datasets, evaluate


# ---------------------------------------------------------------------------
# 神经网络模型
# ---------------------------------------------------------------------------

class DNN(nn.Module):
    """三层全连接神经网络"""

    def __init__(self, input_size=20, hidden1=64, hidden2=32, output_size=2):
        """
        初始化神经网络

        Args:
            input_size: 输入特征数
            hidden1: 第一层隐藏层神经元数
            hidden2: 第二层隐藏层神经元数
            output_size: 输出大小（二分类=2）
        """
        super(DNN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.35)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.35)

        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        return x


# ---------------------------------------------------------------------------
# 训练函数
# ---------------------------------------------------------------------------

def train_epoch(model, train_loader, criterion, optimizer, device, logger):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        features = features.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(features)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"  Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main():
    """主训练函数"""

    # 设置路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, 'training_data.txt')
    test_file = os.path.join(script_dir, 'test_data.txt')
    log_file = os.path.join(script_dir, 'training.log')

    # 初始化日志
    logger = setup_logger(log_file)

    logger.info("=" * 80)
    logger.info("PyTorch深度学习模型训练")
    logger.info("=" * 80)
    logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"训练数据: {train_file}")
    logger.info(f"测试数据: {test_file}")
    logger.info(f"日志文件: {log_file}")

    # 检查文件是否存在
    if not os.path.exists(train_file):
        logger.error(f"训练文件不存在: {train_file}")
        sys.exit(1)

    if not os.path.exists(test_file):
        logger.error(f"测试文件不存在: {test_file}")
        sys.exit(1)

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 加载数据
    logger.info("\n加载数据...")
    batch_size = 96
    train_loader, test_loader, train_dataset, test_dataset = load_datasets(
        train_file, test_file, batch_size=batch_size
    )

    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    logger.info(f"Batch大小: {batch_size}")
    logger.info(f"训练批次数: {len(train_loader)}")
    logger.info(f"测试批次数: {len(test_loader)}")

    # 创建模型
    logger.info("\n初始化模型...")
    model = DNN(input_size=20, hidden1=64, hidden2=32, output_size=2)
    model.to(device)

    logger.info(f"模型架构:\n{model}")

    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"总参数数: {total_params:,}")
    logger.info(f"可训练参数数: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    logger.info(f"\n损失函数: CrossEntropyLoss")
    logger.info(f"优化器: Adam (lr=0.001, weight_decay=1e-5)")
    logger.info(f"学习率调度器: StepLR (step_size=1, gamma=0.5)")

    # 训练循环
    num_epochs = 10
    logger.info(f"\n开始训练 ({num_epochs} epochs)...")
    logger.info("=" * 80)

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        logger.info(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        logger.info("-" * 80)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, logger)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        logger.info(f"训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")

        # 测试
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, logger, phase='测试')
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 学习率调整
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"当前学习率: {current_lr:.6f}")

        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch耗时: {epoch_time:.2f}秒")

    # 训练统计
    total_time = time.time() - start_time

    logger.info("\n" + "=" * 80)
    logger.info("训练完成!")
    logger.info("=" * 80)
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"平均每个epoch耗时: {total_time / num_epochs:.2f}秒")

    logger.info("\n训练过程统计:")
    logger.info(f"{'Epoch':<8} {'Train Loss':<15} {'Train Acc':<15} {'Test Loss':<15} {'Test Acc':<15}")
    logger.info("-" * 80)

    for i in range(num_epochs):
        logger.info(
            f"epoch_{i + 1:<8} TrainLoss:{train_losses[i]:<15.4f} TrainAcc:{train_accuracies[i]:<15.2f} "
            f"TestLoss:{test_losses[i]:<15.4f} TestAcc:{test_accuracies[i]:<15.2f}"
        )

    # 最终结果
    logger.info("\n最终结果:")
    logger.info(f"最低训练损失: {min(train_losses):.4f} (Epoch {np.argmin(train_losses) + 1})")
    logger.info(f"最高训练准确率: {max(train_accuracies):.2f}% (Epoch {np.argmax(train_accuracies) + 1})")
    logger.info(f"最低测试损失: {min(test_losses):.4f} (Epoch {np.argmin(test_losses) + 1})")
    logger.info(f"最高测试准确率: {max(test_accuracies):.2f}% (Epoch {np.argmax(test_accuracies) + 1})")

    logger.info("\n" + "=" * 80)
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

    # 保存模型
    model_path = os.path.join(script_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"\n模型已保存: {model_path}")

    print(f"\n✓ 训练完成！日志已保存到: {log_file}")


if __name__ == "__main__":
    main()

