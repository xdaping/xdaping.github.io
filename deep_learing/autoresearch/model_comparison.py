#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模型对比实验脚本
================
在同一数据集上对比多种机器学习和深度学习模型：

传统机器学习（PyTorch 实现）:
  1. 逻辑回归 (Logistic Regression)
  2. 梯度提升决策树 (Gradient Boosted Trees) - 简化版

深度学习模型 (PyTorch):
  3. 原始 DNN（三层全连接，baseline）
  4. 深层 ResNet（残差网络）
  5. 1D-CNN（一维卷积神经网络）
  6. Transformer Encoder（自注意力机制）
  7. MLP-Mixer（MLP混合架构）

使用: python model_comparison.py
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# 数据加载
# ============================================================================

def load_data(file_path):
    """从 txt 文件加载数据"""
    data = []
    labels = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            if line.strip():
                parts = line.strip().split(',')
                features = [float(x) for x in parts[:-1]]
                label = int(parts[-1])
                data.append(features)
                labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64)


def load_data_as_tensors(file_path):
    """加载数据并转为 PyTorch tensor"""
    X, y = load_data(file_path)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def get_dataloaders(train_file, test_file, batch_size=32):
    """获取 DataLoader"""
    X_train, y_train = load_data_as_tensors(train_file)
    X_test, y_test = load_data_as_tensors(test_file)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False)

    return train_loader, test_loader, X_train, y_train, X_test, y_test


# ============================================================================
# 通用训练/评估函数
# ============================================================================

def train_model(model, train_loader, criterion, optimizer, device, num_epochs, logger, scheduler=None):
    """通用训练循环"""
    model.to(device)
    train_losses = []
    train_accs = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        if scheduler:
            scheduler.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accs.append(accuracy)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"    Epoch [{epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

    return train_losses, train_accs


def evaluate_model(model, test_loader, device, criterion=None):
    """评估模型，返回 (loss, accuracy)"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)

            if criterion:
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(test_loader) if criterion else 0.0
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# ============================================================================
# 模型 1: 逻辑回归 (Logistic Regression)
# ============================================================================

class LogisticRegression(nn.Module):
    """逻辑回归模型 - 线性基线模型"""

    def __init__(self, input_size=20, output_size=2):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)


# ============================================================================
# 模型 2: 原始 DNN (Baseline)
# ============================================================================

class DNNBaseline(nn.Module):
    """原始三层全连接神经网络（baseline）"""

    def __init__(self, input_size=20, hidden1=64, hidden2=32, output_size=2, dropout=0.35):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        x = self.fc3(x)
        return x


# ============================================================================
# 模型 3: 深层 ResNet（残差网络）
# ============================================================================

class ResidualBlock(nn.Module):
    """残差块"""

    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.relu(out)


class DeepResNet(nn.Module):
    """深层残差网络 - 4层残差块"""

    def __init__(self, input_size=20, hidden_dim=128, output_size=2, num_blocks=4, dropout=0.3):
        super().__init__()
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.output_layer(x)
        return x


# ============================================================================
# 模型 4: 1D-CNN（一维卷积神经网络）
# ============================================================================

class CNN1D(nn.Module):
    """1D 卷积神经网络 - 将特征视为序列进行卷积"""

    def __init__(self, input_size=20, output_size=2, channels=[32, 64, 128], kernel_size=3, dropout=0.3):
        super().__init__()
        self.input_size = input_size

        # 将输入特征 reshape 为 (batch, 1, input_size) 后进行1D卷积
        layers = []
        in_channels = 1
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # 自适应池化 + 全连接
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size),
        )

    def forward(self, x):
        # x: (batch, input_size) -> (batch, 1, input_size)
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = self.global_pool(x)
        x = self.fc(x)
        return x


# ============================================================================
# 模型 5: Transformer Encoder（自注意力机制）
# ============================================================================

class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerClassifier(nn.Module):
    """
    Transformer 编码器分类器
    将 20 维特征分成多个 token，通过 Transformer 编码后进行分类
    """

    def __init__(self, input_size=20, output_size=2, d_model=64, nhead=4,
                 num_layers=3, dim_feedforward=128, dropout=0.3, num_tokens=4):
        super().__init__()
        self.input_size = input_size
        self.num_tokens = num_tokens
        self.feature_per_token = input_size // num_tokens  # 每个 token 的特征数

        # 类别 token (CLS)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 特征嵌入: 将每个 token 的特征映射到 d_model 维
        self.feature_embed = nn.Linear(self.feature_per_token, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=num_tokens + 1)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_size),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 将输入特征分成 num_tokens 个 token
        x = x.view(batch_size, self.num_tokens, self.feature_per_token)
        x = self.feature_embed(x)

        # 添加 CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 添加位置编码
        x = self.pos_encoding(x)

        # Transformer 编码
        x = self.transformer_encoder(x)

        # 取 CLS token 的输出用于分类
        cls_output = x[:, 0, :]
        output = self.classifier(cls_output)
        return output


# ============================================================================
# 模型 6: MLP-Mixer（MLP混合架构）
# ============================================================================

class Transpose(nn.Module):
    """维度转置模块"""
    def __init__(self, dim0, dim1):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class MLPMixer(nn.Module):
    """
    MLP-Mixer 风格的分类器
    使用 token-mixing 和 channel-mixing MLP 替代注意力机制
    """

    def __init__(self, input_size=20, output_size=2, num_tokens=4, hidden_dim=64,
                 num_mixer_layers=4, dropout=0.3):
        super().__init__()
        self.num_tokens = num_tokens
        self.feature_per_token = input_size // num_tokens
        self.hidden_dim = hidden_dim

        # 特征嵌入
        self.feature_embed = nn.Linear(self.feature_per_token, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # Mixer 层
        self.mixer_layers = nn.ModuleList([
            MixerLayer(num_tokens, hidden_dim, dropout)
            for _ in range(num_mixer_layers)
        ])

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_tokens * hidden_dim),
            nn.Linear(num_tokens * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 分割为 token 序列并嵌入
        x = x.view(batch_size, self.num_tokens, self.feature_per_token)
        x = self.feature_embed(x)
        x = self.norm(x)

        # 通过 Mixer 层
        for mixer in self.mixer_layers:
            x = x + mixer(x)

        # 展平并分类
        x = x.view(batch_size, -1)
        return self.classifier(x)


class MixerLayer(nn.Module):
    """单个 Mixer 层: token-mixing + channel-mixing"""

    def __init__(self, num_tokens, hidden_dim, dropout):
        super().__init__()
        # Token-mixing: 对 token 维度做 MLP
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_mlp = nn.Sequential(
            Transpose(1, 2),  # (batch, hidden, tokens)
            nn.Linear(num_tokens, num_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_tokens, num_tokens),
            Transpose(1, 2),  # (batch, tokens, hidden)
        )

        # Channel-mixing: 对 hidden 维度做 MLP
        self.channel_norm = nn.LayerNorm(hidden_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, x):
        # Token-mixing
        x = x + self.token_mlp(self.token_norm(x))
        # Channel-mixing
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


# ============================================================================
# 模型 7: Wide & Deep 网络
# ============================================================================

class WideAndDeep(nn.Module):
    """
    Wide & Deep 学习模型
    - Wide 部分: 线性模型直接连接输入到输出（记忆能力）
    - Deep 部分: 深层全连接网络（泛化能力）
    """

    def __init__(self, input_size=20, output_size=2, deep_hidden=[256, 128, 64], dropout=0.3):
        super().__init__()
        # Wide 部分: 直接线性映射（带交叉特征）
        self.wide = nn.Linear(input_size, output_size)

        # Deep 部分: 深层网络
        deep_layers = []
        in_dim = input_size
        for hidden_dim in deep_hidden:
            deep_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        self.deep = nn.Sequential(*deep_layers)

        # 融合层
        self.combined = nn.Sequential(
            nn.Linear(output_size + deep_hidden[-1], deep_hidden[-1] // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(deep_hidden[-1] // 2, output_size),
        )

    def forward(self, x):
        wide_out = self.wide(x)
        deep_out = self.deep(x)
        combined = torch.cat([wide_out, deep_out], dim=1)
        return self.combined(combined)


# ============================================================================
# 简化版梯度提升决策树 (基于 PyTorch 的深度回归树集成)
# ============================================================================

class SimpleGradientBoosting:
    """
    简化版梯度提升（PyTorch 实现）
    使用浅层神经网络作为"弱学习器"（类似决策树的拟合能力）
    按梯度提升的方式迭代训练
    """

    def __init__(self, input_size=20, num_rounds=50, lr=0.1, hidden_dim=32, dropout=0.2):
        self.input_size = input_size
        self.num_rounds = num_rounds
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.trees = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _make_tree(self):
        """创建一个弱学习器（浅层网络）"""
        return nn.Sequential(
            nn.Linear(self.input_size, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 2),  # 二分类 logits
        ).to(self.device)

    def fit(self, X_train, y_train, X_test, y_test, logger):
        """训练梯度提升模型"""
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)

        criterion = nn.CrossEntropyLoss(reduction='none')
        batch_size = 256
        n_samples = X_train.size(0)

        # 初始预测
        logits = torch.zeros(n_samples, 2, device=self.device)
        logits.requires_grad = False

        train_losses = []
        test_accs = []

        for round_i in range(self.num_rounds):
            # 计算梯度（残差）
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                # 梯度: 对正确类别的负对数概率的梯度
                grad = probs.clone()
                grad.scatter_(1, y_train.unsqueeze(1), grad.gather(1, y_train.unsqueeze(1)) - 1)

            # 训练新的弱学习器拟合梯度
            tree = self._make_tree()
            optimizer = optim.Adam(tree.parameters(), lr=0.01, weight_decay=1e-4)
            tree.train()

            # 简单训练几步
            for step in range(20):
                idx = torch.randperm(n_samples, device=self.device)[:batch_size]
                pred = tree(X_train[idx])
                loss = nn.functional.mse_loss(pred, grad[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 更新 logits
            with torch.no_grad():
                tree.eval()
                update = tree(X_train)
                logits = logits - self.lr * update

                # 计算训练损失和测试准确率
                train_loss = criterion(logits, y_train).mean().item()
                train_losses.append(train_loss)

                test_logits = torch.zeros(X_test.size(0), 2, device=self.device)
                # 累加所有树的预测
                for t in self.trees:
                    test_logits = test_logits - self.lr * t(X_test)
                test_logits = test_logits - self.lr * tree(X_test)

                _, predicted = torch.max(test_logits.data, 1)
                test_acc = 100 * (predicted == y_test).sum().item() / y_test.size(0)
                test_accs.append(test_acc)

            self.trees.append(tree)

            if (round_i + 1) % 10 == 0 or round_i == 0:
                logger.info(f"    Round [{round_i+1}/{self.num_rounds}] TrainLoss: {train_loss:.4f}, TestAcc: {test_acc:.2f}%")

        # 最终测试损失
        final_test_logits = torch.zeros(X_test.size(0), 2, device=self.device)
        for t in self.trees:
            final_test_logits = final_test_logits - self.lr * t(X_test)
        final_test_loss = criterion(final_test_logits, y_test).mean().item()
        final_test_acc = test_accs[-1]
        final_train_loss = train_losses[-1]

        return final_train_loss, final_test_loss, final_test_acc

    def predict(self, X):
        """预测"""
        self.eval()
        X = X.to(self.device)
        logits = torch.zeros(X.size(0), 2, device=self.device)
        for t in self.trees:
            logits = logits - self.lr * t(X)
        _, predicted = torch.max(logits.data, 1)
        return predicted

    def eval(self):
        for t in self.trees:
            t.eval()


# ============================================================================
# 主实验函数
# ============================================================================

def run_experiment(model_name, model, train_loader, test_loader, device,
                   num_epochs, lr, weight_decay, logger, scheduler_type='step'):
    """运行单个模型实验"""
    logger.info(f"\n{'='*70}")
    logger.info(f"模型: {model_name}")
    logger.info(f"{'='*70}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 3), gamma=0.5)
    elif scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"参数量: {total_params:,}")
    logger.info(f"训练配置: epochs={num_epochs}, lr={lr}, weight_decay={weight_decay}")

    start_time = time.time()
    train_losses, train_accs = train_model(
        model, train_loader, criterion, optimizer, device, num_epochs, logger, scheduler
    )
    train_time = time.time() - start_time

    # 评估
    test_loss, test_acc = evaluate_model(model, test_loader, device, criterion)

    logger.info(f"\n  >>> 结果: TrainLoss={train_losses[-1]:.4f}, TestLoss={test_loss:.4f}, "
                f"TrainAcc={train_accs[-1]:.2f}%, TestAcc={test_acc:.2f}%")
    logger.info(f"  >>> 训练时间: {train_time:.2f}秒")

    return {
        'name': model_name,
        'params': total_params,
        'train_loss': train_losses[-1],
        'test_loss': test_loss,
        'train_acc': train_accs[-1],
        'test_acc': test_acc,
        'time': train_time,
    }


def main():
    """主函数"""
    import logging

    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(script_dir, 'training_data.txt')
    test_file = os.path.join(script_dir, 'test_data.txt')

    # 检查数据文件
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"错误: 数据文件不存在")
        print(f"  训练数据: {train_file}")
        print(f"  测试数据: {test_file}")
        sys.exit(1)

    # 设置日志
    logger = logging.getLogger('ModelComparison')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(script_dir, 'model_comparison.log'), mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)

    # 设置
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = 20

    logger.info("=" * 70)
    logger.info("多模型对比实验")
    logger.info("=" * 70)
    logger.info(f"设备: {device}")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 加载数据
    logger.info("\n加载数据...")
    train_loader, test_loader, X_train, y_train, X_test, y_test = get_dataloaders(
        train_file, test_file, batch_size=128
    )
    logger.info(f"训练集: {X_train.size(0)} 样本, 测试集: {X_test.size(0)} 样本")

    results = []

    # =========================================================================
    # 实验 1: 逻辑回归 (线性基线)
    # =========================================================================
    model = LogisticRegression(input_size=input_size)
    result = run_experiment(
        "逻辑回归 (Logistic Regression)", model,
        train_loader, test_loader, device,
        num_epochs=50, lr=0.01, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 2: 原始 DNN (Baseline)
    # =========================================================================
    model = DNNBaseline(input_size=input_size, hidden1=64, hidden2=32, dropout=0.35)
    result = run_experiment(
        "DNN Baseline (3层全连接)", model,
        train_loader, test_loader, device,
        num_epochs=15, lr=0.001, weight_decay=1e-6, logger=logger, scheduler_type='step'
    )
    results.append(result)

    # =========================================================================
    # 实验 3: 深层 ResNet
    # =========================================================================
    model = DeepResNet(input_size=input_size, hidden_dim=128, num_blocks=4, dropout=0.3)
    result = run_experiment(
        "深层 ResNet (4层残差)", model,
        train_loader, test_loader, device,
        num_epochs=30, lr=0.002, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 4: 1D-CNN
    # =========================================================================
    model = CNN1D(input_size=input_size, channels=[32, 64, 128], kernel_size=3, dropout=0.3)
    result = run_experiment(
        "1D-CNN (卷积神经网络)", model,
        train_loader, test_loader, device,
        num_epochs=30, lr=0.002, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 5: Transformer Encoder
    # =========================================================================
    model = TransformerClassifier(
        input_size=input_size, d_model=64, nhead=4, num_layers=3,
        dim_feedforward=128, dropout=0.3, num_tokens=4
    )
    result = run_experiment(
        "Transformer Encoder (自注意力)", model,
        train_loader, test_loader, device,
        num_epochs=20, lr=0.002, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 6: MLP-Mixer
    # =========================================================================
    model = MLPMixer(
        input_size=input_size, num_tokens=4, hidden_dim=64,
        num_mixer_layers=4, dropout=0.3
    )
    result = run_experiment(
        "MLP-Mixer (混合架构)", model,
        train_loader, test_loader, device,
        num_epochs=30, lr=0.002, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 7: Wide & Deep
    # =========================================================================
    model = WideAndDeep(
        input_size=input_size, deep_hidden=[256, 128, 64], dropout=0.3
    )
    result = run_experiment(
        "Wide & Deep (记忆+泛化)", model,
        train_loader, test_loader, device,
        num_epochs=30, lr=0.002, weight_decay=1e-5, logger=logger, scheduler_type='cosine'
    )
    results.append(result)

    # =========================================================================
    # 实验 8: 简化版梯度提升 (Gradient Boosting)
    # =========================================================================
    logger.info(f"\n{'='*70}")
    logger.info(f"模型: 简化版梯度提升 (Gradient Boosting)")
    logger.info(f"{'='*70}")

    gb = SimpleGradientBoosting(input_size=input_size, num_rounds=50, lr=0.1, hidden_dim=32)
    start_time = time.time()
    train_loss, test_loss, test_acc = gb.fit(X_train, y_train, X_test, y_test, logger)
    gb_time = time.time() - start_time

    # 训练准确率
    train_preds = gb.predict(X_train)
    train_acc = 100 * (train_preds == y_train).sum().item() / y_train.size(0)

    logger.info(f"\n  >>> 结果: TrainLoss={train_loss:.4f}, TestLoss={test_loss:.4f}, "
                f"TrainAcc={train_acc:.2f}%, TestAcc={test_acc:.2f}%")
    logger.info(f"  >>> 训练时间: {gb_time:.2f}秒")

    results.append({
        'name': '简化版梯度提升 (Gradient Boosting)',
        'params': sum(sum(p.numel() for p in t.parameters()) for t in gb.trees),
        'train_loss': train_loss,
        'test_loss': test_loss,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'time': gb_time,
    })

    # =========================================================================
    # 汇总结果
    # =========================================================================
    logger.info("\n\n" + "=" * 100)
    logger.info("实验结果汇总")
    logger.info("=" * 100)
    logger.info(f"{'排名':<5} {'模型':<40} {'参数量':<12} {'TrainLoss':<12} {'TestLoss':<12} {'TrainAcc':<12} {'TestAcc':<12} {'时间':<8}")
    logger.info("-" * 100)

    # 按测试准确率排序
    sorted_results = sorted(results, key=lambda x: (-x['test_acc'], x['test_loss']))

    for rank, r in enumerate(sorted_results, 1):
        marker = " ★" if rank <= 3 else ""
        logger.info(
            f"#{rank:<4} {r['name']:<38} {r['params']:<12,} "
            f"{r['train_loss']:<12.4f} {r['test_loss']:<12.4f} "
            f"{r['train_acc']:<11.2f}% {r['test_acc']:<11.2f}% {r['time']:<7.1f}s{marker}"
        )

    logger.info("\n" + "=" * 100)
    logger.info(f"最佳模型: {sorted_results[0]['name']}")
    logger.info(f"最佳测试准确率: {sorted_results[0]['test_acc']:.2f}%")
    logger.info(f"最佳测试损失: {sorted_results[0]['test_loss']:.4f}")
    logger.info("=" * 100)
    logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n✓ 多模型对比实验完成！详细日志: model_comparison.log")
    print(f"\n  排名  模型                                      测试准确率  测试损失")
    print(f"  {'='*75}")
    for rank, r in enumerate(sorted_results, 1):
        marker = " ★" if rank <= 3 else ""
        print(f"  #{rank:<4} {r['name']:<38} {r['test_acc']:<11.2f}% {r['test_loss']:<10.4f}{marker}")


if __name__ == "__main__":
    main()

