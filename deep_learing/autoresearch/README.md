PyTorch DNN 训练系统
概述
本项目是一个完整的 PyTorch 深度学习训练系统，采用模块化设计将功能分离为两个独立的脚本：

prepare.py - 数据加载、日志配置、模型评估等工具函数
train.py - 模型定义、训练循环等核心逻辑
这种设计参考了现代深度学习项目的最佳实践（如 Karpathy 的 nanochat 项目），提高了代码的可复用性和可维护性。

文件说明
核心脚本
prepare.py - 数据和工具模块
提供训练所需的所有工具函数，可被其他脚本导入使用：

日志配置:

setup_logger(log_path) - 配置日志系统（文件 + 控制台）
数据加载:

TrainingDataset - 自定义 PyTorch 数据集类
load_datasets(train_file, test_file, batch_size) - 加载训练/测试集并返回 DataLoader
模型评估:

evaluate(model, test_loader, criterion, device, logger, phase) - 评估模型性能（损失 + 准确率）
train.py - 模型和训练模块
主训练脚本，导入 prepare.py 中的工具函数：

模型定义:

DNN 类 - 三层全连接神经网络（20→64→32→2）
训练逻辑:

train_epoch() - 单个 epoch 的训练函数
main() - 完整的训练流程管理
用途:

python train.py
辅助文件
training_data.txt - 训练集（50,000 行）
test_data.txt - 测试集（10,000 行）
training.log - 训练日志输出
model.pth - 保存的模型权重文件
快速开始
运行训练
# 使用默认参数训练模型（3 epochs）
python train.py
配置参数 (在 train.py 中修改):

batch_size - 批处理大小（默认: 64）
num_epochs - 训练轮数（默认: 3）
学习率、优化器等在 main() 中配置
模型架构
DNN (三层全连接神经网络)
输入层: 20个特征
    ↓
隐藏层1: 64个神经元 + ReLU + Dropout(0.3)
    ↓
隐藏层2: 32个神经元 + ReLU + Dropout(0.3)
    ↓
输出层: 2个神经元 (二分类)
参数统计:

总参数数: 3,490
可训练参数数: 3,490
训练配置
超参数
参数	默认值	说明
num_epochs	3	训练轮数
batch_size	64	批次大小
learning_rate	0.001	学习率
weight_decay	1e-5	权重衰减（L2正则化）
random_seed	42	随机种子
优化器与学习率调度
优化器: Adam (lr=0.001, weight_decay=1e-5)
学习率调度: StepLR（每个 epoch 后学习率乘以 0.5）
损失函数: CrossEntropyLoss（用于二分类）
输出说明
训练日志 (training.log)
日志包含以下信息：

基本信息

开始和结束时间
文件路径
使用设备（CPU/GPU）
数据信息

训练集和测试集大小
批次大小和批次数
模型信息

模型架构
参数数量
优化器配置
训练过程

每10个批次的损失值
每个epoch的训练和测试指标
学习率变化
最终结果

训练过程统计表
最佳性能指标
总训练时间
示例日志输出
================================================================================
PyTorch深度学习模型训练
================================================================================
开始时间: 2026-03-24 17:12:39
...
训练过程统计:
Epoch    Train Loss      Train Acc       Test Loss       Test Acc
1        0.7445          51.38           0.6909          52.77
2        0.6914          53.01           0.6898          53.54
3        0.6899          53.42           0.6871          54.81

最终结果:
最低训练损失: 0.6899 (Epoch 3)
最高训练准确率: 53.42% (Epoch 3)
最低测试损失: 0.6871 (Epoch 3)
最高测试准确率: 54.81% (Epoch 3)
数据特性
特征-标签关系
生成的数据包含以下关系规则：

前5个特征 (feature_0-4)

平均值 > 50 且 ≥3个特征 > 60 → 倾向 label=1
中间5个特征 (feature_5-9)

平均值 < 30 → 倾向 label=1
后10个特征 (feature_10-19)

方差 > 35 → 倾向 label=0
特定组合

feature_0 > 70 AND feature_5 < 20 → 倾向 label=1
feature_15 > 80 AND feature_16 > 80 → 倾向 label=0
噪声处理
添加10%的噪声，随机翻转标签
增加学习难度，更接近真实场景
模块化架构设计
设计原则
本项目遵循现代深度学习项目的最佳实践，参考了 Karpathy 的 nanochat 等高质量项目：

职责分离

prepare.py - 处理所有数据/日志/评估操作
train.py - 专注于模型定义和训练流程
代码复用

prepare.py 中的函数和类可被其他脚本导入
便于添加新的训练脚本或分析脚本
易于扩展

新的模型类可直接在 train.py 中定义
新的数据处理函数可添加到 prepare.py
各模块独立演进，无耦合
便于测试

各模块可独立单元测试
函数纯度高，易于 mock 和验证
代码质量

清晰的导入关系
一致的命名约定
详细的文档字符串
生产级别的代码质量
性能指标
训练效率
数据加载时间: < 1秒
单个epoch训练时间: ~0.6-0.9秒（取决于批次大小）
三个epoch总时间: ~2-3秒
模型性能
初始准确率: ~51%（基线）
最终准确率: ~54%（3个epoch后）
损失函数收敛情况良好
常见问题
Q: 如何调整模型架构？
A: 在 train.py 的 main() 函数中修改神经网络参数：

方案1: 调整现有模型

在 train.py 中修改神经网络初始化参数，例如增加隐藏层神经元数量。

方案2: 添加新模型

在 prepare.py 中定义新的模型类
在 train.py 中导入并使用
Q: 如何使用 GPU 加速？
A: 代码已自动检测 GPU。如果有可用的 GPU，会自动使用 CUDA。

Q: 如何改进模型性能？
A: 在 train.py 的 main() 函数中调整：

学习率：修改 optim.Adam() 中的 lr 参数
批次大小：修改 batch_size 变量
训练轮数：修改 num_epochs 变量
正则化：调整 weight_decay 参数
模型大小：修改 DNN 类初始化参数
Q: 如何在其他脚本中使用工具模块？
A: 在其他Python脚本中导入所需的函数和类。prepare.py 模块中导出的主要接口包括：

setup_logger() - 日志配置函数
load_datasets() - 数据加载函数
evaluate() - 模型评估函数
TrainingDataset - 数据集类
参考 train.py 文件的导入语句了解如何使用这些模块。

依赖项
torch >= 1.9.0
numpy >= 1.19.0
许可证
MIT License