#!/usr/bin/env python
#-- coding: utf-8 --
"""
生成训练样本TXT文件
特征数：20个
标签：0/1
样本数：50000行
"""

import os
import random
import numpy as np

def generate_training_data(output_path, num_samples=50000, num_features=20, noise_ratio=0.1):

    """
    生成训练样本，特征和标签之间有明确的关系

    关系定义:
    - 前5个特征(feature_0-4): 如果平均值 > 50 且其中至少3个 > 60, 倾向label=1
    - 中间5个特征(feature_5-9): 如果平均值 < 30, 倾向label=1
    - 后10个特征(feature_10-19): 如果方差 > 30, 倾向label=0
    - 综合得分决定最终标签，加入噪声增加难度

    Args:
        output_path: 输出文件路径
        num_samples: 样本数量（默认50000）
        num_features: 特征数量（默认20）
        noise_ratio: 噪声比例（默认10%，会随机翻转标签）
    """
    random.seed(42)
    np.random.seed(42)

    print(f"正在生成 {num_samples} 行训练样本...")
    print(f"特征数量: {num_features}")
    print(f"噪声比例: {noise_ratio*100}%")
    print(f"输出文件: {output_path}")

    with open(output_path, 'w') as f:
        # 写入头部
        feature_names = [f"feature_{i}" for i in range(num_features)]
        header = ",".join(feature_names) + ",label\n"
        f.write(header)

        # 生成样本数据
        for i in range(num_samples):
            # 生成20个特征，范围在 0-100 之间
            features = [round(random.uniform(0, 100), 2) for _ in range(num_features)]

            # 根据特征计算标签
            label = _calculate_label(features)

            # 加入噪声：以 noise_ratio 的概率翻转标签
            if random.random() < noise_ratio:
                label = 1 - label

            # 转换为字符串并写入一行数据
            feature_str = [str(f) for f in features]
            line = ",".join(feature_str) + "," + str(label) + "\n"
            f.write(line)

            # 每5000行打印进度
            if (i + 1) % 5000 == 0:
                print(f"  已生成: {i + 1}/{num_samples} 行")

    print(f"\n✓ 成功生成训练样本！")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"样本位置: {output_path}")

def _calculate_label(features):

    """
    根据特征计算标签的内部函数

    返回 0 或 1，基于多个特征规则的加权组合
    """
    score = 0.0

    # 规则1: 前5个特征的平均值和高值比例
    front_features = features[:5]
    front_avg = np.mean(front_features)
    high_count = sum(1 for f in front_features if f > 60)

    if front_avg > 50 and high_count >= 3:
        score += 0.4
    elif front_avg > 65:
        score += 0.3

    # 规则2: 中间5个特征的平均值
    middle_features = features[5:10]
    middle_avg = np.mean(middle_features)

    if middle_avg < 30:
        score += 0.3
    elif middle_avg < 45:
        score += 0.15

    # 规则3: 后10个特征的方差和最大最小差
    back_features = features[10:20]
    back_variance = np.var(back_features)
    back_range = max(back_features) - min(back_features)

    if back_variance > 35:
        score -= 0.2
    elif back_range > 60:
        score -= 0.1

    # 规则4: 特定特征的组合关系
    if features[0] > 70 and features[5] < 20:
        score += 0.2

    if features[15] > 80 and features[16] > 80:
        score -= 0.15

    # 将分数转换为概率，生成标签
    # 使用sigmoid函数将分数映射到 [0, 1]
    probability = 1 / (1 + np.exp(-score * 2))

    # 以该概率返回1，否则返回0
    label = 1 if random.random() < probability else 0

    return label

def verify_data(file_path):

    """验证生成的数据"""
    print("\n验证数据:")
    with open(file_path, 'r') as f:
        lines = f.readlines()
        print(f"  总行数: {len(lines)}")
        print(f"  头部信息: {lines[0].strip()}")
        print(f"  第一行样本: {lines[1].strip()}")
        print(f"  最后一行样本: {lines[-1].strip()}")

        # 检查标签分布
        labels = [line.strip().split(',')[-1] for line in lines[1:]]
        label_0_count = labels.count('0')
        label_1_count = labels.count('1')
        print(f"  标签分布: label=0: {label_0_count}, label=1: {label_1_count}")
        print(f"  标签比例: label=0: {label_0_count/(label_0_count+label_1_count)*100:.1f}%, label=1: {label_1_count/(label_0_count+label_1_count)*100:.1f}%")

if __name__ == "__main__":

    # 定义输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "training_data.txt")

    # 生成数据
    generate_training_data(output_file, num_samples=50000, num_features=20)

    # 验证数据
    verify_data(output_file)

    print("\n✓ 完成！")
