autoresearch
让 AI agent 自主优化深度学习策略。

业务背景
本项目是一个完整的 PyTorch 深度学习训练系统，包含模型训练和效果评估。系统采用模块化设计，核心代码分为两个Python文件（train.py 和 prepare.py），配合训练和测试数据集，用于神经网络的二分类任务。

Setup
所有 git 操作都在 master 分支上进行，无需创建新分支。
读取相关文件：
README.md or QUICKSTART.md — 该算法的背景框架介绍
prepare.py — 固定基础设施：数据加载、评估函数。不要修改。
train.py — 模型策略，你修改的唯一文件。
验证数据存在：确认 training_data.txt 和 test_data.txt 存在。training_data.txt是训练集，test_data.txt是测试集。
自动开始：无需用户确认，直接开始实验。
Experimentation
每次实验运行 python train.py。脚本加载数据、评估策略，输出关键指标。

What you CAN do:

修改 train.py — 这是你唯一编辑的文件。所有模型和优化相关逻辑都可以改：

超参数调整（HyperParameters）：所有超参数都可以修改，包括：
- batch_size: 批处理大小（32 ~ 128）
- num_epochs: 训练轮数（3 ~ 10）
- learning_rate: 学习率（0.0001 ~ 0.01）
- weight_decay: 权重衰减（1e-6 ~ 1e-4）
- hidden1, hidden2: 隐藏层大小

架构改进（Architecture）：
- 增加网络层数或改变层的大小
- 使用批归一化（BatchNormalization）
- 添加残差连接（Residual Connections）
- 调整激活函数（LeakyReLU 替代 ReLU）
- 增加 Dropout 比例到 0.5

What you CANNOT do:
修改 prepare.py。它包含固定的数据加载、日志系统，评估函数等。
修改评估指标（在 prepare.py 中）。
添加外部依赖（只能用标准库）。
Metric

指标: 最低训练损失（train loss）、最高训练准确率（train accuracy）、最低测试损失（test loss）、最高测试准确率（test accuracy）

脚本输出中可 grep 关键指标。脚本在训练完成后会输出详细报告，包含每个 epoch 的训练和测试结果。

当前输出格式示例：
```
最低训练损失: 0.6899 (Epoch 3)
最高训练准确率: 53.42% (Epoch 3)
最低测试损失: 0.6871 (Epoch 3)
最高测试准确率: 54.81% (Epoch 3)
```

Output format
脚本打印完整报告后，最终在日志中输出关键指标。可通过 grep 提取：

最低训练损失: 0.6899
最高训练准确率: 53.42
最低测试损失: 0.6871
最高测试准确率: 54.81
实验结束后自动记录到 results.tsv（tab-separated）。

重要说明：
- ✅ txt/log/pth 文件已在 .gitignore 中，完全不上传
- ✅ git 只上传 .py 和 .md 文件：git add src/test/ai/dnn/*.py src/test/ai/dnn/*.md
- ✅ 所有 git 操作自动执行，无需用户确认
- ✅ 只在性能改进时才提交和 push
- ✅ results.tsv 不提交，仅本地记录

results.tsv 格式说明：
commit	TrainLoss	TestLoss	TrainAcc   TestAcc	status	description
git commit hash (short, 7 chars)
TrainLoss (e.g. 0.689000) — use 999.000000 for crashes
TestLoss (e.g. 0.6891)
TrainAcc (e.g. 0.8312)
TestAcc (e.g. 0.5312)
status: keep, discard, or crash
short text description
Example:

commit	TrainLoss	TestLoss	TrainAcc    TestAcc	status	description
a1b2c3d	0.689000	0.689100	0.8312  0.5281	keep	baseline
b2c3d4e	0.685200	0.686300	0.83120.5312	keep	increase hidden layers to 128
c3d4e5f	0.698200	0.700100	0.83120.5150	discard	weight_decay too high 1e-3
d4e5f6g	999.000000	0.0000	0.8312  0.0000	crash	model initialization error
The experiment loop
LOOP FOREVER:

查看当前状态：确保在 master 分支上
修改 train.py：根据实验想法直接修改代码
运行实验：python train.py > run.log 2>&1
读取结果：自动从日志中提取关键指标（grep "最高训练准确率\|最高测试准确率\|最低测试损失" run.log）
如果 grep 输出为空，说明运行崩溃。自动读取日志末尾 50 行诊断问题。
记录结果：自动添加到 results.tsv（tab-separated，不提交）
判断结果：
  - 如果测试准确率提高或测试损失降低：
    自动提交：git add src/test/ai/dnn/*.py src/test/ai/dnn/*.md && git commit -m "experiment: <description>"
    自动推送：git push origin master
  - 如果测试准确率相同或更差：
    回滚提交：git reset --hard HEAD~1
    强制推送：git push -f origin master
Timeout: 每次实验应在 30 秒内完成。如果超过 2 分钟，自动 kill 并视为失败。

崩溃处理：如果是 typo 或小 bug，自动修复后重跑。如果是根本性问题，记 crash，revert，继续。

NEVER STOP: 实验循环开始后，不要暂停问用户。用户可能在睡觉。你是自主的。如果没灵感了，重读 program.md 和 prepare.py 找新思路，尝试组合之前差一点成功的实验，或者尝试更激进的改动。循环一直跑到用户手动中断。

Experiment ideas (starter list)

超参数微调（Hyperparameter Tuning）:
1. 学习率搜索：从 0.0001 到 0.01 逐步增加
2. 批次大小实验：尝试 32、64、128
3. 权重衰减调整：1e-6、1e-5、1e-4（防止过拟合）
4. Epoch 数增加：从 3 增加到 5、10 观察过拟合情况

结构性改动（Architecture Changes）:
1. 隐藏层调整：增加第三个隐藏层或扩大现有层（64->128）
2. 添加批归一化（BatchNormalization）在每个隐藏层后
3. 激活函数替换：LeakyReLU（alpha=0.2）替代 ReLU
4. Dropout 增加：将 0.3 增加到 0.5
5. 残差连接：在某些层添加跳过连接

组合实验（Combined Experiments）:
- 高学习率 + 高 Dropout（0.5）+ 小 batch size（32）
- 低学习率 + BatchNorm + 更多隐藏单元
- 多层网络 + 弱正则化


