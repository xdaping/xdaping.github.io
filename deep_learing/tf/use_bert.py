#!/user/bin/env python
# encoding=utf-8
# @Author: daping
# @Date: 2024/8/4 18:14


import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

# 加载预训练的BERT模型和分词器
# 缓存目录 /Users/daping/.cache/huggingface/hub/models--bert-base-chinese
#tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#model = TFBertModel.from_pretrained('bert-base-chinese')

tokenizer = BertTokenizer.from_pretrained('/Users/daping/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')
model = TFBertModel.from_pretrained('/Users/daping/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f')

# 待处理的文本
# sentences = "今天天气真好。"
# inputs = tokenizer(sentences, return_tensors="tf")

sentences = ["今天天气真好。", "明天可能会下雨，带伞出门吧。"]
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="tf")

inputs["da"] = "poing"

print("inputs \n", inputs)

#inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
# 使用BERT模型对编码后的文本进行处理
outputs = model(inputs)
# 获取编码后的文本的最后一层隐藏状态
last_hidden_states = outputs.last_hidden_state


print(last_hidden_states.shape, last_hidden_states)

print(outputs.pooler_output.shape, outputs.pooler_output) # 聚合后的

flattened_output = tf.keras.layers.Flatten()(last_hidden_states)
print(flattened_output.shape)