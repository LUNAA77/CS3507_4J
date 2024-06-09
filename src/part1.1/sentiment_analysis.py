import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments

np.random.seed(42)

data_path = './data/washed_data.csv'
# data_path = 'data/twitter/unwashed_data.csv'
df = pd.read_csv(data_path)

# to numpy
X = df['datas'].values
y = df['labels'].values
print(type(X))
print('length of X:', len(X))

shuffle_index = np.random.permutation(len(X))
X, y = X[shuffle_index], y[shuffle_index]

n_samples = 10000

X = X[:n_samples]
y = y[:n_samples]
print('length of X:', len(X))

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# to list
X_train = X_train.tolist()
X_test = X_test.tolist()


# model_path = 'model/gpt2'

# 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2)

# padding
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

print("Tokenizer pad token:", tokenizer.pad_token)
print("Model pad token ID:", model.config.pad_token_id)


# 数据预处理
train_encodings = tokenizer(
    X_train, truncation=True, padding=True, max_length=512)
val_encodings = tokenizer(X_test, truncation=True,
                          padding=True, max_length=512)

# 准备数据集


class GPT2Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = GPT2Dataset(train_encodings, y_train)
val_dataset = GPT2Dataset(val_encodings, y_test)

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=3,              # 训练轮次
    per_device_train_batch_size=8,   # 每个设备的批次大小
    per_device_eval_batch_size=16,   # 每个设备的评估批次大小
    warmup_steps=500,                # 预热步数
    learning_rate=3e-5,              # 学习率
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
    report_to="none"
)

# 初始化训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 开始训练
trainer.train()

# 保存模型
# model_path = 'models/gpt2-twitter'
# model.save_pretrained(model_path)
# tokenizer.save_pretrained(model_path)


# 预测
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
print('length of test data:', len(y_pred))

# # print文本和标签
# for i in range(5):
#     print(f"Text: {X_test[i]}...")
#     print(f"True label: {y_test[i]}")
#     print(f"Predicted label: {y_pred[i]}")
#     print()

# 评估
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='binary')
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1: {f1}")
