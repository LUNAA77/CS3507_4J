import datasets
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
train_data, test_data = datasets.load_dataset('imdb', split=['train', 'test'], cache_dir='data/')

# 将训练数据转换为Pandas DataFrame
train_df = pd.DataFrame(train_data)

# 计算文本长度
train_df['text_length'] = train_df['text'].apply(len)

# 绘制文本长度的直方图
plt.figure(figsize=(10, 6))
plt.hist(train_df['text_length'], bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Text Lengths in IMDB Train Dataset')
plt.xlabel('Text Length')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 保存
plt.savefig('text_length_distribution.png')