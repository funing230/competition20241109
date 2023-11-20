from transformers import TFBertForSequenceClassification, BertTokenizer
import pandas as pd
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from util import *

# # 加载训练集和测试集
# train_set = pd.read_csv('ShuffleSplit_train_set1117.csv')
# test_set  = pd.read_csv('ShuffleSplit_test_set1117.csv')
#
#

dataset = read_data('train.json')
# 划分数据集
test_size = 0.2
train_set, test_set = split_dataset(dataset, test_size=test_size)

# 提取特征和标签 =train_set.drop(['id', 'label_id'], axis=1)  test_set.drop(['id', 'label_id'], axis=1)
X_train =train_set.apply(lambda x: x['title'] + ' ' + x['assignee'] + ' ' + x['abstract'], axis=1)
y_train = train_set['label_id']
X_test = test_set.apply(lambda x: x['title'] + ' ' + x['assignee'] + ' ' + x['abstract'], axis=1)
y_test = test_set['label_id']


# 使用SMOTE进行过采样以解决类不平衡问题
# smote = SMOTE(k_neighbors=2)
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 初始化BERT Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 将文本数据转换为BERT的输入格式
train_encodings = tokenizer(X_train.tolist(), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(X_test.tolist(), truncation=True, padding=True, max_length=128)

# 创建TensorFlow数据集
train_dataset = tf.data.Dataset.from_tensor_slices((dict(train_encodings), y_train))
test_dataset = tf.data.Dataset.from_tensor_slices((dict(test_encodings), y_test))

# 初始化BERT模型
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=36)

# 编译模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_dataset.shuffle(1000).batch(32), epochs=3, batch_size=32)

# 评估模型
model.evaluate(test_dataset.batch(32))
