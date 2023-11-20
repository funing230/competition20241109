from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.metrics import confusion_matrix, f1_score
# 导入必要的库
import pandas as pd
from imblearn.over_sampling import SMOTE

from util import *
import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')

# 你的代码

# # 假设 read_data 返回的是DataFrame
# train_dataset = read_data('train.json')
#
# # 分词，并将分词结果存储在字典中
# train_tokenized_data = token_data(train_dataset.values)
#
# # 加载保存的Word2Vec模型
# loaded_model = Word2Vec.load("word2vec_model202311091711.model")
#
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM=100
# # 将文本数据转换为向量表示
# padded_features_vector = get_features_vectors(train_tokenized_data, loaded_model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
#
# # 创建包含所有向量的DataFrame
# vector_df = pd.DataFrame(padded_features_vector.reshape(padded_features_vector.shape[0], -1))
#
# # 选取id和label_id列
# id_label_df = train_dataset[['id', 'label_id']]
#
# # 按顺序合并DataFrame
# combined_df = pd.concat([id_label_df, vector_df], axis=1)
#
# # 划分数据集
# test_size = 0.2
# train_set, test_set = split_dataset(combined_df, test_size=test_size)
#
# # 数据分割
# X_train, y_train = train_set.drop(['id', 'label_id'], axis=1), train_set['label_id']
# X_test, y_test = test_set.drop(['id', 'label_id'], axis=1), test_set['label_id']
#
# print('X_train', X_train.shape)
# print('y_train', y_train.shape)
# print('X_test', X_test.shape)
# print('y_test', y_test.shape)
#


train_set = pd.read_csv('ShuffleSplit_train_set1117.csv')
test_set  = pd.read_csv('ShuffleSplit_test_set1117.csv')


print(train_set.shape)
print(test_set.shape)

# X_train, y_train  X_test  y_test
# 分离训练集中的特征和标签
X_train = train_set.drop(['id', 'label_id'], axis=1)
y_train = train_set['label_id']

# 分离测试集中的特征和标签
X_test = test_set.drop(['id', 'label_id'], axis=1)
y_test = test_set['label_id']

print("train Data Shape :",y_train.value_counts())

smote = SMOTE(k_neighbors=2)
X_resampled, y_resampled = smote.fit_resample(X_train,y_train)

print("Final Data Shape :",y_resampled.value_counts())



# # 初始化模型-------------------------------1
# model = Sequential()
# model.add(LSTM(64, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))) # 移除return_sequences=True以使用单层LSTM
# model.add(Dropout(0.5))
# model.add(Dense(36, activation='softmax'))  # 假设有36个类别
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.summary()
#------------------------------------------1
#------------------------------------------2
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.callbacks import EarlyStopping
#
# model = Sequential()
# model.add(LSTM(128, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)))  # 增加LSTM单元的数量
# model.add(Dropout(0.5))  # 维持dropout层以减少过拟合
# model.add(Dense(64, activation='relu'))  # 添加一个Dense层，使用ReLU激活函数
# model.add(Dense(36, activation='softmax'))  # 最终输出层保持不变
#
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # 设置早停法以防止过拟合
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
#
# model.summary()
#------------------------------------------2
#------------------------------------------3
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
#
# # 假设输入数据维度是100
# input_dim = 100
#
# model = Sequential()
# model.add(Dense(10, activation='relu', input_shape=(input_dim,)))  # 只有一个隐藏层，10个神经元
# model.add(Dense(1, activation='sigmoid'))  # 输出层，适用于二分类问题
#
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#
# model.summary()

#------------------------------------------3
#------------------------------------------4
from transformers import TFBertForSequenceClassification, BertTokenizer
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
model_name = 'bert-base-uncased'  # 可以根据需要选择不同的BERT模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=36)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
# model.fit(X_train_array, validation_data=val_dataset, epochs=3, callbacks=[early_stopping])

#------------------------------------------4

# 由于X_train的形状是(766, 250)，确保它与模型输入层兼容
X_train_array = X_train.to_numpy().reshape(-1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
X_test_array = X_test.to_numpy().reshape(-1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

# 训练模型
model.fit(X_train_array, y_train, epochs=500, batch_size=32, validation_split=0.2)

# 评估模型
loss, accuracy = model.evaluate(X_test_array, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# 预测和性能评估
y_pred = model.predict(X_test_array)
y_pred_labels = np.argmax(y_pred, axis=1)

accuracy = f1_score(y_test, y_pred_labels, average='macro')
cm = confusion_matrix(y_test, y_pred_labels)
print('F1 Score:', accuracy)
print('Confusion Matrix:\n', cm)
