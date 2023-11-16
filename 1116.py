from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.metrics import confusion_matrix, f1_score
# 导入必要的库
import pandas as pd
from util import *
import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')

# 你的代码

# 假设 read_data 返回的是DataFrame
train_dataset = read_data('train.json')

# 分词，并将分词结果存储在字典中
train_tokenized_data = token_data(train_dataset.values)

# 加载保存的Word2Vec模型
loaded_model = Word2Vec.load("word2vec_model202311091711.model")

MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM=300
# 将文本数据转换为向量表示
padded_features_vector = get_features_vectors(train_tokenized_data, loaded_model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

# 创建包含所有向量的DataFrame
vector_df = pd.DataFrame(padded_features_vector.reshape(padded_features_vector.shape[0], -1))

# 选取id和label_id列
id_label_df = train_dataset[['id', 'label_id']]

# 按顺序合并DataFrame
combined_df = pd.concat([id_label_df, vector_df], axis=1)

# 划分数据集
test_size = 0.2
train_set, test_set = split_dataset(combined_df, test_size=test_size)

# 数据分割
X_train, y_train = train_set.drop(['id', 'label_id'], axis=1), train_set['label_id']
X_test, y_test = test_set.drop(['id', 'label_id'], axis=1), test_set['label_id']

print('X_train', X_train.shape)
print('y_train', y_train.shape)
print('X_test', X_test.shape)
print('y_test', y_test.shape)

# 初始化模型
model = Sequential()
model.add(LSTM(64, input_shape=(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))) # 移除return_sequences=True以使用单层LSTM
model.add(Dropout(0.5))
model.add(Dense(36, activation='softmax'))  # 假设有36个类别

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


model.summary()

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
