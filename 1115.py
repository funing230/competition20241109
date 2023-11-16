from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from sklearn.metrics import confusion_matrix, f1_score
from tensorflow import keras
# 导入必要的库
import pandas as pd
from util import *
import tensorflow as tf
# 假设 read_data 返回的是DataFrame
train_dataset = read_data('train.json')

# 分词，并将分词结果存储在字典中
train_tokenized_data = token_data(train_dataset.values)

# 加载保存的Word2Vec模型
loaded_model = Word2Vec.load("word2vec_model202311091711.model")

# 汉字通过word2vec数字化 Pad the feature vectors
MAX_SEQUENCE_LENGTH = 250
# 假设 padded_features_vector 是通过 get_features_vectors 获得的
padded_features_vector = get_features_vectors(train_tokenized_data, loaded_model, MAX_SEQUENCE_LENGTH)

# 将每个向量序列转换为列表，并确保它们保持原有顺序
vector_list = [list(vector) for vector in padded_features_vector]

# 创建一个包含所有向量的DataFrame
vector_df = pd.DataFrame(vector_list)

# 选取id和label_id列
id_label_df = train_dataset[['id', 'label_id']]

# 按顺序合并DataFrame
combined_df = pd.concat([id_label_df, vector_df], axis=1)

# print(combined_df.shape)
# print(combined_df.columns)
# print(combined_df.head(1))# -----------------------------------

test_size=0.2
train_set,test_set=split_dataset(combined_df,test_size=test_size)

# 数据分割
X_train,y_train, X_test, y_test =\
    train_set.drop(['id', 'label_id'], axis=1),\
    train_set['label_id'],\
    test_set.drop(['id', 'label_id'], axis=1),\
    test_set['label_id']


print('X_train',X_train.shape)
print('y_train',y_train.shape)
print('X_test',X_test.shape)
print('y_test',y_test.shape)

# 假设 EMBEDDING_DIM 和 MAX_SEQUENCE_LENGTH 已经设置
EMBEDDING_DIM = 300  # 根据Word2Vec模型的维度
MAX_SEQUENCE_LENGTH = 250

bugreport_input = keras.Input(shape=(X.shape[1],), name="bugreport")

features = Embedding(output_dim=embeddings_matrix.shape[1],
                     input_dim=embeddings_matrix.shape[0],
                     weights=[embeddings_matrix],
                     input_length=1500)(bugreport_input)

lstm_out = LSTM(128, dropout=0.3)(features)
hidden_x = Dense(32, activation='tanh')(lstm_out)
output = Dense(7, activation='softmax')(hidden_x)
model = keras.Model(inputs=[bugreport_input, complexity_input], outputs=output)
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              # metrics = ['accuracy'],)
              metrics=METRICS, )
model.summary()

X_final = X  # np.array(X)
y_final = np.array(Y_labels)



# No need for list comprehension if X_train and X_test are already correct shape
# X_train_list = [np.concatenate(list(row)) for index, row in X_train.iterrows()]
# X_train_array = np.array(X_train_list, dtype='float32')
# X_test_list = [np.concatenate(list(row)) for index, row in X_test.iterrows()]
# X_test_array = np.array(X_test_list, dtype='float32')
X_train_array = X_train.to_numpy().reshape(-1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
X_test_array = X_test.to_numpy().reshape(-1, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)

# Train the model
model.fit(X_train_array, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test_array, y_test)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Predictions
y_pred = model.predict(X_test_array)
y_pred_labels = np.argmax(y_pred, axis=1)

# Performance evaluation
accuracy = f1_score(y_test, y_pred_labels, average='macro')
cm = confusion_matrix(y_test, y_pred_labels)
print('F1 Score:', accuracy)
print('Confusion Matrix:\n', cm)