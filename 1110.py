import pandas as pd
from util import *
from gensim.models import Word2Vec


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

print(train_set.shape)
print(test_set.shape)

print("-------------------------Begin--------------------------------------------------")

train_set.to_csv('train_set.csv', index=False)
test_set.to_csv('test_set.csv', index=False)


print("--------------------------End-------------------------------------------------")