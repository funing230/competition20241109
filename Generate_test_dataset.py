from util import *

# 读取测试集数据
test_dataset = read_data('testA.json')
# 分词，并将分词结果存储在字典中
test_tokenized_data = token_data(test_dataset.values)
# 加载保存的Word2Vec模型
loaded_model = Word2Vec.load("jieba_word2vec_corpus_model20231117.model")
# 汉字通过word2vec数字化 Pad the feature vectors
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM=100
# 将文本数据转换为向量表示
padded_features_vector = get_features_vectors(test_tokenized_data, loaded_model, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
# 将每个向量序列转换为列表，并确保它们保持原有顺序
vector_list = [list(vector) for vector in padded_features_vector]
# 创建包含所有向量的DataFrame
vector_df = pd.DataFrame(padded_features_vector.reshape(padded_features_vector.shape[0], -1))
# 选取id和label_id列
id_label_df = test_dataset[['id']]
# 按顺序合并DataFrame
combined_df = pd.concat([id_label_df, vector_df], axis=1)
combined_df.to_csv('finall_test_dataset.csv',index=False)

print('---------------over--------------!!!!')