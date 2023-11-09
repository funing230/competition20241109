from util import *
from gensim.models import Word2Vec

train_dataset = read_data('train.json')
# 分词，并将分词结果存储在字典中
train_tokenized_data = token_data(train_dataset)

#-----------第一次使用时打开------------------------------
# 训练Word2Vec模型
# train_w2v(tokenized_data)
#-----------第一次使用时打开------------------------------

# 加载保存的Word2Vec模型
loaded_model = Word2Vec.load("word2vec_model202311091711.model")

# 汉字通过word2vec数字化 Pad the feature vectors
MAX_SEQUENCE_LENGTH = 500
padded_features_vector=get_features_vectors(train_tokenized_data,loaded_model,MAX_SEQUENCE_LENGTH)

# Add any specific print statements if necessary
print(padded_features_vector)