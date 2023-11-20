from util_new import *
from gensim.models import Word2Vec
import thulac


# thu = thulac.thulac()
train_dataset = read_excel('corpus.xlsx')
# 分词，并将分词结果存储在字典中
train_tokenized_data = token_data_corpus(train_dataset.values)
EMBEDDING_DIM=100
#-----------第一次使用时打开------------------------------
# 训练Word2Vec模型
train_w2v(train_tokenized_data,EMBEDDING_DIM)
#-----------第一次使用时打开------------------------------

