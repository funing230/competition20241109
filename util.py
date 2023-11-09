import nltk
from jieba import cut
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import sklearn
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import jieba
import json

#读取数据json
def read_data(filename):
    data_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            data_list.append(data)

    # 将JSON数据转换为DataFrame
    dataset = pd.DataFrame(data_list)
    # dataset = dataset.values
    return dataset
#分解单词 title  assignee  abstract
def token_data(dataset):
    tokenized_data = []
    for item in dataset:
        # t1=item
        # t2=item[1]
        # t3=cut(item[1])
        title_tokens = list(cut(item[1]))  # 分词标题
        assignee_tokens = list(cut(item[2]))  # 分词专利权人
        abstract_tokens = list(cut(item[3]))  # 分词摘要
        tokenized_data.append({
            # "id":item[0],
            "title": title_tokens,
            "assignee": assignee_tokens,
            "abstract": abstract_tokens,
            # "label_id":item[4]
        })
    return tokenized_data
#数据清洗
def process_data(messages):
    with open('stop_words.txt', 'r', encoding='utf-8') as file:
        chinese_stop_words = set([line.strip() for line in file])
    processed_corpus = []
    for text in messages:
        # 去除中文标点符号
        text = re.sub(r'[^\w\s]', '', text)
        # 中文分词
        words = jieba.cut(text)
        # 过滤停用词并拼接成字符串
        processed_text = ' '.join([word for word in words if word not in chinese_stop_words])
        # 将处理后的文本添加到列表中
        if processed_text != '':
            processed_corpus.append(processed_text)
        # processed_corpus.append([item for item in processed_text if len(item) != 0])
    return processed_corpus
# 训练word2vec模型
def train_w2v(tokenized_data):
    sentences=[process_data(item["title"]) + process_data(item["assignee"]) + process_data(item["abstract"]) for item in tokenized_data]
    # 清除句子中所有的空字符串
    cleaned_nested_list = [[word for word in sublist if word != ''] for sublist in sentences]
    # 训练word2vec模型
    model = Word2Vec(sentences=cleaned_nested_list,
                     vector_size=300, window=5, min_count=1, workers=4)
    # 假设你的Word2Vec模型是model
    model.save("word2vec_model202311091711.model")

# 将汉字文本转换为Word2Vec向量
def text_to_vectors(text, model):
    vectors = []
    for word in text:
        if word in model.wv:
            vectors.append(model.wv[word])
        else:
            vectors.append([0.0] * model.vector_size)  # 用零向量代替未登录词
    return vectors

# 汉字通过word2vec数字化
# def get_features_vectors(tokenized_data,loaded_model,MAX_SEQUENCE_LENGTH):
#     features_vectors = []
#
#     for item in tokenized_data:
#         # Combine processed data
#         combined_data = process_data(item["title"]) + process_data(item["assignee"]) + process_data(item["abstract"])
#         features_vectors.append(text_to_vectors(combined_data, loaded_model))
#
#     padded_features_vector = pad_sequences(features_vectors, maxlen=MAX_SEQUENCE_LENGTH, padding='post',truncating='post')
#
#     return padded_features_vector

def get_features_vectors(tokenized_data, loaded_model, MAX_SEQUENCE_LENGTH):
    features_vectors = []
    max_length = 0  # 初始化最长长度为0
    max_data = None  # 初始化最长的数据为None

    for item in tokenized_data:
        # Combine processed data
        combined_data = process_data(item["title"]) + process_data(item["assignee"]) + process_data(item["abstract"])
        combined_length = len(combined_data)  # 计算结合后数据的长度
        features_vectors.append(text_to_vectors(combined_data, loaded_model))

        # 检查当前字段长度是否是最长的
        if combined_length > max_length:
            max_length = combined_length
            max_data = item  # 保存最长的数据项

    padded_features_vector = pad_sequences(features_vectors, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # 打印最长字段的信息
    print("最长字段的长度:", max_length)
    print("最长字段的数据:", max_data)

    return padded_features_vector
