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
from sklearn.model_selection import StratifiedShuffleSplit


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

def read_excel(filename):
    fname = "corpus.xlsx"  # 请确保文件路径正确
    dataset = pd.read_excel(fname, sheet_name='Sheet1')
    # time_index = np.array(rawData['专利名称'])  # 月份时间索引
    # fault_counts = np.array(rawData['简要说明'])  # 累积故障数量
    return dataset[['专利名称','专利名称']]




#分解单词 title  assignee  abstract
def token_data_corpus(dataset):
    tokenized_data = []
    for item in dataset:
        title_tokens = list(cut(item[0]))  # 分词标题
        abstract_tokens = list(cut(item[1]))  # 分词摘要
        tokenized_data.append({
            "title": title_tokens,
            "abstract": abstract_tokens,
        })
    return tokenized_data





#分解单词 title  assignee  abstract
def token_data(dataset,thu):
    tokenized_data = []
    for item in dataset:
        # t1=item
        # t2=item[1]
        # t3=cut(item[1])
        title_tokens = list(thu.cut(item[1], text=True))  # 分词标题
        assignee_tokens = list(thu.cut(item[2], text=True))  # 分词专利权人
        abstract_tokens = list(thu.cut(item[3], text=True))  # 分词摘要
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
        words = cut(text)
        # 过滤停用词并拼接成字符串
        processed_text = ' '.join([word for word in words if word not in chinese_stop_words])
        # 将处理后的文本添加到列表中
        if processed_text != '':
            processed_corpus.append(processed_text)
        # processed_corpus.append([item for item in processed_text if len(item) != 0])
    return processed_corpus

# 训练word2vec模型
def train_w2v(tokenized_data,EMBEDDING_DIM):
    sentences=[process_data(item["title"]) + process_data(item["abstract"]) for item in tokenized_data]
    # 清除句子中所有的空字符串
    cleaned_nested_list = [[word for word in sublist if word != ''] for sublist in sentences]
    # 训练word2vec模型
    model = Word2Vec(sentences=cleaned_nested_list,
                     vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=4)
    # 假设你的Word2Vec模型是model
    model.save("jieba_word2vec_corpus_model20231117.model")
