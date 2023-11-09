import pandas as pd
import numpy as np
import sklearn
import json
from util import *
from gensim.models import Word2Vec
# 从JSON文件中逐个读取数据（使用utf-8编码）
data_list = []
with open('train.json', 'r', encoding='utf-8') as file:
    for line in file:
        data = json.loads(line)
        data_list.append(data)

# 将JSON数据转换为DataFrame
df = pd.DataFrame(data_list)

# 打印DataFrame
print(df)

# 假设你的分词结果为tokenized_sentences，是一个列表的列表，每个内部列表包含分好词的词语
model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)
