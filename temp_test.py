
import pandas as pd
from util import *
from gensim.models import Word2Vec

print("---------------------------------------------------------------------------")
train_set = pd.read_csv('train_set.csv')
test_set  = pd.read_csv('test_set.csv')


print(train_set.shape)
print(test_set.shape)


# 分离训练集中的特征和标签
train_features = train_set.drop(['id', 'label_id'], axis=1)
train_labels = train_set['label_id']

# 分离测试集中的特征和标签
test_features = test_set.drop(['id', 'label_id'], axis=1)
test_labels = test_set['label_id']
