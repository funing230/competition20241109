# 导入必要的库
import pandas as pd
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import jieba
import numpy as np

# 读取数据
def read_data(filename):
    return pd.read_json(filename, lines=True)

# 加载数据
train_dataset = read_data('train.json')

# Word2Vec 特征提取
def create_word2vec_features(dataset):
    sentences = [list(jieba.cut(text)) for text in dataset['title'] + ' ' + dataset['assignee'] + ' ' + dataset['abstract']]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    word_vectors = model.wv

    def get_average_word2vec(tokens_list, vector, generate_missing=False, k=100):
        if len(tokens_list) < 1:
            return np.zeros(k)
        if generate_missing:
            vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
        else:
            vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
        length = len(vectorized)
        summed = np.sum(vectorized, axis=0)
        averaged = np.divide(summed, length)
        return averaged

    def get_word2vec_embeddings(vectors, clean_sentences, generate_missing=False):
        embeddings = clean_sentences.apply(lambda x: get_average_word2vec(x, vectors,
                                                                        generate_missing=generate_missing))
        return list(embeddings)

    w2v_embeddings = get_word2vec_embeddings(word_vectors, sentences)
    return np.array(w2v_embeddings)

X = create_word2vec_features(train_dataset)
y = train_dataset['label']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 过采样处理
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 定义和训练模型
def train_and_optimize_model(X_train, y_train):
    # # 定义模型
    # xgb_model = XGBClassifier()
    # catboost_model = CatBoostClassifier(verbose=0)
    # adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    #
    # # 定义参数网格
    # param_grid = {
    #     'base_estimator__max_depth': [1, 2, 3],  # DecisionTreeClassifier的参数
    #     'n_estimators': [50, 100, 200],
    #     'learning_rate': [0.01, 0.1, 1]
    # }
    # # 网格搜索
    # grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1)
    # grid_search_xgb.fit(X_train_resampled, y_train_resampled)
    # best_xgb_model = grid_search_xgb.best_estimator_
    #
    # # CatBoostClassifier 的参数网格
    # catboost_model = CatBoostClassifier(verbose=0)
    # param_grid_cat = {
    #     'iterations': [100, 200, 300],
    #     'learning_rate': [0.01, 0.1, 1],
    #     'depth': [4, 6, 8]
    # }
    # grid_search_cat = GridSearchCV(estimator=catboost_model, param_grid=param_grid_cat, cv=3, n_jobs=-1)
    # grid_search_cat.fit(X_train_resampled, y_train_resampled)
    # best_catboost_model = grid_search_cat.best_estimator_

    # AdaBoostClassifier 的参数网格
    adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    param_grid_ada = {
        'base_estimator__max_depth': [1, 2, 3],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    grid_search_ada = GridSearchCV(estimator=adaboost_model, param_grid=param_grid_ada, cv=3, n_jobs=-1)
    grid_search_ada.fit(X_train_resampled, y_train_resampled)
    best_adaboost_model = grid_search_ada.best_estimator_
    # 组合模型
    ensemble_model = VotingClassifier(
        estimators=[
            # ('xgb_model', best_xgb_model),
            # ('CatBoostClassifier', best_catboost_model),
            ('adaboost', best_adaboost_model),
        ], voting='soft', verbose=3
    )
    ensemble_model.fit(X_train, y_train)

    return ensemble_model

# 训练模型
model = train_and_optimize_model(X_train_resampled, y_train_resampled)

# 进行预测
y_pred = model.predict(X_test)

# 性能评估
accuracy = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion Matrix:\n', cm)
