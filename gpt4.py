# 导入必要的库
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# 读取数据
def read_data(filename):
    return pd.read_json(filename, lines=True)

# 加载数据
train_dataset = read_data('train.json')

# TF-IDF 特征提取
def create_tfidf_features(dataset):
    corpus = dataset['title'] + ' ' + dataset['assignee'] + ' ' + dataset['abstract']
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer.get_feature_names_out()

X, feature_names = create_tfidf_features(train_dataset)
y = train_dataset['label_id']

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查最小类别的样本数量
min_class_size = y_train.value_counts().min()
n_neighbors = min(5, min_class_size - 1)  # 确保 n_neighbors 小于最小类别的样本数

# 过采样处理
smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# 定义和训练模型
def train_and_optimize_model(X_train, y_train):
    # 定义模型
    xgb_model = XGBClassifier()
    # 定义参数网格
    param_grid = {
        'base_estimator__max_depth': [1, 2, 3],  # DecisionTreeClassifier的参数
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    # 网格搜索
    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, n_jobs=-1)
    grid_search_xgb.fit(X_train_resampled, y_train_resampled)
    best_xgb_model = grid_search_xgb.best_estimator_

    # CatBoostClassifier 的参数网格
    catboost_model = CatBoostClassifier(verbose=0)
    param_grid_cat = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 1],
        'depth': [4, 6, 8]
    }
    grid_search_cat = GridSearchCV(estimator=catboost_model, param_grid=param_grid_cat, cv=3, n_jobs=-1)
    grid_search_cat.fit(X_train_resampled, y_train_resampled)
    best_catboost_model = grid_search_cat.best_estimator_

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
            ('xgb_model', best_xgb_model),
            ('CatBoostClassifier', best_catboost_model),
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
