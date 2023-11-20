# 导入必要的库
import sys
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn import linear_model
from sklearn import naive_bayes
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, accuracy_score,f1_score
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import LabelEncoder
from mealpy.evolutionary_based import GA
from joblib import dump, load
import numpy as np
import random
random.seed(7)
np.random.seed(42)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from util import *

train_set = pd.read_csv('ShuffleSplit_train_set1117.csv')
test_set  = pd.read_csv('ShuffleSplit_test_set1117.csv')


print(train_set.shape)
print(test_set.shape)

# X_train, y_train  X_test  y_test
# 分离训练集中的特征和标签
X_train = train_set.drop(['id', 'label_id'], axis=1)
y_train = train_set['label_id']

# 分离测试集中的特征和标签
X_test = test_set.drop(['id', 'label_id'], axis=1)
y_test = test_set['label_id']


# # 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义和训练模型
def train_and_optimize_model_old(X_train, y_train):
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
    grid_search_xgb.fit(X_train, y_train)
    best_xgb_model = grid_search_xgb.best_estimator_

    # CatBoostClassifier 的参数网格
    catboost_model = CatBoostClassifier(verbose=0)
    param_grid_cat = {
        'iterations': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 1],
        'depth': [4, 6, 8]
    }
    grid_search_cat = GridSearchCV(estimator=catboost_model, param_grid=param_grid_cat, cv=3, n_jobs=-1)
    grid_search_cat.fit(X_train, y_train)
    best_catboost_model = grid_search_cat.best_estimator_

    # AdaBoostClassifier 的参数网格
    adaboost_model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
    param_grid_ada = {
        'base_estimator__max_depth': [1, 2, 3],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1]
    }
    grid_search_ada = GridSearchCV(estimator=adaboost_model, param_grid=param_grid_ada, cv=3, n_jobs=-1)
    grid_search_ada.fit(X_train, y_train)
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

def train_and_optimize_model(X_train, y_train):
    # 直接定义模型参数
    xgb_model = XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.1)
    catboost_model = CatBoostClassifier(iterations=200, learning_rate=0.1, depth=6, verbose=0)
    adaboost_model = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=2),
        n_estimators=100,
        learning_rate=0.1
    )

    # 训练模型
    xgb_model.fit(X_train, y_train)
    catboost_model.fit(X_train, y_train)
    adaboost_model.fit(X_train, y_train)

    # 组合模型
    ensemble_model = VotingClassifier(
        estimators=[
            ('xgb_model', xgb_model),
            ('CatBoostClassifier', catboost_model),
            ('adaboost', adaboost_model),
        ], voting='soft', verbose=3
    )
    ensemble_model.fit(X_train, y_train)

    return ensemble_model



base_classifier = GradientBoostingClassifier(n_estimators=100, max_depth=9, random_state=1016, verbose=3)
# Create an AdaBoostClassifier and use the gradient lift tree as the base classifier
adaboost = AdaBoostClassifier(base_estimator=base_classifier, learning_rate=0.001, n_estimators=100, random_state=1016)
# Fit the AdaBoostClassifier model on the training set
# 训练模型
adaboost.fit(X_train, y_train)
# 保存模型
dump(adaboost, 'ada_model1117.joblib')
# 进行预测
y_pred = adaboost.predict(X_test)



# # 加载模型
# ada_model_loaded = load('ada_model.joblib')
# # 使用加载的模型进行预测
# y_pred = ada_model_loaded.predict(X_test)

# 性能评估
accuracy = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
print('Confusion Matrix:\n', cm)


