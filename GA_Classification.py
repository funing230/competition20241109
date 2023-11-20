
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
# import threading
# from wrapt_timeout_decorator import timeout
from numpy import array, reshape
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from multiprocessing import freeze_support
# timeout_seconds = 10
import json
from sklearn.preprocessing import StandardScaler
import warnings

# 屏蔽所有警告
warnings.filterwarnings('ignore')

def count_calls(func):
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        return func(*args, **kwargs)

    wrapper.calls = 0
    return wrapper

class TimeoutException(Exception):
    pass

def timeout_handler(event):
    print("Timeout! Exiting the statement execution.")
    event.set()

class HybridMlp:
    def __init__(self,  epoch, pop_size):  #dataset,
        self.X_train, self.X_test, self.Y_train, self.Y_test = None, None, None, None
        # self.n_hidden_nodes = n_hidden_nodes
        self.epoch = epoch
        self.pop_size = pop_size
        self.model, self.problem, self.optimizer, self.solution, self.best_fit = None, None, None, None, None
        self.n_dims, self.n_inputs = None, None
        self.data=None
        # self.dataset=dataset
        self.term_dict = None

    def create_problem(self):
        # LABEL ENCODER
        SVC_C_ENCODER = LabelEncoder()
        SVC_C_ENCODER.fit(['0.001','0.01', '0.1', '1.0',])
        PENALTY_ENCODER = LabelEncoder()
        PENALTY_ENCODER.fit(['l1', 'l2', 'none'])
        SOLVER_ENCODER = LabelEncoder()
        SOLVER_ENCODER.fit(['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])

        DATA = {}
        DATA["SVC_C_ENCODER"] = SVC_C_ENCODER
        DATA["PENALTY_ENCODER"] = PENALTY_ENCODER
        DATA["SOLVER_ENCODER"] = SOLVER_ENCODER

        LB = [4,    0,     1,       0,       0.00001,      0,       50,      50,    0.00001,  3,
              1,    0.5,   0.5,     10 ,     0.00001,      1,       0.01   , 2,     10,      0.001]
        UB = [10,   3,     7.389,    3,      0.0001,       5,       200,     200,   0.1,     10,
              10,   1,     1 ,      100,     0.1,          10,      1  ,     15,     100,     0.1]
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "max",
            "log_to": None,
            "obj_weights": [1],
            "save_population": False,

        }
        self.term_dict = {
            # When creating this object, it will override the default epoch you define in your model
            "mode": "MG",
            "quantity": 100  # 1000 epochs
        }
        self.data = DATA
        return self.problem

    def decode_solution(self,solution, data):

        # XGboost
        max_depth =  int(solution[0])
        # SVC_C
        svc_c_integer=int(solution[1])
        opt_svc_c =float(data["SVC_C_ENCODER"].inverse_transform([svc_c_integer])[0])
        # RidgeClassifier
        alpha = solution[2]
        penalty_integer=int(solution[3])
        penalty =data["PENALTY_ENCODER"].inverse_transform([penalty_integer])[0]
        C_log=solution[4]
        solver_integer=int(solution[5])
        solver =data["SOLVER_ENCODER"].inverse_transform([solver_integer])[0]
        max_iter=int(solution[6])
        # XGBClassifier
        XGB_n_estimators=int(solution[7])
        XGB_learning_rate=solution[8]
        XGB_max_depth=int(solution[9])
        XGB_min_child_weight=int(solution[10])
        XGB_subsample=solution[11]
        XGB_colsample_bytree=solution[12]
        # CatBoostClassifier
        iterations_cat=int(solution[13])
        learning_rate_cat=solution[14]
        depth_cat=int(solution[15])
        l2_leaf_reg_cat=solution[16]
        # adaboost
        depth_ada = int(solution[17])
        n_estimators_ada = int(solution[18])
        learning_rate_ada = solution[19]

        return {
            # XGboost
            "max_depth": max_depth,
            # SVC_C
            "opt_svc_c": opt_svc_c,
            # RidgeClassifier
            "alpha": alpha,
            "penalty": penalty,
            "C_log": C_log,
            "solver": solver,
            "max_iter": max_iter,
            # XGBClassifier
            "XGB_n_estimators": XGB_n_estimators,
            "XGB_learning_rate": XGB_learning_rate,
            "XGB_max_depth":XGB_max_depth,
            "XGB_min_child_weight": XGB_min_child_weight,
            "XGB_subsample": XGB_subsample,
            "XGB_colsample_bytree": XGB_colsample_bytree,
            # CatBoostClassifier
            "iterations_cat": iterations_cat,
            "learning_rate_cat": learning_rate_cat,
            "depth_cat": depth_cat,
            "l2_leaf_reg_cat": l2_leaf_reg_cat,
            #adaboost
            "n_estimators_ada":n_estimators_ada,
            "learning_rate_ada": learning_rate_ada,
            "depth_ada": depth_ada
        }

    @count_calls
    # @timeout(20)
    def fitness_function(self, solution):

        try :
            structure = self.decode_solution(solution, self.data)

            # rc = linear_model.RidgeClassifier(alpha=structure['alpha'], class_weight='balanced', solver='auto',probability=True)
            #
            # sv = svm.SVC(C=structure['opt_svc_c'], kernel='linear', decision_function_shape='ovr',probability=True)

            # lr = LogisticRegression(penalty=structure['penalty'],
            #                         C=structure['C_log'],
            #                         solver='lbfgs',
            #                         max_iter=structure['max_iter'],
            #                         multi_class='multinomial',
            #                         verbose=3)
            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                num_class=36,  # 设置类别数量
                n_estimators=structure['XGB_n_estimators'],
                learning_rate=structure['XGB_learning_rate'],
                max_depth=structure['XGB_max_depth'],
                min_child_weight=structure['XGB_min_child_weight'],
                subsample=structure['XGB_subsample'],
                colsample_bytree=structure['XGB_colsample_bytree'],
                random_state=42
            )
            CatBoost = CatBoostClassifier(iterations=structure['iterations_cat'],
                                          learning_rate=structure['learning_rate_cat'],
                                          depth=structure['depth_cat'],
                                          l2_leaf_reg=structure['l2_leaf_reg_cat']
                                          )

            adaboost = AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=structure['depth_ada']),
                n_estimators=structure['n_estimators_ada'],
                learning_rate=structure['learning_rate_ada'],
                random_state=1016,)

            ensemble_model = VotingClassifier(
                estimators=[
                    # ('lr', lr),
                    # ('rc', rc),
                    # ('sv', sv),
                    ('xgb_model', xgb_model),
                    ('CatBoostClassifier', CatBoost),
                    ('adaboost', adaboost),
                ],voting='soft',verbose=3
            )

            # timeout_event = threading.Event()
            # timer_thread = threading.Timer(timeout_seconds, timeout_handler, args=(timeout_event,))
            # # timer_thread.daemon = True
            # timer_thread.start()
            #
            ensemble_model.fit(X_train, y_train)
            # CatBoost.fit(X_train, y_train)
            # timer_thread.cancel()
            # if timeout_event.is_set():
            #     raise TimeoutException("Function execution timed out")
            y_pred = ensemble_model.predict(X_test)
            # y_pred=CatBoost.predict(X_test)
            accuracy = f1_score(y_test, y_pred,average='macro')


            structure.update({'-------------------accuracy----------------------------------%%%%%%':accuracy})

            serialized_data = json.dumps(structure)
            with open("./model/data_file_all_six_0726.json", "a") as file:
                file.write(serialized_data + "\n")

            print('-----{' + str(self.fitness_function.calls) + '}---------', -1)

            dump(ensemble_model, "./model/" + str(accuracy) +'_all_six_0726.joblib')

            return accuracy

        except TimeoutError:
            print('-----{' + str(self.fitness_function.calls) + '}---------', -1)
            return -1
        except Exception as e:
            print('-----{' + str(self.fitness_function.calls) + '}---------', -1)
            print("except:", e)

            return -1

    def prediction_value(self,solution):
        structure = self.decode_solution(solution, self.data)
        n_steps = int(structure["n_steps"])



    def training(self):
        self.optimizer = GA.BaseGA(epoch=self.epoch, pop_size=self.pop_size,pc=0.8,pm=0.2)
        self.solution, self.best_fit = self.optimizer.solve(self.create_problem())  #funing 20220817
        # self.optimizer = GA.BaseGA(self.create_problem(),epoch=self.epoch, pop_size=self.pop_size)
        # self.solution, self.best_fit = self.optimizer.solve()  #funing 20220817

    def best_fitness(self):
        accuracy= self.best_fit
        print('-------------------------------------')
        print("accuracy : " + str(accuracy))
        print('-------------------------------------')

    def best_model(self):  #    DS3_0.8628637164174774
        structure = self.decode_solution(self.solution, self.data)
        print("------FINALLY------------structure---------------------------")






class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
	    self.terminal = stream
	    self.log = open(filename, 'a')

    def write(self, message):
	    self.terminal.write(message)
	    self.log.write(message)

    def flush(self):
	    pass

sys.stdout = Logger("./model/_0726.log", sys.stdout)
sys.stderr = Logger("./model/test_error.log", sys.stderr)


# freeze_support()




train_set = pd.read_csv('train_set1116.csv')
test_set  = pd.read_csv('test_set1116.csv')

print(train_set.shape)
print(test_set.shape)

# X_train, y_train  X_test  y_test
# 分离训练集中的特征和标签
X_train_ = train_set.drop(['id', 'label_id'], axis=1)
y_train = train_set['label_id']

# 分离测试集中的特征和标签
X_test_ = test_set.drop(['id', 'label_id'], axis=1)
y_test = test_set['label_id']

# 数据转换示例
def convert_string_to_array(str_series):
    return np.array([np.fromstring(x[1:-1], sep=' ') for x in str_series])

X_train_converted = X_train_ #convert_string_to_array(X_train_.iloc[:, 0])
X_test_converted = X_test_ #convert_string_to_array(X_test_.iloc[:, 0])

# 特征缩放
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_converted)
X_test = scaler.transform(X_test_converted)

# # 分离训练集中的特征和标签
# train_features = train_set.drop('label', axis=1)
# train_labels = train_set['label']
#
# # 分离测试集中的特征和标签
# test_features = test_set.drop('label', axis=1)
# test_labels = test_set['label']



# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.values)
# X_test = scaler.transform(X_test.values)


## Create hybrid model
epoch = 200
pop_size = 10
i = 0
model = HybridMlp(epoch, pop_size) #dataset,
model.training()
model.best_model()
model.best_fitness()








