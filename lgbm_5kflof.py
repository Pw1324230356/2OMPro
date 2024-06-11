import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score, matthews_corrcoef, recall_score, precision_score
from sklearn.feature_selection import SelectFromModel
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical as labelEncoding
from sklearn.metrics import confusion_matrix, matthews_corrcoef, precision_score, recall_score, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
import numpy as np
from model_tensorflow import ourmodel
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn import svm
import sys

for i in range(1,6):
    data3 = np.load(f'data_5kflod/data_npz/{i}/3mer.npz')
    a3 = np.mean(data3['x_train'],axis=1)
    b3 = np.mean(data3['x_test'],axis=1)

    df_train = pd.read_csv(f"data_5kflod/5kflod_data/{i}/train.tsv", sep='\t').iloc[:,3:]
    df_train = df_train.values

    df_test = pd.read_csv(f"data_5kflod/5kflod_data/{i}/test.tsv", sep='\t').iloc[:,3:]
    df_test =df_test.values

    y_1 = pd.read_csv(f"data_5kflod/5kflod_data/{i}/train.tsv", sep='\t').iloc[:,1]
    y1 = y_1.values
    y_2 = pd.read_csv(f"data_5kflod/5kflod_data/{i}/test.tsv", sep='\t').iloc[:,1]
    y2 = y_2.values
    





    # train_X = np.mean(a3,axis=1)
    train_X = np.concatenate((a3,df_train), axis=1)
    train_y = y1

    # test_X = np.mean(b3,axis=1)
    test_X = np.concatenate((b3,df_test), axis=1)
    test_y  = y2

    # print(a3.shape)
    # print(b3.shape)
    # print(df_train.shape)
    # print(df_test.shape)
    # sys.exit()
    # # 建立预测模型，并预测
    model = lgb.LGBMClassifier(
                                num_leaves=20,
                                max_depth=15,
                                learning_rate=0.1,
                                n_estimators=200,
                               min_child_samples=10,
                               subsample=0.8,
                               colsample_bytree=0.7,
                               # reg_alpha=0.0,
                               # reg_lambda=0.1,
                               # scale_pos_weight=1.0
                               )


    # 相关参数的预测
    model.fit(train_X, train_y)
    result_proba = model.predict_proba(test_X)
    pd.DataFrame(result_proba).to_csv(f'data_5kflod/word2vec_data/{i}/{i}_result', index=False)


    print("ACC:", model.score(test_X, test_y))

    y_predict = model.predict_proba(test_X)
    auc_score = roc_auc_score(test_y, y_predict[:,1])
    print("auc_score:", auc_score)

    mcc = matthews_corrcoef(test_y, model.predict(test_X))
    print("MCC:", mcc)

    recall = recall_score(test_y, model.predict(test_X))
    print("RECALL:", recall)

    precision = precision_score(test_y, model.predict(test_X))
    print("precision:", precision)









# df_train = pd.read_csv(r"LGBM_435_feature")
# feature_order =df_train.columns
# df_train = df_train.values

# df_test = pd.read_csv(r"test/A.csv")
# df_test = df_test[feature_order]
# df_test =df_test.values


# data1 = np.load('data_npz/1mer.npz')
# a1 = np.mean(data1['x_train'],axis=1)
# b1 = np.mean(data1['x_test'],axis=1)
# # a1 = np.array(a1).reshape((2176,-1))
# # b1 = np.array(b1).reshape((934,-1))

# data3 = np.load('data_npz/3mer.npz')
# a3 = np.mean(data3['x_train'],axis=1)
# b3 = np.mean(data3['x_test'],axis=1)


# data4 = np.load('data_npz/4mer.npz')
# a4 = np.mean(data4['x_train'],axis=1)
# b4 = np.mean(data4['x_test'],axis=1)


# data5 = np.load('data_npz/5mer.npz')
# a5 = data5['x_train']
# b5 = data5['x_test']
# a5 = np.mean(data5['x_train'],axis=1)
# b5 = np.mean(data5['x_test'],axis=1)
# # print(a1.shape)
# # print(b1.shape)
# y_1 = pd.read_csv('old_data/train/A_label.csv').to_numpy()
# y1 = labelEncoding(y_1, dtype=int)
# y1 = y1[:, 1]
# y_2 = pd.read_csv('old_data/test/A_label.csv').to_numpy()
# y2 = labelEncoding(y_2,dtype=int)
# y2 = y2[:, 1]





# # train_X = np.mean(a3,axis=1)
# train_X = np.concatenate((a3,df_train), axis=1)
# train_y = y1

# # test_X = np.mean(b3,axis=1)
# test_X = np.concatenate((b3,df_test), axis=1)
# test_y  = y2






# # # 使用交叉验证进行最优参数的获取

# # clf = lgb.LGBMClassifier()
# # param_grid = {
# #     'num_leaves': [10,20,50,100],
# #     'max_depth': [5, 10, 15,30,50],
# #     # 'learning_rate': [0.01, 0.1, 0.2],
# #     'n_estimators': [100, 200,500,1000],
# #     'min_child_samples': [10, 20, 30],
# #     'subsample': [0.8, 0.9, 1.0],
# #     'colsample_bytree': [0.7, 0.8, 0.9],
# #     # 'reg_alpha': [0.0, 0.1, 0.2],
# #     # 'reg_lambda': [0.0, 0.1, 0.2],
# #     # 'scale_pos_weight': [1.0, 2.0, 3.0]
# # }
# # scoring = {
# #     'accuracy': make_scorer(accuracy_score),
# #     'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
# #     'mcc': make_scorer(matthews_corrcoef),
# #     'recall': make_scorer(recall_score),
# #     'precision': make_scorer(precision_score)
# # }
# # grid_search = GridSearchCV(clf, param_grid, scoring=scoring, cv=5, refit='accuracy')
# # grid_search.fit(train_X, train_y)
# # # 获取不同性能指标的值
# # accuracy = grid_search.cv_results_['mean_test_accuracy'][grid_search.best_index_]
# # roc_auc = grid_search.cv_results_['mean_test_roc_auc'][grid_search.best_index_]
# # mcc = grid_search.cv_results_['mean_test_mcc'][grid_search.best_index_]
# # recall = grid_search.cv_results_['mean_test_recall'][grid_search.best_index_]
# # precision = grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]

# # print("Best Parameters:", grid_search.best_params_)
# # print("Best Score (Accuracy):", grid_search.best_score_)
# # print("Best Score (roc_auc):", roc_auc)
# # print("Best Score (mcc):", mcc)
# # print("Best Score (recall):", recall)
# # print("Best Score (precision):", precision)




