import pandas as pd 
import os
import numpy as np
from sklearn.metrics import precision_recall_curve, confusion_matrix, matthews_corrcoef, auc
from sklearn.metrics import roc_curve, roc_auc_score
import statistics
import sys

i = 1
file_path = f'5kflod_fine_tune/{i}mer'
file_names = os.listdir(file_path)



ACC = []
AUC = []
MCC = []
Recall = []
Mcc = []
# 循环遍历文件名
for name in file_names:
    if len(name)==1:
       df1 = pd.read_csv(f'5kflod_fine_tune/{i}mer/{name}/test_results.tsv', delimiter='\t').iloc[:,1]
       df2 = pd.read_csv(f'5kflod/{i}mer/{name}/test.tsv', delimiter='\t').iloc[:,0]
