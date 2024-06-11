import numpy as np
import pandas as pd
from sklearn.utils import shuffle

df1 = pd.read_csv(r'/home/wpeng/ensemble_1_3_5/i2om_5mer_data/1train.tsv')
df = shuffle(df1) # 对DataFrame进行打乱操作
# print(df1)
# print(df)
df.to_csv('/home/wpeng/ensemble_1_3_5/i2om_5mer_data/train.tsv',index=False)