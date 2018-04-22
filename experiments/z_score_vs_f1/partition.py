from sklearn.utils import shuffle

from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
# df = pd.DataFrame(np.random.random((10,5)), columns=list('ABCDE'))
df=pd.read_csv("tweets_3k_annotated.csv",sep =',')
df = shuffle(df)
# df.to_csv
# df.to_csv('shuffled.csv' ,sep=',',   encoding='utf-8')

kf = KFold(n_splits=5,random_state=1000) #random split, different each time
for train_ind,test_ind in kf.split(df):
    print(train_ind,test_ind)
    # print(df.iloc[train_ind])
    df.iloc[train_ind].to_csv('shuffled3.csv' ,sep=',',   encoding='utf-8')
    # df.iloc[test_ind]
    print("**********************")
    # xtrain = df[train_ind:]
    # xtest = df[test_ind:]
