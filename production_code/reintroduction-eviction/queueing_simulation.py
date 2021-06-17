import datetime
from threading import Thread
import random
import math
from queue  import Queue
import pandas as pd 
import warnings
import time
import trie as trie
import pickle
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy import spatial
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA as sklearnPCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import os
import sys

tweets=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/jimJordanKnew.csv",sep =',')
tweets['Time'] =  pd.to_datetime(tweets['Time'])
tweets = tweets.sort_values(by='Time',ascending=True)

print(list(tweets.columns.values))

length=len(tweets)
batch_size=250
print(length,batch_size)
batch_epoch=[]

for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
	batch_epoch.append(tweet_batch['Time'].values[-1])
	print(tweet_batch['Time'].values[-1])
# print(batch_epoch)

epoch_diff=[(t - s)/np.timedelta64(1, 's') for s, t in zip(batch_epoch[:-1], batch_epoch[1:])]


# epoch_diff= pd.Series(batch_epoch).diff().dt.second.dropna().tolist()

# print(epoch_diff)

print(len(batch_epoch),len(epoch_diff))

bins=int((max(epoch_diff)-min(epoch_diff))/50)

plt.hist(epoch_diff, color = 'blue', edgecolor = 'black', bins = bins)
plt.title('Histogram of Inter-Arrival Time between Tweet Batches')
plt.xlabel('Inter-Arrival Time in seconds')
plt.ylabel('Frequency')

plt.show()