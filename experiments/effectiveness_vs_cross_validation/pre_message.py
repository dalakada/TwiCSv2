
#import SatadishaModule as phase1
import SatadishaModule_final_trie as phase1
import phase2_Trie as phase2
import datetime
from threading import Thread
import random
import math
from queue  import Queue
import pandas as pd 
import warnings
import numpy as np
import time
import trie as trie
import pickle
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

thread_processed=0
stream_count=0
queue = Queue(1000)
#time_in=datetime.datetime.now()
#time_out=datetime.datetime.now()
fieldnames=['candidate','freq','length','cap','start_of_sen','abbrv','all_cap','is_csl','title','has_no','date','is_apostrp','has_inter_punct','ends_verb','ends_adverb','change_in_cap','topic_ind','entry_time','entry_batch','@mention']

global total_time
total_time=0
Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()


tweets=pd.read_csv("tweets_3k_annotated.csv",sep =',')
# tweets=tweets[:1000:]
print('Tweets are in memory...')
batch_size=500
length=len(tweets)
val=math.ceil(length/batch_size)-1


Z_scores=[-0.5]
# 

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
execution_time_list=[]
accuracy_list=[]
batch_size_recorder=[]

# whole_level=[]
# for z_score in Z_scores:
#     # batch_size_ratio_float= batch_size_ratio/100.0
#     # # print(batch_size_ratio_float)
#     # batch_size=len(tweets)*batch_size_ratio_float
#     # batch_size_recorder.append(batch_size)
val=math.ceil(length/batch_size)-1
Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
total_time=0
execution_time_list=[]
tweets_been_processed_list=[]
tweets_been_processed=0
batch_size=500

for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
    tuple_of= Phase1.extract(tweets,0)
    tweet_base=tuple_of[0]
    tweet_base.to_csv('tweet_base.csv' ,sep=',',   encoding='utf-8')

