
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
import SVM as svm
import matplotlib.ticker as ticker
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style



def p1_f(x,p1):
    return p1[x]

def p2_f(x,p2):
    return p2[x]


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
tweets=pd.read_csv("new_set4.csv",sep =',')
tweets=tweets[:50000:]
# tweets = tweets.sample(frac=1).reset_index(drop=True)
# annotated_tweets=pd.read_csv("political_annotated.csv",sep =',')
# tweets=tweets[:1000:]
print('Tweets are in memory...')
batch_size=250000
print(len(tweets))
length=len(tweets)


# Z_scores=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# 

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
execution_time_list=[]
accuracy_list=[]
tp_count=[]
eviction_parameter_recorder=[]

whole_level=[]
val=math.ceil(length/batch_size)

for i in range(val):
    print(i)

print("anani siki2m")
# val =3

my_classifier= svm.SVM1('training.csv')

#last one is the without eviction, that why i added one more.
#look the defter notes to see mapping.
eviction_parameter=1
eviction_parameter_recorder.append(eviction_parameter)
Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
total_time=0
execution_time_list=[]
tweets_been_processed_list=[]
tweets_been_processed=0
total_mentions_discovered=[]

level_holder=[]

# annotated_tweet_evenly_partitioned_list=np.array_split(annotated_tweets, val)
# for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):

tuple_of= Phase1.extract(tweets,0)
tweet_base=tuple_of[0]

candidate_base=tuple_of[1]
phase2stopwordList=tuple_of[4]
elapsedTime= tuple_of[3] - tuple_of[2]
total_time+=elapsedTime
print(elapsedTime,total_time)
print (0,' ', 'Produced')
print("**********************************************************")


tweets_been_processed=tweets_been_processed+len(tweet_base)
tweets_been_processed_list.append(tweets_been_processed)

time_in,time_out,total_mentions=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.5,eviction_parameter,my_classifier)
elapsedTime= time_out-time_in
total_time+=elapsedTime
execution_time_list.append(total_time)
total_mentions_discovered.append(total_mentions)

converted=Phase2.convertedd_ones()
print(converted)

