
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
import matplotlib
from matplotlib import rc
import matplotlib.font_manager as fm

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'dejavusans','serif':['Times']})
rc('text', usetex=False)
csfont = {'fontname':'DejaVu Sans Condensed'}

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

whole_level=[]
tweets_been_processed_list=[]
tweets_been_processed=0
batch_size=500

level_holder=[]
for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
    tuple_of= Phase1.extract(tweet_batch,g)
    tweet_base=tuple_of[0]
    tweet_base.to_csv('tweet_base.csv' ,sep=',',   encoding='utf-8')

    # with open('tweet_base'+str(g)+'.pkl', 'wb') as output:
    #     pickle.dump(tweet_base, output, pickle.HIGHEST_PROTOCOL)

    candidate_base=tuple_of[1]
    phase2stopwordList=tuple_of[4]
    # candidateList=candidate_base.displayTrie("",[])
    # candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
    # candidateBase.to_csv('candidate_base.csv' ,sep=',', encoding='utf-8')
    
    # with open('candidate_base'+str(g)+'.pkl', 'wb') as output2:
    #     pickle.dump(candidate_base, output2, pickle.HIGHEST_PROTOCOL)



    # print('len of tweet_base = ' , len(tweet_base))
    elapsedTime= tuple_of[3] - tuple_of[2]
    total_time+=elapsedTime
    print(elapsedTime,total_time)
    print (g,' ', 'Produced')
    print("**********************************************************")
    if(g==val):
        candidateList=candidate_base.displayTrie("",[])
        candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
        #print(len(candidateBase))
        candidateBase.to_csv('candidateBase.csv' ,sep=',', encoding='utf-8')
        print('Finished writing Candidate Base')
    time_in=time.time()

    tweets_been_processed=tweets_been_processed+len(tweet_batch)
    tweets_been_processed_list.append(tweets_been_processed)
    phase2TweetBase=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.5,tweet_base)
    accuracy_list,accuracy_list_stanford,accuracy_list_ritter,accuracy_list_opencalai=Phase2.finish()
    Phase2.finish_other_systems()
    time_out=time.time()
    elapsedTime= time_out-time_in
    total_time+=elapsedTime
    execution_time_list.append(total_time)

    print(elapsedTime,total_time)
    print(g,' ','Consumed')
    print("**********************************************************")
#print(len(phase2TweetBase))


# level_holder.append(execution_time_list)
# level_holder.append(accuracy_list)
# level_holder.append(tweets_been_processed_list)


whole_level.append(accuracy_list)
whole_level.append(accuracy_list_stanford)
whole_level.append(accuracy_list_ritter)
whole_level.append(accuracy_list_opencalai)
