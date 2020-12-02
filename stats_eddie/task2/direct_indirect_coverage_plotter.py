
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
import SVM as svm
import matplotlib
from matplotlib import rc
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore")

thread_processed=0
stream_count=0
queue = Queue(1000)
#time_in=datetime.datetime.now()
#time_out=datetime.datetime.now()
fieldnames=['candidate','freq','length','cap','start_of_sen','abbrv','all_cap','is_csl','title','has_no','date','is_apostrp','has_inter_punct','ends_verb','ends_adverb','change_in_cap','topic_ind','entry_time','entry_batch','@mention']

global total_time
total_time=0
# Phase1= phase1.SatadishaModule()
# Phase2 = phase2.EntityResolver()
# tweets=pd.read_csv("tweets_1million_for_others.csv",sep =',')
# tweets=tweets[:50000:]

# tweets2=pd.read_csv("deduplicated_test.csv",sep =',')
# tweets2=tweets2[:100:]
# tweets_holder=[]
# tweets_holder.append(tweets)
# tweets_holder.append(tweets2)
# print('Tweets are in memory...')
# batch_size=5000
# length=len(tweets)
# val=math.ceil(length/batch_size)-1



# # define ready thresholds based on batch size .

# ready_thresholds=range(val)

# print(ready_thresholds)


# execution_time_list=[]
# accuracy_list=[]
# batch_size_recorder=[]

# whole_level=[]

# my_classifier= svm.SVM1('training.csv')

# Phase1= phase1.SatadishaModule()
# Phase2 = phase2.EntityResolver()

# tweets_been_processed_list=[]
# tweets_been_processed=0

# tweet_with_cws=0
# tweet_with_cws_list=[]

# indirect_counter=0
# indirect_counter_list=[]


# level_holder=[]

# # tweet_with_cws_list=[]
# # tweets_been_processed_list=[]
# for tweet_db in tweets_holder:
#     Phase1= phase1.SatadishaModule()
#     Phase2 = phase2.EntityResolver()

#     tuple_of= Phase1.extract(tweet_db,0)

#     tweet_base=tuple_of[0]
#     candidate_base=tuple_of[1]
#     phase2stopwordList=tuple_of[4]

#     number_of_tweets_with_cws=Phase1.get_number_of_tweets_with_cws()

#     print ('Produced')
#     print("**********************************************************")


#     tweets_been_processed=len(tweet_base)
#     tweets_been_processed_list.append(tweets_been_processed)

#     tweet_with_cws=number_of_tweets_with_cws
#     tweet_with_cws_list.append(tweet_with_cws/len(tweet_base))

#     ######## phase2 starts....
#     df_holder=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.15)
#     indirect_counter=Phase2.calculate_indirect_coverage(df_holder)/len(tweet_base)
#     indirect_counter_list.append(indirect_counter)

# level_holder.append(tweets_been_processed_list)
# level_holder.append(tweet_with_cws_list)
# level_holder.append(indirect_counter_list)


# whole_level.append(copy.deepcopy(level_holder))


# print(whole_level)
# for i in whole_level:
#     print(i)
whole_level=[[[1713105, 56551, 1287], [0.4977330636475873, 0.39518310905200615, 0.3572961373390558], [0.4267555111916666, 0.36528089688953336, 0.15879828326180256]]]
whole_level[0][0]=[1,2,3]





sol = [x  for x in whole_level[0][0]]
sag= [x+0.25 for x in whole_level[0][0]]
width = 0.25       # the width of the bars

fig= plt.subplots()
fontPath = "/Users/satadisha/Downloads/abyssinica-sil/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=24)


font_legend = fm.FontProperties(fname=fontPath, size=18)

fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)
# ax = plt.subplot(111)
rects1=ax.bar(sol, whole_level[0][1],width=0.25 ,color='b',align='center')
rects2=ax.bar(sag, whole_level[0][2],width=0.25 ,color='g',align='center')

a=np.asarray([1,2,3])
plt.tick_params(axis='both', which='major', labelsize=14)

ax.legend( (rects1[0], rects2[0]), ('Direct Coverage', 'Indirect Coverage'),loc="upper right",frameon=False,prop=font_legend )
ax.set_xlabel('Tweets in Input Stream',fontproperties=font_axis)
ax.set_ylabel('Percentages',fontproperties=font_axis)
ax.set_xticks(a + width / 2)
ax.set_xticklabels(('D4', 'D5', 'WNUT'))
# fig.savefig("direct-indirect-coverage.pdf",dpi=1200,bbox_inches='tight')

plt.show()
# level_holder=[]
# for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
#     # tweet_with_cws_list=[]
#     # tweets_been_processed_list=[]
#     tuple_of= Phase1.extract(tweet_batch,g)

#     tweet_base=tuple_of[0]
#     candidate_base=tuple_of[1]
#     phase2stopwordList=tuple_of[4]

#     number_of_tweets_with_cws=Phase1.get_number_of_tweets_with_cws()

#     print (g,' ', 'Produced')
#     print("**********************************************************")


#     tweets_been_processed=len(tweet_base)+tweets_been_processed
#     tweets_been_processed_list.append(tweets_been_processed)

#     tweet_with_cws=number_of_tweets_with_cws
#     tweet_with_cws_list.append(tweet_with_cws)

#     ######## phase2 starts....
#     df_holder=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.15)
#     indirect_counter=Phase2.calculate_indirect_coverage(df_holder)
#     indirect_counter_list.append(indirect_counter)

# level_holder.append(tweets_been_processed_list)
# level_holder.append(tweet_with_cws_list)
# level_holder.append(indirect_counter_list)


# whole_level.append(copy.deepcopy(level_holder))

# whole_level.append(copy.deepcopy(level_holder))

# for i in whole_level:
#     print(i)

# sol = [x-300 for x in whole_level[0][0]]
# sag= [x for x in whole_level[0][0]]

# ax = plt.subplot(111)
# rects1=ax.bar(sol, whole_level[0][1],width=300,color='b',align='center')
# rects2=ax.bar(sag, whole_level[0][2],width=300,color='g',align='center')

# ax.legend( (rects1[0], rects2[0]), ('# of Tweet with CWS', 'Indirect Coverage') )
# ax.set_xlabel('# of Seen Tweets')
# ax.set_ylabel('Counts')
# plt.show()
