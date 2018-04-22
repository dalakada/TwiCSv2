
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
Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
tweets=pd.read_csv("tweets_3k_annotated.csv",sep =',')
#print('Tweets are in memory...')
# tweets=tweets[:10000:]
batch_size=500
length=len(tweets)
val=math.ceil(length/batch_size)-1


# define ready thresholds based on batch size .

ready_thresholds=range(val)

print(ready_thresholds)


execution_time_list=[]
accuracy_list=[]
batch_size_recorder=[]

whole_level=[]

my_classifier= svm.SVM1('training.csv')

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()

tweets_been_processed_list=[]
tweets_been_processed=0

support=0
support_list=[]

recall=0
recall_list=[]


level_holder=[]
for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
    # tweet_with_cws_list=[]
    # tweets_been_processed_list=[]
    print(len(tweet_batch))
    tuple_of= Phase1.extract(tweet_batch,g)
    tweet_base=tuple_of[0]
    recall=Phase1.calculate_recall(tweet_base)
    support=Phase1.calculate_support(tweet_base)

    gap_dict=tuple_of[5]
    print (g,' ', 'Produced')
    print("**********************************************************")


    tweets_been_processed=len(tweet_base)+tweets_been_processed
    tweets_been_processed_list.append(tweets_been_processed)

    recall=recall
    recall_list.append(recall)
    
    supoort=support
    support_list.append(support)

level_holder.append(tweets_been_processed_list)
level_holder.append(recall_list)
level_holder.append(support_list)
    # level_holder.append(accuracy_list)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=20)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis2 = fm.FontProperties(fname=fontPath, size=24)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=16)

whole_level.append(copy.deepcopy(level_holder))

# print(gap_dict)
for i in whole_level:
    print(i)

sol = [x-300 for x in whole_level[0][0]]
sag= [x for x in whole_level[0][0]]

fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

rects1=ax.bar(sol, whole_level[0][1],width=300,color='b',align='center')
rects2=ax.bar(sag, whole_level[0][2],width=300,color='g',align='center')
plt.tick_params(axis='both', which='major', labelsize=12)

ax.set_xlabel('Tweets in Input Stream',fontproperties=font_axis2)
ax.set_ylabel('CS Interestingness Measures',fontproperties=font_axis)

ax.legend( (rects1[0], rects2[0]), ('Recall', 'Confidence'),loc="upper right",ncol=2,frameon=False,prop=font_legend )
fig.savefig("recall-with-confidence-histogram.pdf",dpi=1200,bbox_inches='tight')

# for idx,level in enumerate(whole_level):

#     # print(level[0])
#     # print(level)
#     # accuracy=level[1]
#     p1_divided=[]
    
#     for i in range(len(level[1])):
#         p1_divided.append(no_reintro_[1][i]/level[1][i])
#         # print(p1_divided)

#     # tweets_been_processed_list=level[2]
#     # p1_divided=sorted(p1_divided)
#     p2=[]
#     for i in range(len(level[0])):
#         p2.append(no_reintro_[0][i]-level[0][i])

#     p1_holder.append(p1_divided)
#     p2_holder.append(p2)

#     tweets_been_proccessed=level[2]
#     print(tweets_been_proccessed)


# p1_holder_tranpsosed=list(map(list, zip(*p1_holder)))
# p2_holder_tranpsosed=list(map(list, zip(*p2_holder)))

# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
# print(whole_level[0][0], whole_level[0][1],whole_level[0][2])
# # pass the no_reintro..
# ready_thresholds=ready_thresholds[1:]
# for idx,level in enumerate(p1_holder_tranpsosed[1:]):
#     p1=level
#     p2=p2_holder_tranpsosed[idx]

# ax1.plot(whole_level[0][0], whole_level[0][1],marker='o')
#     ax1.text(ready_thresholds[0], p1[0], 'p1')
#     ax2.plot(ready_thresholds, p2,label=tweets_been_proccessed[idx+1])
#     ax2.text(ready_thresholds[0], p2[0], 'p2')

#     # ax3.scatter(x_int, y_int, marker='x')

#     tick_spacing = 1
#     ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))




#     plt.grid(True)
#     plt.legend(loc='upper left')
plt.show()



# import matplotlib.pyplot as plt

# x = [datetime.datetime(2011, 1, 4, 0, 0),
#      datetime.datetime(2011, 1, 5, 0, 0),
#      datetime.datetime(2011, 1, 6, 0, 0)]
# x = date2num(x)

# y = [4, 9, 2]
# z=[1,2,3]
# k=[11,12,13]

# ax = plt.subplot(111)
# ax.bar(x-0.2, y,width=0.2,color='b',align='center')
# ax.bar(x, z,width=0.2,color='g',align='center')
# ax.bar(x+0.2, k,width=0.2,color='r',align='center')
# ax.xaxis_date()

plt.show()