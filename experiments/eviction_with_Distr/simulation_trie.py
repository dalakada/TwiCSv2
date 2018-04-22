
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
tweets=pd.read_csv("tweets_10k_from_1mill.csv",sep =',')
annotated_tweets=pd.read_csv("tweets_3k_annotated.csv",sep =',')
# tweets=tweets[:1000:]
print('Tweets are in memory...')
batch_size=922
length=len(tweets)
val=math.ceil(length/batch_size)-1


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


# val =3

my_classifier= svm.SVM1('training.csv')

#last one is the without eviction, that why i added one more.
#look the defter notes to see mapping.
for eviction_parameter in range(val):
    eviction_parameter_recorder.append(eviction_parameter)
    Phase1= phase1.SatadishaModule()
    Phase2 = phase2.EntityResolver()
    total_time=0
    execution_time_list=[]
    tweets_been_processed_list=[]
    tweets_been_processed=0


    level_holder=[]

    annotated_tweet_evenly_partitioned_list=np.array_split(annotated_tweets, val)
    for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):

        # concat annotated partitons with big tweets
        print(len(annotated_tweet_evenly_partitioned_list[g]),len(tweet_batch))
        tweet_batch = pd.concat([tweet_batch,annotated_tweet_evenly_partitioned_list[g]])
        print(len(tweet_batch))
        print(tweet_batch.tail())
        tuple_of= Phase1.extract(tweet_batch,g)
        tweet_base=tuple_of[0]

        candidate_base=tuple_of[1]
        phase2stopwordList=tuple_of[4]
        elapsedTime= tuple_of[3] - tuple_of[2]
        total_time+=0
        print(elapsedTime,total_time)
        print (g,' ', 'Produced')
        print("**********************************************************")
        # if(g==val):
        #     candidateList=candidate_base.displayTrie("",[])
        #     candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
        #     #print(len(candidateBase))
        #     candidateBase.to_csv('candidateBase.csv' ,sep=',', encoding='utf-8')
        #     print('Finished writing Candidate Base')
        time_in=time.time()

        tweets_been_processed=tweets_been_processed+len(tweet_base)
        tweets_been_processed_list.append(tweets_been_processed)


        time_out=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.7,eviction_parameter,my_classifier)
        accuracy_list,tp_count=Phase2.finish()
        elapsedTime= time_out-time_in
        total_time+=elapsedTime
        execution_time_list.append(total_time)

        print(elapsedTime,total_time)
        print(g,' ','Consumed')
        print("**********************************************************")
    #print(len(phase2TweetBase))

    print(execution_time_list)
    level_holder.append(execution_time_list)
    level_holder.append(accuracy_list)
    level_holder.append(tweets_been_processed_list)
    level_holder.append(eviction_parameter)
    level_holder.append(tp_count)
    whole_level.append(copy.deepcopy(level_holder))





for i in whole_level:
    print("********************************************")
    print(i)
    print("********************************************")











    
without_eviction_id=len(whole_level)-1
without_eviction=whole_level[without_eviction_id]

# timing=[[0.7756309509277344, 1.404196949005127, 2.1200640201568604, 2.8386363983154297, 3.569007158279419],
# [0.7308433055877686, 1.4264043292999268, 2.184626636505127, 3.0043627166748047, 3.820970058441162],
# [0.7488808631896973, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.923989772796631],
# [0.7770745754241943, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.943989772796631],
# [0.7539031505584717, 1.4265043292999268, 2.204626636505127, 3.1043627166748047, 3.963989772796631]]

# timing_id=len(timing)-1
# timing_max=timing[timing_id]

# timing_sliced=timing[:-1]

p1_holder=[]
p2_holder=[]

# print("Without eviction time : ",without_eviction[0])
for idx,level in enumerate(whole_level[:-1]):
    # print(level[0])

    # print(level)
    # accuracy=level[1]
    p1_divided=[]
    
    for i in range(len(level[1])):
        p1_divided.append(level[1][i]/without_eviction[1][i])
        # print(p1_divided)

    # tweets_been_processed_list=level[2]
    # p1_divided=sorted(p1_divided)
    p2=[]
    # for i in range(len(level[0])):
    #     p2.append(without_eviction[0][i]-level[0][i])
    for i in range(len(level[0])):
        # p2.append(timing_max[i]-timing_sliced[idx][i])
        p2.append(level[0][i]-without_eviction[0][i])


    tweets_been_proccessed=level[2]

    p1xp2=[]

    # p2=sorted(p2)

    for i in range(len(p1_divided)):
        p1xp2.append(p2[i]*p1_divided[i])

    # print('P1 : ',p1_divided,'Recall without :',without_eviction[1])

    # print('Recall : ',level[1],'Recall without :',without_eviction[1])

    # print('TP: ' ,level[4],'Without ', without_eviction[4])

    p1_holder.append(p1_divided)
    p2_holder.append(p2)

p1_holder_tranpsosed=list(map(list, zip(*p1_holder)))
p2_holder_tranpsosed=list(map(list, zip(*p2_holder)))

print("***************************************************************")
for i in p2_holder:
    print(i)
for i in p1_divided:
    print(i)
# print(eviction_parameter_recorder)
# for i in p1_holder:
#     print(i)
eviction_parameter_recorder=eviction_parameter_recorder[:-1]
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p1,label=tweets_been_proccessed[idx+1])
    ax1.text(eviction_parameter_recorder[0], p1[0], 'p1')
    ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    ax2.text(eviction_parameter_recorder[0], p2[0], 'p2')

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')

    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p1')
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")

plt.show()

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p1,label=tweets_been_proccessed[idx+1])
    #ax1.text(eviction_parameter_recorder[0], p1[0], tweets_been_proccessed[idx])
    #ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax2.text(eviction_parameter_recorder[0], p2[0], tweets_been_proccessed[idx])

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')

    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p1')
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")

plt.show()

fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    # fit = np.polyfit(eviction_parameter_recorder,p1,1)
    # fit_fn1 = np.poly1d(fit) 

    # fit = np.polyfit(eviction_parameter_recorder,p2,1)
    # fit_fn2 = np.poly1d(fit) 

    # h = lambda x: fit_fn1(x)- fit_fn2(x)

    # x = np.arange(-500, 200, 1)


    # x_int = scipy.optimize.fsolve(h, 0)
    # y_int = fit_fn1 (x_int)

    # print('************************************')
    # print(tweets_been_proccessed[idx])
    # print(x_int, y_int)
    # print(fit_fn1,fit_fn2)
    # print('************************************')

    ax1.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax1.text(eviction_parameter_recorder[0], p1[0], tweets_been_proccessed[idx])
    #ax2.plot(eviction_parameter_recorder, p2,label=tweets_been_proccessed[idx+1])
    #ax2.text(eviction_parameter_recorder[0], p2[0], tweets_been_proccessed[idx])

    ###plt.plot(x, f(x), zorder=1)
    ###plt.plot(x, g(x), zorder=1)

    # idx = np.argwhere(np.isclose(fit_fn1, fit_fn2, atol=10)).reshape(-1)




    # ax3.scatter(x_int, y_int, marker='x')



    ax1.set_xlabel('Eviction Parameter ')
    ax1.set_ylabel('p2')


    tick_spacing = 1
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    #ax2.set_ylabel('p2')

    ## AFTER ####
    # plt.plot( tweets_been_proccessed,p1xp2,marker='o' , label=eviction_parameter_recorder[idx],alpha=0.5)



    plt.grid(True)
    plt.legend(loc='upper left')
    # plt.savefig("Execution-Time-vs-Batch-Size.png")


plt.show()
