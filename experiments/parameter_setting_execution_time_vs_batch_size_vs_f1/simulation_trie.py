
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

warnings.filterwarnings("ignore")

thread_processed=0
stream_count=0
queue = Queue(1000)
#time_in=datetime.datetime.now()
#time_out=datetime.datetime.now()
fieldnames=['candidate','freq','length','cap','start_of_sen','abbrv','all_cap','is_csl','title','has_no','date','is_apostrp','has_inter_punct','ends_verb','ends_adverb','change_in_cap','topic_ind','entry_time','entry_batch','@mention']
#total_time=0
# class Phase1Thread(Thread):

#     def run(self):
#         global queue
#         global fieldnames
#         global total_time
#         self.Phase1= phase1.SatadishaModule()
#         #tweets=pd.read_csv("malcolmx.csv") (testTB.csv 5 ;) (tweets_1million_for_others.csv 20000 ,) (deduplicated_test.csv 5000 ;) (500_annotated.csv 500 ,) (3000_tweet_data_frame.csv 500 ;)
#         tweets=pd.read_csv("testTB.csv",sep =';')
#         #tweets=tweets[:6000:]
#         print('Tweets are in memory...')
#         batch_size=5
#         length=len(tweets)
#         val=math.ceil(length/batch_size)-1
#         #print(val)
#         for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
#             #print(g)
#             tuple_of= self.Phase1.extract(tweet_batch,g)
#             tweet_base=tuple_of[0]
#             candidate_base=tuple_of[1]
#             print('len of tweet_base = ' , len(tweet_base))
#             elapsedTime= tuple_of[3] - tuple_of[2]
#             total_time+=elapsedTime
#             print(elapsedTime,total_time)
#             queue.put(tuple_of)
#             print (g,' ', 'Produced')
#             print("**********************************************************")
#             if(g==val):
#                 candidateList=candidate_base.displayTrie("",[])
#                 candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
#                 print(len(candidateBase))
#                 candidateBase.to_csv('candidateBase.csv' ,sep=',', encoding='utf-8')
#                 print('Finished writing Candidate Base')
#             time.sleep(random.random())



# class Phase2Thread(Thread):
#     def run(self):
#         self.output_queue_holder=[]
#         self.unique_good_holder=[]
#         self.unique_bad_holder=[]
#         self.Phase2 = phase2.EntityResolver()
#         global queue
#         global stream_count
#         global thread_processed
#         counter=0
#         while (True):
#             if(counter == 15):
#                 break

#             tuple_of = queue.get()
#             tweet_base=tuple_of[0]
#             candidate_base=tuple_of[1]
#             queue.task_done()
#             print(counter,' aldim')
#             #time_in=datetime.datetime.now()
#             time_in=time.time()
#             self.Phase2.extract(tweet_base,candidate_base,-0.15)
#             #output_queue,unique_good,unique_bad=self.Phase2.run(tweet_base,candidate_base,-0.15)
#             #time_out=datetime.datetime.now()
#             time_out=time.time()
#             print(counter,' ','Consumed')
#             print(' ')
#             #self.output_queue_holder.append(output_queue)
#             #self.unique_good_holder.append(unique_good)
#             #self.unique_bad_holder.append(unique_bad)
#             #print(len(output_queue),len(unique_good),len(unique_bad))
#             elapsedTime= time_out - time_in
#             print(elapsedTime)
#             #print(divmod(elapsedTime.total_seconds(), 60))
#             #print(unique_good)
#             #print(unique_bad)
#             counter=counter+1
#             time.sleep(random.random())
#             #print(output_queue)
#             #output_queue.to_csv("sample_output_queue.csv", sep=',', encoding='utf-8')
#             #thread_processed= thread_processed +1
#             #print(thread_processed)










#iterative sequential run
#(tweets_1million_for_others.csv 20000) (deduplicated_test.csv, 5000)
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


# Z_scores=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# 

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
execution_time_list=[]
accuracy_list=[]


whole_level=[]
for batch_size_ratio in range(2,50,5):
    batch_size_ratio_float= batch_size_ratio/100.0
    # print(batch_size_ratio_float)
    batch_size=len(tweets)*batch_size_ratio_float
    val=math.ceil(length/batch_size)-1
    Phase1= phase1.SatadishaModule()
    Phase2 = phase2.EntityResolver()
    total_time=0
    execution_time_list=[]
    tweets_been_processed_list=[]
    tweets_been_processed=0

    level_holder=[]
    for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):
        tuple_of= Phase1.extract(tweet_batch,g)
        tweet_base=tuple_of[0]
        #tweet_base.to_csv('tweet_base.csv' ,sep=',', encoding='utf-8')

        print('ananisikim',batch_size)
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

        tweets_been_processed=tweets_been_processed+len(tweet_base)
        tweets_been_processed_list.append(tweets_been_processed)
        phase2TweetBase=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,-0.15)
        accuracy_list=Phase2.finish()
        time_out=time.time()
        elapsedTime= time_out-time_in
        total_time+=elapsedTime
        execution_time_list.append(total_time)

        print(elapsedTime,total_time)
        print(g,' ','Consumed')
        print("**********************************************************")
    #print(len(phase2TweetBase))


    level_holder.append(execution_time_list)
    level_holder.append(accuracy_list)
    level_holder.append(tweets_been_processed_list)


    whole_level.append(copy.deepcopy(level_holder))




for level in whole_level:

    execution_time=level[0]
    tweets_been_processed_list=level[2]

    plt.plot( tweets_been_processed_list,execution_time ,alpha=0.5)


    plt.xlabel('# of Tweets Seen')
    plt.ylabel('Execution Time')
    plt.grid(True)
    plt.savefig("F_score_mention_VS_F_score_entity.png")
    plt.show()

    # thefile = open('time_'+str(batch_size)+'.txt', 'w')
    # thefile2= open('number_of_processed_tweets'+str(batch_size)+'.txt', 'w')



    # for item in execution_time_list:
    #   thefile.write("%s\n" % item)

    # with open('accuracy_'+str(batch_size)+'.txt', 'w') as fp:
    #     fp.write('\n'.join('%s %s %s %s %s' % x for x in accuracy_list))


    # for item in tweets_been_processed_list:
    #   thefile2.write("%s\n" % item)
        



