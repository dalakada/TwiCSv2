
#import SatadishaModule as phase1
import SatadishaModule_final_trie as phase1
# import phase2_Trie as phase2
import phase2_Trie_baseline_reintroduction_effectiveness as phase2
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
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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

# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/TwiCSv2/production_code/tweets_3k_annotated.csv",sep =',', encoding='utf-8',keep_default_na=False)
tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/tweets_3k_annotated.csv",sep =',', encoding='utf-8',keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/venezuela.csv",sep =',', encoding='utf-8',keep_default_na=False)

# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/roevwade.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billdeblasio.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/pikapika.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/ripcity.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billnye.csv",sep =',', keep_default_na=False)

# tweets=tweets[:1000:]
print('Tweets are in memory...')
batch_size=600
length=len(tweets_unpartitoned)
val=math.ceil(length/batch_size)-1


# Z_scores=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# Z_scores=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
Z_scores=[1,2,3]
# 
# Z_scores=[-0.2]

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
execution_time_list=[]
accuracy_list=[]
batch_size_recorder=[]

whole_level=[]
# tweets_unpartitoned = shuffle(tweets_unpartitoned)
max_batch_value=112
test_sets=[]

# kf = KFold(n_splits=5,random_state=1000) 

#  #validation
# for train_ind,test_ind in kf.split(tweets_unpartitoned):
#     # print(train_ind,test_ind)
#     tweets=tweets_unpartitoned.iloc[train_ind]
#     test=tweets_unpartitoned.iloc[test_ind]
#     test_sets.append(test)
#     print("**********************")
#     # xtrain = df[train_ind:]
#     # xtest = df[test_ind:]

#     level_holder=[]

#     for z_score in Z_scores:
#         # batch_size_ratio_float= batch_size_ratio/100.0
#         # # print(batch_size_ratio_float)
#         # batch_size=len(tweets)*batch_size_ratio_float
#         # batch_size_recorder.append(batch_size)
#         val=math.ceil(length/batch_size)-1
#         Phase1= phase1.SatadishaModule()
#         Phase2 = phase2.EntityResolver()
#         total_time=0
#         execution_time_list=[]
#         tweets_been_processed_list=[]
#         tweets_been_processed=0
#         batch_size=500

        

#         tuple_of= Phase1.extract(tweets,0)
#         tweet_base=tuple_of[0]
#         #tweet_base.to_csv('tweet_base.csv' ,sep=',',   encoding='utf-8')

#         # with open('tweet_base'+str(g)+'.pkl', 'wb') as output:
#         #     pickle.dump(tweet_base, output, pickle.HIGHEST_PROTOCOL)

#         candidate_base=tuple_of[1]
#         phase2stopwordList=tuple_of[4]
#         # candidateList=candidate_base.displayTrie("",[])
#         # candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
#         # candidateBase.to_csv('candidate_base.csv' ,sep=',', encoding='utf-8')
        
#         # with open('candidate_base'+str(g)+'.pkl', 'wb') as output2:
#         #     pickle.dump(candidate_base, output2, pickle.HIGHEST_PROTOCOL)



#         # print('len of tweet_base = ' , len(tweet_base))
#         elapsedTime= tuple_of[3] - tuple_of[2]
#         total_time+=elapsedTime
#         print(elapsedTime,total_time)
#         print ('Produced')
#         print("**********************************************************")
#         # if(g==val):
#         #     candidateList=candidate_base.displayTrie("",[])
#         #     candidateBase=pd.DataFrame(candidateList, columns=fieldnames)
#         #     #print(len(candidateBase))
#         #     candidateBase.to_csv('candidateBase.csv' ,sep=',', encoding='utf-8')
#         #     print('Finished writing Candidate Base')
#         time_in=time.time()

#         tweets_been_processed=tweets_been_processed+len(tweet_base)
#         tweets_been_processed_list.append(tweets_been_processed)
#         # phase2TweetBase=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,z_score,tweet_base)
#         reintroduction_threshold_dummy=0
         
#         # # #phase2_Trie_baseline_reintroduction_effectiveness
#         candidate_base_post_Phase2, converted_candidates, complete_tweet_dataframe_grouped_df_sorted= Phase2.executor(max_batch_value,tweet_base,candidate_base,phase2stopwordList,z_score,reintroduction_threshold_dummy,tweets)
#         # #taking phase2 output time in phase 2 class due to unrelated index reset operation at the end of last batch
#         time_out=time.time()

#         accuracy_list=Phase2.finish()

#         time_out=time.time()
#         elapsedTime= time_out-time_in
#         total_time+=elapsedTime
#         execution_time_list.append(total_time)

#         print(elapsedTime,total_time)
#         # print(g,' ','Consumed')
#         # print("**********************************************************")
#         #print(len(phase2TweetBase))


#         # level_holder.append(execution_time_list)
#         level_holder.append(accuracy_list)
#         # level_holder.append(tweets_been_processed_list)



#     whole_level.append(copy.deepcopy(level_holder))

# print('BURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBAREN')

# whole_level_transposed=list(map(list, zip(*whole_level)))

# print(whole_level)



# fig, ax = plt.subplots()

# print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")



# best_z_scores=[]


# for each_partition in whole_level:
#     maximum=0
#     best_z_score=0
#     for idx,each_zscore_tuple in enumerate(each_partition):
#         if each_zscore_tuple[1]>=maximum:
#             maximum=each_zscore_tuple[1]
#             best_z_score=each_zscore_tuple[0]

#     print(best_z_score)
#     best_z_scores.append(best_z_score)



# #testing
best_z_scores=[1,1,1,1,1]
counter=0
new_acc_list=[]

stanford=[]
ritter=[]
opencalai=[]
neuroner=[]

Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()

ind=0
batch_count=0
acc_holder=[]
# for train_ind,test_ind in kf.split(tweets_unpartitoned):
# for ind in range(len(test_sets)):
# for ind in range(len([1])):
for g, tweet_batch in tweets_unpartitoned.groupby(np.arange(length) //batch_size):
        
    # input_size_arr.append(convert_bytes(sys.getsizeof(tweet_batch)))
    # print('input_size_arr: ',input_size_arr)
    #tweet_base.to_csv('tweet_base


    # tweets=tweets_unpartitoned.iloc[test_ind]
    # tweets=tweets_unpartitoned
    # tweets=test_sets[ind]
    z_score=1
        # batch_size_ratio_float= batch_size_ratio/100.0
        # # print(batch_size_ratio_float)
        # batch_size=len(tweets)*batch_size_ratio_float
        # batch_size_recorder.append(batch_size)
    val=math.ceil(length/batch_size)-1
    # Phase1= phase1.SatadishaModule()
    # Phase2 = phase2.EntityResolver()
    total_time=0
    execution_time_list=[]
    tweets_been_processed_list=[]
    tweets_been_processed=0
    # batch_size=600

    
    tuple_of= Phase1.extract(tweet_batch,batch_count)
    ind+=1


    tweet_base=tuple_of[0]

    candidate_base=tuple_of[1]
    phase2stopwordList=tuple_of[4]


    # print('len of tweet_base = '  len(tweet_base))
    elapsedTime= tuple_of[3] - tuple_of[2]
    total_time+=elapsedTime
    print(elapsedTime,total_time)
    print ('Produced')
    print("**********************************************************")

    time_in=time.time()

    tweets_been_processed=tweets_been_processed+len(tweet_base)
    tweets_been_processed_list.append(tweets_been_processed)
    # phase2TweetBase=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,best_z_scores[counter],tweet_base)
    reintroduction_threshold_dummy=0  
    # # #phase2_Trie_baseline_reintroduction_effectiveness
    candidate_base_post_Phase2, converted_candidates, complete_tweet_dataframe_grouped_df_sorted= Phase2.executor(max_batch_value,tweet_base,candidate_base,phase2stopwordList,z_score,reintroduction_threshold_dummy,tweet_batch)

    new_acc_list=Phase2.finish()
    # accuracy_list_stanford,accuracy_list_opencalai,accuracy_list_ritter,accuracy_list_neuroner=Phase2.finish_other_systems()
    time_out=time.time()
    elapsedTime= time_out-time_in
    total_time+=elapsedTime
    execution_time_list.append(total_time)

    print(elapsedTime,total_time)
    # print(g,' ','Consumed')
    # print("**********************************************************")
    #print(len(phase2TweetBase))


    # level_holder.append(execution_time_list)
    acc_holder.append(new_acc_list)
    batch_count+=1

    # stanford.append(accuracy_list_stanford)
    # opencalai.append(accuracy_list_opencalai)
    # ritter.append(accuracy_list_ritter)
    # neuroner.append(accuracy_list_neuroner)

    counter=counter+1

print("-------------------------------------")
# print(acc_holder)
twics_f1=[]
twics_precision=[]
twics_recall=[]
for i in acc_holder:
    twics_f1.append(i[1])
    twics_precision.append(i[2])
    twics_recall.append(i[3])
print('twics_f1:', twics_f1)
print('twics_precision:', twics_precision)
print('twics_recall:',twics_recall)

print(sum(twics_f1)/len(twics_f1))
print(sum(twics_precision)/len(twics_precision))
print(sum(twics_recall)/len(twics_recall))

#f= 0.6846846846846847+0.6461538461538461+0.5912408759124088+0.6411764705882353+0.46956521739130436
#p= 0.6229508196721312+0.6222222222222222+0.6183206106870229+0.6411764705882353+0.43548387096774194
#r=  0.76+.672+0.5664335664335665+0.6411764705882353+0.5094339622641509


# print("Stanford\n",stanford)
# print("Ritter\n",ritter)
# print("Opencalai\n",opencalai)
# print("Neuroner\n",neuroner)

# print("-------------------------------------")

# partition=[0,1,2,3,4]
# for idx,level in enumerate(whole_level):

#     # f1=level
#     f1=[]
#     for tuple_level in level:
#         f1.append(tuple_level[0])
#     print(Z_scores, f1 , partition)
#     plt.plot( Z_scores,f1 ,marker='o' , label=partition[idx],alpha=0.5)
    
#     major_ticks = np.arange(-1.0, 1.2, 0.2)                                              
#     minor_ticks = np.arange(-1.0, 1.2, 0.1)                                               

#     ax.set_xticks(major_ticks)                                                       
#     ax.set_xticks(minor_ticks, minor=True)                                           
#     # ax.set_yticks(major_ticks)                                                       
#     # ax.set_yticks(minor_ticks, minor=True)                                           

#     # and a corresponding grid                                                       

#     ax.grid(which='both')                                                            

#     # or if you want differnet settings for the grids:                               
#     ax.grid(which='minor', alpha=0.2)                                                
#     ax.grid(which='major', alpha=0.5)     
#     ax.set_ylim([0.45,0.90])
#     ax.set_xlim([-1.1,1.1])
    
#     # tick_spacing = 0.1
#     # ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

#     plt.xlabel('Z Scores')
#     plt.ylabel('F1 Score')
#     plt.grid(True)
#     plt.legend(loc='upper right',title="Input Size")
#     plt.savefig("z-score-VS-f1-score-Mention.png")

# plt.show()

    # thefile = open('time_'+str(batch_size)+'.txt', 'w')
    # thefile2= open('number_of_processed_tweets'+str(batch_size)+'.txt', 'w')



    # for item in execution_time_list:
    #   thefile.write("%s\n" % item)

    # with open('accuracy_'+str(batch_size)+'.txt', 'w') as fp:
    #     fp.write('\n'.join('%s %s %s %s %s' % x for x in accuracy_list))


    # for item in tweets_been_processed_list:
    #   thefile2.write("%s\n" % item)
        



