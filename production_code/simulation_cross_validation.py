
#import SatadishaModule as phase1
import SatadishaModule_final_trie as phase1

# import phase2_Trie_baseline_reintroduction as phase2
import phase2_Trie_reintroduction as phase2

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
from scipy import spatial
# from sklearn.decomposition import PCA as sklearnPCA
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
import adjustText
# from pandas.tools.plotting import parallel_coordinates

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

#input names: 3K; deduplicated--> politics; malcolm; 1M
# input_name="D1"
# tweets_unpartitoned=pd.read_csv("tweets_3k_annotated.csv",sep =',')

# input_name="D2"
#tweets_unpartitoned=pd.read_csv("malcolmx.csv",sep =',')
# tweets_unpartitoned=pd.read_csv("deduplicated_test.csv",sep =';')

# /Users/satadisha/Documents/GitHub/tweets_1million_for_others.csv #---- for my Mac
# /home/satadisha/Desktop/GitProjects/data/tweets_1million_for_others.csv #---- for my lab PC
tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/tweets_1million_for_others.csv",sep =',')
tweets_unpartitoned=tweets_unpartitoned[:200000:]

print("***",len(tweets_unpartitoned))
print('Tweets are in memory...')
batch_size=10000
# batch_size=1000

# batch_size=len(tweets_unpartitoned)

# Z_scores=[-1.0,-0.9,-0.8,-0.7,-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# # 
# # Z_scores=[-0.2]
# execution_time_list=[]
# accuracy_list=[]
# batch_size_recorder=[]

# whole_level=[]


#z_score=-0.078      #-----20K
#z_score=-0.8      #-----3K
#z_score=-0.09      #-----50K
#z_score=-0.078         #-----50K, multiple batches
#z_score=-0.08         #-----deduplicated_tweets,
#z_score=-0.1119        #-----tweets_1million_for_others, 200K


#print(entity_level_arr)
#kf = KFold(n_splits=5,random_state=1000) 
# for train_ind,test_ind in kf.split(tweets_unpartitoned):
#     print(train_ind,test_ind)
#     tweets=tweets_unpartitoned.iloc[train_ind]
#     test=tweets_unpartitoned.iloc[test_ind]
#     print("**********************")
#     # xtrain = df[train_ind:]
#     # xtest = df[test_ind:]

    # level_holder=[]

    # for z_score in Z_scores:
        # batch_size_ratio_float= batch_size_ratio/100.0
        # # print(batch_size_ratio_float)
        # batch_size=len(tweets)*batch_size_ratio_float
        # batch_size_recorder.append(batch_size)
# for iter in range(10):
#     print('run: ',str(iter))

# tweets = shuffle(tweets_unpartitoned)
tweets=tweets_unpartitoned
z_score=-0.1119
entity_level_arr=[[-1]*20]*20
mention_level_arr=[[-1]*20]*20
sentence_level_arr=[[-1]*20]*20
Phase1= phase1.SatadishaModule()
Phase2 = phase2.EntityResolver()
total_time=0
execution_time_list=[]
tweets_been_processed_list=[]
tweets_been_processed=0
length=len(tweets)
val=math.ceil(length/batch_size)-1
count=0
#reintroduction_threshold_array=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
reintroduction_threshold_array=[0.0]
iter=0
print('run: ',str(iter))
candidates_to_annotate=[]
for reintroduction_threshold in reintroduction_threshold_array:

    for g, tweet_batch in tweets.groupby(np.arange(length) //batch_size):

        tuple_of= Phase1.extract(tweet_batch,g)
        tweet_base=tuple_of[0]
        #tweet_base.to_csv('tweet_base.csv' ,sep=',',   encoding='utf-8')

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
        print(len(tweet_base))
        print (g,' ','Produced')
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
        #reintroduction_threshold=0.2
        Phase2.executor(tweet_base,candidate_base,phase2stopwordList,z_score,reintroduction_threshold,tweet_base)
        # candidates_to_annotate_in_iter=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,z_score,reintroduction_threshold,tweet_base)
        # candidates_to_annotate+=candidates_to_annotate_in_iter
        # entity_level_arr=Phase2.entity_level_arr
        # mention_level_arr=Phase2.mention_level_arr
        # sentence_level_arr=Phase2.sentence_level_arr

        #print('::',len(phase2TweetBase))
        accuracy_list=Phase2.finish()
        time_out=time.time()
        elapsedTime= time_out-time_in
        total_time+=elapsedTime
        execution_time_list.append(total_time)
        print(elapsedTime,total_time)
        print(g,' ','Consumed')
        print("**********************************************************")

#-----------------------------------------------------This part is for propagation estimates-------------------------------
# # column_headers=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
# # #Taking propagation estimates
# # entity_dataframe=pd.DataFrame(entity_level_arr, columns=column_headers)
# # entity_dataframe.to_csv('entity-level-estimates-'+str(iter)+'.csv')
# # mention_dataframe=pd.DataFrame(mention_level_arr, columns=column_headers)
# # mention_dataframe.to_csv('mention-level-estimates-'+str(iter)+'.csv')
# # incomplete_sentence_dataframe=pd.DataFrame(sentence_level_arr, columns=column_headers)
# # incomplete_sentence_dataframe.to_csv('sentence-level-estimates-'+str(iter)+'.csv')
    
#         candidate_records=pd.read_csv("candidate_base_new.csv",sep =',')
#         bad_conversions=['republican','republicans','dad']
        
#         if (g==0):
#             ambiguous_candidates=candidate_records[(candidate_records['status']=='a')].candidate.tolist()
#             # ambiguous_candidates=candidate_records[(candidate_records['probability']<0.75)&(candidate_records['probability']>0.7)].candidate.tolist()
#             print(ambiguous_candidates)
#         print("+++>",len(ambiguous_candidates))
#         y=candidate_records['status']
#         candidate_records['normalized_length']=candidate_records['length']/(candidate_records['length'].max())
#         x=candidate_records[['normalized_length','normalized_cap','normalized_capnormalized_substring-cap','normalized_s-o-sCap','normalized_all-cap','normalized_non-cap','normalized_non-discriminative']]

#         if(g<1):
#             tsne = TSNE(n_components=2, perplexity=28,  learning_rate=50,
#              early_exaggeration=4.0, n_iter=5000,
#                     min_grad_norm=0, init='pca', method='exact', verbose=1)
#         else:
#             tsne = TSNE(n_components=2, perplexity=38,  learning_rate=100,
#              early_exaggeration=4.0, n_iter=5000,
#                     min_grad_norm=0, init='random', method='exact', verbose=1)

#         for j in range(5):

#             # s=2

#             transformed = tsne.fit_transform(x)
#             # print(len(transformed),len(y))

#             plt.figure()
#             plt.scatter(transformed[y=='g'][:, 0], transformed[y=='g'][:, 1], label='Entity', c='lightsalmon')
#             # # # for i in range(len(transformed[y=='g'])):
#             # # #     #print(i)
#             # # #     plt.annotate(str(i), (transformed[y=='g'][i:(i+1),0],transformed[y=='g'][i:(i+1),1]))

#             # # # #print(transformed[y==2])
#             plt.scatter(transformed[y=='a'][:, 0], transformed[y=='a'][:, 1], label='Ambiguous', c='cyan')
#             # # for i in transformed[y=='a']:
#             # #   print(transformed.index(i))
#             #   #plt.annotate(str(i+(len(transformed[y=='a']))), (transformed[y=='a'][i:(i+1),0],transformed[y=='a'][i:(i+1),1]))

#             plt.scatter(transformed[y=='b'][:, 0], transformed[y=='b'][:, 1], label='Non-Entity', c='lightgreen')
#             # # # for i in range(len(transformed[y=='b'])):
#             # # #   #print(i)
#             # # #   plt.annotate(str(i+(len(transformed[y=='b']))), (transformed[y=='b'][i:(i+1),0],transformed[y=='b'][i:(i+1),1]))

#             if(g>0):
#                 # candidates_for_annotation=candidate_records[(candidate_records['candidate'].isin(candidates_to_annotate))&(candidate_records['status']!='a')].candidate.tolist()
#                 candidates_for_annotation=candidate_records[(candidate_records['candidate'].isin(candidates_to_annotate))].candidate.tolist()
#             else:
#                 candidates_for_annotation=ambiguous_candidates

#             # labels=candidate_records[(candidate_records['candidate'].isin(candidates_for_annotation))].status.tolist()
            
#             candidates_for_annotation = [x for x in candidates_for_annotation if x not in bad_conversions]
#             # s=s+2
#             texts=[]
#             for i in range(len(candidates_for_annotation)):
#                 # if((candidate_records[candidate])['status']=='g'):
#                 candidate=candidates_for_annotation[i]
#                 # if(labels[i]=='g'):
#                 #     # label='Entity'
#                 #     c='lightsalmon'
#                 # elif(labels[i]=='a'):
#                 #     # label='Ambiguous'
#                 #     c='cyan'
#                 # else:
#                 #     # label='Non-Entity'
#                 #     c='lightgreen'
#                 a_index=candidate_records.index[(candidate_records['candidate']==candidate)][0]
#                 # plt.plot(transformed[a_index:(a_index+1),0],transformed[a_index:(a_index+1),1])
#                 texts.append(plt.text(transformed[a_index:(a_index+1),0],transformed[a_index:(a_index+1),1], candidate, fontsize='xx-small', weight= 'bold'))
#                 # texts.append(plt.annotate(candidate, (transformed[a_index:(a_index+1),0],transformed[a_index:(a_index+1),1]), fontsize='xx-small', weight= 'bold'))
#             #     #print(a_index)
                

#             adjustText.adjust_text(texts, force_points=0.2, force_text=0.2, expand_points=(1,1), expand_text=(1,1),arrowprops=dict(arrowstyle="-|>", color='black', lw=0.5))
#             plt.xlabel('Transformed X-axis')
#             plt.ylabel('Transformed Y-axis')
#             plt.legend(fontsize = 'x-small')

#             plt.title("t-SNE plot of Entity Candidates for "+input_name+" (iteration "+str(g)+")")

#             plt.savefig('tsne_'+input_name+'_'+str(g)+'.png', dpi = 600)
#             plt.show()

#         ambiguous_candidates=candidate_records[(candidate_records['status']=='a')].candidate.tolist()



#--------------------------------------------------------------------IGNORE from here!!!!!!!!!!!!!!!!!!!!!!--------------------------------------------------------------------

#print(len(phase2TweetBase))


# level_holder.append(execution_time_list)
#level_holder.append(accuracy_list)
        # level_holder.append(tweets_been_processed_list)



    # whole_level.append(copy.deepcopy(level_holder))

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
#         if each_zscore_tuple[0]>maximum:
#             maximum=each_zscore_tuple[0]
#             best_z_score=each_zscore_tuple[1]

#     print(best_z_score)
#     best_z_scores.append(best_z_score)

# #validation
# counter=0
# new_acc_list=[]

# stanford=[]
# ritter=[]
# opencalai=[]

# acc_holder=[]
# for train_ind,test_ind in kf.split(tweets_unpartitoned):

#     tweets=tweets_unpartitoned.iloc[test_ind]
#         # batch_size_ratio_float= batch_size_ratio/100.0
#         # # print(batch_size_ratio_float)
#         # batch_size=len(tweets)*batch_size_ratio_float
#         # batch_size_recorder.append(batch_size)
#     val=math.ceil(length/batch_size)-1
#     Phase1= phase1.SatadishaModule()
#     Phase2 = phase2.EntityResolver()
#     total_time=0
#     execution_time_list=[]
#     tweets_been_processed_list=[]
#     tweets_been_processed=0
#     batch_size=500

    
#     tuple_of= Phase1.extract(tweets,0)
#     tweet_base=tuple_of[0]

#     candidate_base=tuple_of[1]
#     phase2stopwordList=tuple_of[4]


#     # print('len of tweet_base = '  len(tweet_base))
#     elapsedTime= tuple_of[3] - tuple_of[2]
#     total_time+=elapsedTime
#     print(elapsedTime,total_time)
#     print ('Produced')
#     print("**********************************************************")

#     time_in=time.time()

#     tweets_been_processed=tweets_been_processed+len(tweet_base)
#     tweets_been_processed_list.append(tweets_been_processed)
#     phase2TweetBase=Phase2.executor(tweet_base,candidate_base,phase2stopwordList,best_z_scores[counter],tweet_base)
#     new_acc_list=Phase2.finish()
#     accuracy_list_stanford,accuracy_list_opencalai,accuracy_list_ritter=Phase2.finish_other_systems()
#     time_out=time.time()
#     elapsedTime= time_out-time_in
#     total_time+=elapsedTime
#     execution_time_list.append(total_time)

#     print(elapsedTime,total_time)
#     # print(g,' ','Consumed')
#     # print("**********************************************************")
#     #print(len(phase2TweetBase))


#     # level_holder.append(execution_time_list)
#     acc_holder.append(new_acc_list)
#     stanford.append(accuracy_list_stanford)
#     opencalai.append(accuracy_list_opencalai)
#     ritter.append(accuracy_list_ritter)

#     counter=counter+1

# print("-------------------------------------")
# print(acc_holder)
# print("Stanford\n",stanford)
# print("Ritter\n",ritter)
# print("Opencalai\n",opencalai)

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
        



