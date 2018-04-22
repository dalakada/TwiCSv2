
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
warnings.filterwarnings("ignore")
rc('font',**{'family':'dejavusans','serif':['Times']})
rc('text', usetex=False)
csfont = {'fontname':'DejaVu Sans Condensed'}

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


print("Our result startsss")
print(accuracy_list)
print("Our result endsss")
print('BURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBARENBURADAN ITIBAREN')

whole_level_transposed=list(map(list, zip(*whole_level)))



# whole_level=[[0.8503865393360618, 0.81530676786843781, 0.787321063394683, 0.75321888412017168, 0.74468085106382975, 0.75667893137098063, 0.75925636619278236], [0.58005698005698, 0.542741935483871, 0.5365058670143416, 0.5432627913206537, 0.5358711566617862, 0.5556448372840498, 0.56], [0.7051899411449972, 0.649390243902439, 0.65203007518797, 0.6240241752707127, 0.6138749429484254, 0.6271474419482727, 0.6279498525073746], [0.5639344262295082, 0.5046439628482972, 0.5171032357473035, 0.48112470710752414, 0.4678724416944312, 0.44398766700924974, 0.4399111290648354]]
# tweets_been_processed_list=[500,1000,1500,2000,2500,3000,3200]


# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_axis = fm.FontProperties(fname=fontPath, size=24)

# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_legend = fm.FontProperties(fname=fontPath, size=18)

# fig, ax = plt.subplots()
# params = {
#    'text.usetex': False,
#     'legend.fontsize': 20,
#    'figure.figsize': [40, 400]
#    }
# matplotlib.rcParams.update(params)

# print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

# plt.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="Our System")
# plt.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="Stanford")
# plt.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="Ritter")
# plt.plot( tweets_been_processed_list, whole_level[3],marker='o' , markersize=8, linewidth=1,label="OpenCalais")


# tick_spacing = 0.1
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# plt.tick_params(axis='both', which='major', labelsize=12)

# plt.xlabel('# of Seen Tweets',fontproperties=font_axis)
# plt.ylabel('F1 Score',fontproperties=font_axis)#prop=20)
# plt.grid(True)
# plt.ylim((0.1,1.0))
# plt.legend(loc="lower right",ncol=2,frameon=False,prop=font_legend)
# # plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
# #            ncol=2,frameon=False,prop=font)
# fig.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight')

# plt.show()

    # thefile = open('time_'+str(batch_size)+'.txt', 'w')
    # thefile2= open('number_of_processed_tweets'+str(batch_size)+'.txt', 'w')



    # for item in execution_time_list:
    #   thefile.write("%s\n" % item)

    # with open('accuracy_'+str(batch_size)+'.txt', 'w') as fp:
    #     fp.write('\n'.join('%s %s %s %s %s' % x for x in accuracy_list))


    # for item in tweets_been_processed_list:
    #   thefile2.write("%s\n" % item)
        



