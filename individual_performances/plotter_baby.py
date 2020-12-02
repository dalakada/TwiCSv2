

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



whole_level=[
[0.8779342723004694, 0.8281338627339762, 0.7853820598006644, 0.767388781431335, 0.7772618107125927, 0.785632041981788],
[#0.8519527702089011, 
0.8129927467675813, 0.7868601986249045, 0.7598216181779571, 0.7311879870747007, 0.7413941018766757, 0.75037091805298829],  
[#0.7933333333333334, 
0.6766043456291057, 0.6392676275808337, 0.605890603085554, 0.5537232671991753, 0.5514752706431756, 0.5520887322483673],
[#0.5512476007677544, 
0.5272073921971253, 0.5044339038977109, 0.48212105354962503, 0.4700960644561512, 0.4725825093103891, 0.4720684448917967]]

tweets_been_processed_list=[2000,4000,6000,8000,10000,10500]
incomplete_tweets=[6441,
9692,
6674,
7461,
8534,
8932,
9731,
10285,
10493,
11848,
9244]
incomplete_tweets=[6474, 9725, 7185, 17494, 18634, 10266, 11500, 10715, 12395, 13307, 9641]
#tweets_been_processed_list=[179202, 358646, 536083, 693100, 859600, 1038784, 1199605, 1351483, 1528241, 1693810, 1751118]

# percent_incomplete= [ (sum(incomplete_tweets[0:(i+1)])/tweets_been_processed_list[i])*100 for i in range(len(tweets_been_processed_list))]
# percent_incomplete= [3.594267921116952, 4.498307523295952, 4.254378519744144, 4.367046602221902, 4.513959981386692, 4.5951805187603965, 4.790326815910237, 5.013011632406771, 5.119807674313148, 5.318837413877589, 5.47266169384359]

# iteration=[1,2,3,4,5,6,7,8,9,10,11]

# print(percent_incomplete)


fontPath = "/Users/satadisha/Downloads/abyssinica-sil/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=16)
font_legend = fm.FontProperties(fname=fontPath, size=10)

fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

# plt.plot( iteration, percent_incomplete,marker='s' ,markersize=8,linewidth=1)

# plt.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="Weak Tagger + Forward Scan(wClassifier)+ Rescan ")
# plt.plot( tweets_been_processed_list, whole_level[1],marker='o' ,markersize=8,linewidth=1, label="Weak Tagger + Forward Scan(wClassifier)")
# plt.plot( tweets_been_processed_list, whole_level[2],marker='>' ,markersize=8,linewidth=1, label="Weak Tagger + Forward Scan(w/oClassifier)")
# plt.plot( tweets_been_processed_list, whole_level[3],marker='x' ,markersize=8,linewidth=1, label="Weak Tagger")

plt.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="Phase I + Phase II(wClassifier)+ Rescan ")
plt.plot( tweets_been_processed_list, whole_level[1],marker='o' ,markersize=8,linewidth=1, label="Phase I + Phase II(wClassifier)")
plt.plot( tweets_been_processed_list, whole_level[2],marker='>' ,markersize=8,linewidth=1, label="Phase I + Phase II(w/oClassifier)")
plt.plot( tweets_been_processed_list, whole_level[3],marker='x' ,markersize=8,linewidth=1, label="Phase I")

# plt.grid(True)
# tick_spacing = 1.0
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.tick_params(axis='both', which='major', labelsize=12)

plt.xlabel('Tweets in Input Stream',fontproperties=font_axis)
plt.ylabel('F1 Score',fontproperties=font_axis)#prop=20)


plt.grid(True)
plt.ylim((0.4,1.0))
plt.legend(loc="upper right",ncol=1,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
fig.savefig("system-variants-TwiCS.pdf",dpi=1200,bbox_inches='tight')
# fig.savefig("percent-incomplete.pdf",dpi=1200,bbox_inches='tight')
plt.show()

    # thefile = open('time_'+str(batch_size)+'.txt', 'w')
    # thefile2= open('number_of_processed_tweets'+str(batch_size)+'.txt', 'w')



    # for item in execution_time_list:
    #   thefile.write("%s\n" % item)

    # with open('accuracy_'+str(batch_size)+'.txt', 'w') as fp:
    #     fp.write('\n'.join('%s %s %s %s %s' % x for x in accuracy_list))


    # for item in tweets_been_processed_list:
    #   thefile2.write("%s\n" % item)
        




