

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



whole_level=[[0.8519527702089011, 0.8129927467675813, 0.7868601986249045, 0.7598216181779571, 0.7511879870747007, 0.7613941018766757, 0.7637091805298829], [0.5512476007677544, 0.5272073921971253, 0.5044339038977109, 0.48212105354962503, 0.4700960644561512, 0.4725825093103891, 0.4720684448917967], [0.7933333333333334, 0.6766043456291057, 0.6392676275808337, 0.605890603085554, 0.5537232671991753, 0.5514752706431756, 0.5520887322483673]]
tweets_been_processed_list=[500,1000,1500,2000,2500,3000,3200]


fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=24)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=15)

fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

plt.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="Phase1+Phase2+Classifier")
plt.plot( tweets_been_processed_list, whole_level[2],marker='>' ,markersize=8,linewidth=1, label="Phase1+Phase2")
plt.plot( tweets_been_processed_list, whole_level[1],marker='x' ,markersize=8,linewidth=1, label="Phase1")


tick_spacing = 0.1
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.tick_params(axis='both', which='major', labelsize=12)

plt.xlabel('Tweets in Input Stream',fontproperties=font_axis)
plt.ylabel('F1 Score',fontproperties=font_axis)#prop=20)
plt.grid(True)
plt.ylim((0.4,1.0))
plt.legend(loc="upper left",ncol=2,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
fig.savefig("system-variants.pdf",dpi=1200,bbox_inches='tight')

plt.show()

    # thefile = open('time_'+str(batch_size)+'.txt', 'w')
    # thefile2= open('number_of_processed_tweets'+str(batch_size)+'.txt', 'w')



    # for item in execution_time_list:
    #   thefile.write("%s\n" % item)

    # with open('accuracy_'+str(batch_size)+'.txt', 'w') as fp:
    #     fp.write('\n'.join('%s %s %s %s %s' % x for x in accuracy_list))


    # for item in tweets_been_processed_list:
    #   thefile2.write("%s\n" % item)
        




