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
import matplotlib
from matplotlib import rc
import matplotlib.font_manager as fm
fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=24)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=18)

whole_level=[[0.8112379280070237, 0.76951444376152422, 0.75372340425531914, 0.74427940586109997, 0.74272930648769575, 0.73903743315508019], 
[0.8112379280070237, 0.77074370006146287, 0.75478723404255321, 0.74588518667201931, 0.74432726110578462, 0.74037433155080212], 
[0.8112379280070237, 0.77074370006146287, 0.75797872340425532, 0.74789241268566842, 0.74656439757110893, 0.74251336898395726],
[0.8112379280070237, 0.77074370006146287, 0.75797872340425532, 0.74789241268566842, 0.74656439757110893, 0.74278074866310162], 
[0.8112379280070237, 0.77074370006146287, 0.75797872340425532, 0.74789241268566842, 0.74656439757110893, 0.74278074866310162],
[0.8112379280070237, 0.77074370006146287, 0.75797872340425532, 0.74789241268566842, 0.74656439757110893, 0.74278074866310162]]


tweets_seen=[398437,598437,897878,1197775,1496932,1796324]


without_eviction_id=len(whole_level)-1
without_eviction=whole_level[without_eviction_id]


p1_holder=[]
p2_holder=[]

eviction_parameter_recorder=[0,1,2,3,4,5]

for idx,level in enumerate(whole_level[:-1]):

    p1_divided=[]
    
    for i in range(len(level)):
        p1_divided.append(level[i]/without_eviction[i])


    tweets_been_proccessed=tweets_seen
    p1_holder.append(p1_divided)

# p1_holder_tranpsosed=list(map(list, zip(*p1_holder)))
p1_holder_tranpsosed=list(map(list, zip(*p1_holder)))




eviction_parameter_recorder=eviction_parameter_recorder[:-1]
fig, ax = plt.subplots()

params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }

matplotlib.rcParams.update(params)

markers=['s','>','x','o','D']


for idx,level in enumerate(p1_holder_tranpsosed[1:]):
    p1=level

    ax.plot(eviction_parameter_recorder, p1,marker=markers[idx] ,markersize=8,linewidth=1,label=tweets_been_proccessed[idx+1])
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xlabel('Eviction Parameter ',fontproperties=font_axis)
    ax.set_ylabel('P1',fontproperties=font_axis)
    plt.grid(True)
    legend=plt.legend(loc="lower right",ncol=1,frameon=False,prop=font_legend,title="# of Input Tweets")
    plt.setp(legend.get_title(),fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=12)

fig.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight')

plt.show()


