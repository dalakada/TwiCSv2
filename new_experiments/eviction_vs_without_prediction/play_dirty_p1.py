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
font_axis = fm.FontProperties(fname=fontPath, size=28)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=22)

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

    tick_spacing_y = 0.001
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_y))



    ax.set_xlabel('Eviction Parameter ',fontproperties=font_axis)
    ax.set_ylabel('P1',fontproperties=font_axis)

    # ax.text(0.8, 0.6, '598,437',ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)
    # ax.text(0.8, 0.5, '897,878',ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)
    # ax.text(0.8, 0.4, '1,197,775',ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)
    # ax.text(0.8, 0.3, '1,496,932',ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)
    # ax.text(0.8, 0.2, '1,796,324',ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

    my_label=['598,437','897,878','1,197,775','1,496,932','1,796,324']

    plt.grid(True)
    legend=plt.legend(loc="lower right",ncol=1,frameon=False,prop=font_legend,title="# of Input Tweets",labels=my_label)
    # legend.get_texts()[0].set_text('598,437')
    # legend.get_texts()[-1].set_text('897,878')
    # legend.get_texts()[2].set_text('1,197,775')
    # legend.get_texts()[3].set_text('1,496,932')
    # legend.get_texts()[4].set_text('1,796,324')

    plt.setp(legend.get_title(),fontsize='18')

    plt.tick_params(axis='both', which='major', labelsize=20)

fig.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight')

plt.show()


