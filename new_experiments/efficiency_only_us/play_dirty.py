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
whole_level=[[120.94201159477234, 239.9398741722107, 360.35196113586426, 478.82545709609985, 598.3550672531128, 716.7902600765228],
 [116.42404341697693, 244.16021251678467, 374.42199087142944, 508.1540787220001, 636.5065896511078, 766.4505112171173],
  [115.93312788009644, 244.16021251678467, 385.82897782325745, 547.2711956501007, 682.4377548694611, 830.5768580436707], 
  [118.62982392311096, 244.16021251678467, 385.82897782325745, 558.0065402984619, 721.2026655673981, 878.9471497535706], 
  [126.68061304092407, 244.16021251678467, 385.82897782325745, 558.0065402984619, 730.2533712387085, 902.7245206832886], 
  [119.01075839996338, 244.16021251678467, 385.82897782325745, 558.0065402984619, 730.2533712387085, 903.7556545734406]]


tweets_seen=[398437,598437,897878,1197775,1496932,1796324]


without_eviction_id=len(whole_level)-1
without_eviction=whole_level[without_eviction_id]


p1_holder=[]
p2_holder=[]

eviction_parameter_recorder=[0,1,2,3,4,5]

for idx,level in enumerate(whole_level[:-1]):
    p2=[]
    for i in range(len(level)):
        # p2.append(timing_max[i]-timing_sliced[idx][i])
        p2.append((without_eviction[i]-level[i])/without_eviction[i])


    tweets_been_proccessed=tweets_seen
    p2_holder.append(p2)

p2_holder_tranpsosed=list(map(list, zip(*p2_holder)))

eviction_parameter_recorder=eviction_parameter_recorder[:-1]
fig, ax = plt.subplots()
# ax2 = ax1.twinx()
# fig, ax1 = plt.subplots()
# plt.tick_params(axis='both', which='major', labelsize=12)
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }

matplotlib.rcParams.update(params)

markers=['s','>','x','o','D']

for idx,level in enumerate(p2_holder_tranpsosed[1:]):
    p1=level
    p2=p2_holder_tranpsosed[idx+1]

    ax.plot(eviction_parameter_recorder, p2,marker=markers[idx] ,markersize=8,linewidth=1,label=tweets_been_proccessed[idx+1])
    ax.set_xlabel('Eviction Parameter ',fontproperties=font_axis)
    ax.set_ylabel('P2',fontproperties=font_axis)
    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.grid(True)
    #plt.legend(loc='upper right',title="#of Input Tweets")
    legend=plt.legend(loc="upper right",ncol=1,frameon=False,prop=font_legend,title="# of Input Tweets")
    plt.setp(legend.get_title(),fontsize='18')
    plt.tick_params(axis='both', which='major', labelsize=12)

fig.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight')
plt.show()
