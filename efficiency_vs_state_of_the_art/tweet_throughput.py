#[0] is ours
##whole_level[1] calais 
#[2] ritter
#[3] stanford
import datetime
from threading import Thread
import random
import math
from queue  import Queue
import pandas as pd 
import warnings
import numpy as np
import time
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
[
624.4496273382,
599.538489739,
598.893825896,
593.0779039889,
593.7100842157,
577.428132923,
561.2754353931,
545.8337188619,
526.4669523172,
513.1998487798,
506.630081448

],
[89.5640633394,
118.4711199253,
96.3090415825,
69.4446894312,
70.4703616003,
63.4164763923,
91.2659496508,
72.7096369201,
90.8050266542,
74.6798459723,
63.739368563
],
[
162.9386337256,
163.7383551941,
172.9323870302,
178.479101259,
181.2837050482,
179.489753385,
178.5312757034,
178.1750899369,
176.124022846,
175.3352080522,
175.2457127925
],
[0.4525892875,
0.4502415853,
0.4507962226,
0.4512990927,
0.4509288361,
0.4502073412,
0.4502332785,
0.4506728408,
0.4504445161,
0.450527144,
0.4501071098
]
]


tweets_been_processed_list=[100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000,1035000]


fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=24)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=18)

fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

plt.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")
plt.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")
plt.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="TwitterNLP")
plt.plot( tweets_been_processed_list, whole_level[3],marker='o' , markersize=8, linewidth=1,label="Stanford")


tick_spacing = 50
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.tick_params(axis='both', which='major', labelsize=12)

plt.xlabel('# of Seen Tweets',fontproperties=font_axis)
plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
plt.grid(True)
# plt.ylim((0.1,1.0))
plt.legend(loc="lower right",ncol=2,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
fig.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight')

plt.show()