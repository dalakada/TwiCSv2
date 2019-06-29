##whole_level[0] ours
##whole_level[1] ritter 
#[2] stanford
#[3] calai
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


###################################### 1.03M plots ######################################
# whole_level=[
# [
# 1126.3322818224,
# 958.1404466066,
# 853.3159010131,
# 835.7372111007,
# 770.4920114716,
# 747.1900792419,
# 760.7222558938,
# 771.8006909649,
# 767.7461716009,
# 763.934412098,
# 759.3073067169


# ],
# [62.7163088052,
# 57.2848160185,
# 55.5015980127,
# 50.6254109385,
# 46.609644083,
# 44.5478540794,
# 43.7548633233,
# 44.7345889811,
# 45.7837521867,
# 46.9585495297,
# 47.2156625961
# ],

# [246.5864401214,
# 222.1348208823,
# 221.1447836517,
# 221.1936121679,
# 207.8910770056,
# 203.4615073984,
# 208.5523298916,
# 216.102775894,
# 217.5802910369,
# 220.6285003371,
# 221.5740757932
# ],
# [
# 0.6079677158,
# 0.5429530813,
# 0.448010302,
# 0.4126238887,
# 0.372648492,
# 0.3759523934,
# 0.3875743132,
# 0.404436624,
# 0.4147277694,
# 0.4234270353,
# 0.4254905788

# ],

# ]


# tweets_been_processed_list=[100000,200000,300000,400000,500000,600000,700000,800000,900000,1000000,1035000]


# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_axis = fm.FontProperties(fname=fontPath, size=19)


# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_axis2 = fm.FontProperties(fname=fontPath, size=24)

# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_legend = fm.FontProperties(fname=fontPath, size=18)


# f, (ax, ax2,ax3) = plt.subplots(3, 1, sharex=True)

# #fig, ax = plt.subplots()
# params = {
#    'text.usetex': False,
#     'legend.fontsize': 20,
#    'figure.figsize': [40, 400]
#    }
# matplotlib.rcParams.update(params)

# print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

# ax.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")
# ax3.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")
# ax2.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")


# ax2.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")
# ax3.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")

# ax2.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="TwitterNLP")
# ax3.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="TwitterNLP")

# ax3.plot( tweets_been_processed_list, whole_level[3],marker='o' , markersize=8, linewidth=1,label="Stanford")

# ax.set_ylim(700,1200)  # outliers only
# ax2.set_ylim(30, 290)
# ax3.set_ylim(0,1)


# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax3.spines['top'].set_visible(False)


# ax.xaxis.tick_top()
# ax2.xaxis.tick_top()
# ax.tick_params(labeltop='off')  # don't put tick labels at the top
# ax.tick_params(labelbottom='off',axis='both', which='major', labelsize=12)
# ax2.tick_params(labeltop='off',axis='both', which='major', labelsize=12)
# ax3.tick_params(labeltop='off',axis='both', which='major', labelsize=12)  # don't put tick labels at the top
#   # don't put tick labels at the top
# # ax2.xaxis.tick_bottom()
# ax3.xaxis.tick_bottom()

# d = 0.01  # how big to make the diagonal lines in axes coordinates
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


# tick_spacing = 100
# ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# tick_spacing = 50
# ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
# plt.tick_params(axis='both', which='major', labelsize=12)

# abc=f.text(0.03, 0.5, 'Mention Discovery Throughput',fontproperties=font_axis, ha='center', va='center', rotation='vertical')

# ax.text(0.2, 0.7,'TwiCS', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

# ax2.text(0.5, 0.45, 'TwitterNLP',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

# ax2.text(0.15, -0.05, 'OpenCalais',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

# ax3.text(0.8, 0.15, 'Stanford',ha='center', va='center', transform=ax3.transAxes,FontProperties=font_legend)



# plt.xlabel('Tweets in Input Stream',fontproperties=font_axis2)
# # plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
# ax2.grid(True)
# ax3.grid(True)
# ax.grid(True)
# # plt.ylim((0.1,1.0))
# # plt.legend(loc="lower right",ncol=4,frameon=False,prop=font_legend)
# # plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
# #            ncol=2,frameon=False,prop=font)
# f.savefig("entity-discovery-throughput.pdf",dpi=1200,bbox_inches='tight',bbox_extra_artists=[abc])

# plt.show()



###################################### NIST plots ######################################
whole_level=[
[
551.0474854353778,
543.3865623844708,
542.7947612687782,
582.6695676285368,
584.019541565539,
582.6996731123535,
575.2829616204824
],
[
93.97489671,
95.22811367,
96.80831331,
90.85375183,
95.61161668,
94.13769581,
93.77343898
],

[
66.80165397,
74.46562949,
78.24144147,
71.4210247,
79.75171875,
81.88712952,
76.02595178
]

]


tweets_been_processed_list=tweets_been_processed_list=[
1140333,
2361123,
3582278,
4723393,
5932183,
7190180,
9002784
]


fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=19)


fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis2 = fm.FontProperties(fname=fontPath, size=24)

fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=18)


f, (ax, ax2,ax3) = plt.subplots(3, 1, sharex=True)

#fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

ax.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")
ax3.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")
ax2.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS")


ax2.plot( tweets_been_processed_list, whole_level[1],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")
ax3.plot( tweets_been_processed_list, whole_level[1],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")

ax3.plot( tweets_been_processed_list, whole_level[2],marker='o' , markersize=8, linewidth=1,label="NeuroNER")

ax.set_ylim(500,600)  # outliers only
ax2.set_ylim(85, 100)
ax3.set_ylim(60,85)


ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)


ax.xaxis.tick_top()
ax2.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax.tick_params(labelbottom='off',axis='both', which='major', labelsize=12)
ax2.tick_params(labeltop='off',axis='both', which='major', labelsize=12)
ax3.tick_params(labeltop='off',axis='both', which='major', labelsize=12)  # don't put tick labels at the top
  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()
ax3.xaxis.tick_bottom()

d = 0.01  # how big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


tick_spacing = 50
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
tick_spacing = 10
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

tick_spacing_x_axis = 2000000
labels=['2M','4M','6M','8M','10M']
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))
ax.set_xticklabels(labels)

plt.tick_params(axis='both', which='major', labelsize=12)

abc=f.text(0.03, 0.5, 'Mention Discovery Throughput',fontproperties=font_axis, ha='center', va='center', rotation='vertical')

ax.text(0.2, 0.6,'TwiCS', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

ax2.text(0.30, 0.55, 'Twitter NLP',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

ax3.text(0.8, 0.52, 'NeuroNER',ha='center', va='center', transform=ax3.transAxes,FontProperties=font_legend)



plt.xlabel('Tweets in Input Stream D6 (NIST 2011)',fontproperties=font_axis2)
# plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
ax2.grid(True)
ax3.grid(True)
ax.grid(True)
# plt.ylim((0.1,1.0))
# plt.legend(loc="lower right",ncol=4,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
f.savefig("entity-discovery-throughput-NIST.pdf",dpi=1200,bbox_inches='tight',bbox_extra_artists=[abc])

plt.show()