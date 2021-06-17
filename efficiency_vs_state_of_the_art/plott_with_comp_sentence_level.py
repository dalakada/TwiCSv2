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
# rc('font',**{'family':'dejavusans','serif':['Times']})
# rc('text', usetex=False)
# csfont = {'fontname':'DejaVu Sans Condensed'}

# ###################################### 1.03M plots ######################################
whole_level=[
[
1504.8556637058,
1474.0166767628,
1487.5693582649,
1427.492861504,
1429.4794534909,
1403.0959481,
1362.2105172903,
1322.4710465937,
1305.7584773903,
1282.9578994762,
1266.2010545884
], #TWICS-C
[
1504.8556637058,
1474.0166767628,
1510.6024076462,
1457.8514232959,
1473.08375261,
1470.0240742668,
1429.0044559318,
1399.0414116902,
1380.9075064637,
1358.6717217197,
1346.6964592156
], #TWICS-CE
# 957.5997480193,
# 918.9456178399,
# 887.7443107511,
# 865.9070840766,
# 867.5351121568,
# 854.0421928591,
# 820.7739165486,
# 789.5320993221,
# 761.808209342,
# 739.9418059677,
# 729.7979400888
# ],
[83.5613470219,
86.1817237598,
91.4628130584,
89.9526238481,
86.8751099374,
92.2544695707,
95.2050934217,
97.7485665828,
99.7710068994,
101.1100325883,
101.5450283182
], #OpenCalais
[
282.5355908803,
286.9383684093,
304.265311919,
304.7142461953,
307.9814362368,
307.7235153447,
303.0300608151,
298.1720040698,
293.5781982817,
290.6551030755,
290.0621321868

], #Twitter NLP
# [
# 100,
# 88,
# 91,
# 90,
# 90.59,
# 92.053,
# 93.284,
# 94.133,
# 93.79,
# 94.89,
# 95.4
# ], #NeuroNER
[
300.6560403,
287.3597756,
291.1624117,
288.7675,
289.6825076,
287.9537011,
283.5546099,
279.1268333,
277.1008342,
274.955837,
274.2830957
], #Gaguilar et al.

[
0.7847898245,
0.7890123589,
0.7931519113,
0.7704950433,
0.7660793922,
0.771851223,
0.7642034553,
0.7541908589,
0.7508384564,
0.7468438025,
0.7450788343
] #Stanford NER
]


# tweets_been_processed_list=[173400,
# 350484,
# 527834,
# 682913,
# 849446,
# 1028661,
# 1188145,
# 1338782,
# 1500195,
# 1657711,
# 1713105
# ]
tweets_been_processed_list=[
179202,
358646,
536083,
693100,
859600,
1038784,
1199605,
1351483,
1528241,
1693810,
1751118
]

fontPath = "/Users/satadisha/Downloads/abyssinica-sil/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=19)

# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=19)

# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_axis2 = fm.FontProperties(fname=fontPath, size=24)


# fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
font_legend = fm.FontProperties(fname=fontPath, size=18)


f, (ax,ax2,ax3,ax4) = plt.subplots(4, 1, sharex=True)

#fig, ax = plt.subplots()
params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

# print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

ax.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")
ax3.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")
ax2.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")
ax4.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")

ax.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiCS-CE")
ax3.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiCS-CE")
ax2.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiCS-CE")
ax4.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiCS-CE")


ax3.plot( tweets_been_processed_list, whole_level[2],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")
ax4.plot( tweets_been_processed_list, whole_level[2],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")

ax2.plot( tweets_been_processed_list, whole_level[3],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")
ax3.plot( tweets_been_processed_list, whole_level[3],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")
ax4.plot( tweets_been_processed_list, whole_level[3],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")


ax2.plot( tweets_been_processed_list, whole_level[4],marker='p' ,markersize=8,linewidth=1, label="Gaguilar et al.")
ax3.plot( tweets_been_processed_list, whole_level[4],marker='p' ,markersize=8,linewidth=1, label="Gaguilar et al.")
ax4.plot( tweets_been_processed_list, whole_level[4],marker='p' ,markersize=8,linewidth=1, label="Gaguilar et al.")

ax4.plot( tweets_been_processed_list, whole_level[5],marker='o' , markersize=8, linewidth=1,label="Stanford NER")




ax.set_ylim(1200,1525)  # outliers only
ax2.set_ylim(260, 310)
ax3.set_ylim(80, 110)
ax4.set_ylim(0,1)


ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax4.spines['top'].set_visible(False)


ax.xaxis.tick_top()
ax2.xaxis.tick_top()
ax.tick_params(labeltop='off')  # don't put tick labels at the top
ax.tick_params(bottom='off',axis='both', which='major', labelsize=12)
ax2.tick_params(top='off',axis='both', which='major', labelsize=12)
ax3.tick_params(top='off',axis='both', which='major', labelsize=12)  # don't put tick labels at the top
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

kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal


kwargs.update(transform=ax4.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


tick_spacing = 100
ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

tick_spacing_ax2 = 25
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_ax2))

tick_spacing_ax3 = 10
ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_ax3))

# tick_spacing_x_axis = 400000
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))

tick_spacing_x_axis = 400000
labels=['0','400K','800K','1.2M','1.6M']
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))
ax.set_xticklabels(labels)

plt.tick_params(axis='both', which='major', labelsize=12)

abc=f.text(0.03, 0.5, 'Tweet Processing Throughput',fontproperties=font_axis, ha='center', va='center', rotation='vertical')

ax.text(0.5, 0.48,'TwiCS-C', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

ax.text(0.8, 0.8,'TwiCS-CE', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

ax2.text(0.80, 0.84, 'Twitter NLP',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

ax3.text(0.15, -0.05, 'OpenCalais',ha='center', va='center', transform=ax3.transAxes,FontProperties=font_legend)

ax2.text(0.65, 0.11, 'Aguilar et al.',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

ax4.text(0.8, 0.55, 'Stanford NER',ha='center', va='center', transform=ax4.transAxes,FontProperties=font_legend)



plt.xlabel('Tweet (Sentences) in Input Stream D5',fontproperties=font_axis2)
# plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
ax.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)
# plt.ylim((0.1,1.0))
# plt.legend(loc="lower right",ncol=4,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
f.savefig("tweet-processing-throughput-w-gaguilar.pdf",dpi=1200,bbox_inches='tight',bbox_extra_artists=[abc])

# # plt.show()
# # ((1544/527)+(1471/521)+(1414/524)+(1366/521)+(1293/517)+(1259/510)+(1243/510))/7

# ##################################### NIST plots ######################################

# whole_level=[
# [
# 1544.3838415894395,
# 1471.9197961065597,
# 1414.9828846567912,
# 1366.9296488246123,
# 1293.8241411955635,
# 1259.5325419655237,
# 1243.2883729249393
# ], #TwiCS-C
# [
# 1597.6881085833895,
# 1562.142999985726,
# 1493.2881758329768,
# 1439.870040933887,
# 1358.9509684312811,
# 1321.3338122519517,
# 1301.6811098024182
# ], #TwiCS-CE
# [
# 527.0307783,
# 521.4092022,
# 524.3615233,
# 521.7318844,
# 517.3489646,
# 510.43652,
# 510.299068
# ], #TwitterNLP
# # [
# # 119.2457328,
# # 119.3225615,
# # 118.5742003,
# # 122.2119689,
# # 125.6602219,
# # 125.890977,
# # 111.2549642
# # ] #NeuroNER
# [
# 315.7460943,
# 295.9976864,
# 278.2922682,
# 279.7490325,
# 270.836901,
# 274.333813,
# 272.0863275
# ] #Gaguilar et al.
# ]

# tweets_been_processed_list=[
# 2408748,
# 4100185,
# 6506668,
# 8180959,
# 10783618,
# 12411229,
# 14144886
# ]

# fontPath = "/Users/satadisha/Downloads/abyssinica-sil/AbyssinicaSIL-R.ttf"
# font_axis = fm.FontProperties(fname=fontPath, size=16)

# # fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_axis2 = fm.FontProperties(fname=fontPath, size=16)


# # fontPath = "/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-R.ttf"
# font_legend = fm.FontProperties(fname=fontPath, size=16)

# f, (ax,ax2,ax3) = plt.subplots(3, 1, sharex=True)

# #fig, ax = plt.subplots()
# params = {
#    'text.usetex': False,
#     'legend.fontsize': 20,
#    'figure.figsize': [40, 400]
#    }
# matplotlib.rcParams.update(params)

# print("BITTI BITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTIBITTI")

# ax.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")
# ax3.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")
# ax2.plot( tweets_been_processed_list, whole_level[0],marker='s' ,markersize=8,linewidth=1, label="TwiCS-C")

# ax.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiLight-CE")
# ax3.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiLight-CE")
# ax2.plot( tweets_been_processed_list, whole_level[1],marker='s' ,markersize=8,linewidth=1, label="TwiLight-CE")



# ax2.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")
# ax3.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="Twitter NLP")


# # ax3.plot( tweets_been_processed_list, whole_level[3],marker='p' ,markersize=8,linewidth=1, label="NeuroNER")
# ax3.plot( tweets_been_processed_list, whole_level[3],marker='p' ,markersize=8,linewidth=1, label="Gaguilar et al.")



# ax.set_ylim(1000,1650)  # outliers only
# ax2.set_ylim(500, 550)
# ax3.set_ylim(250, 350)


# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
# ax3.spines['top'].set_visible(False)


# # ax.xaxis.tick_top()
# # ax2.xaxis.tick_top()

# # ax.tick_params(labeltop='off')  # don't put tick labels at the top
# ax.tick_params(top='off',bottom='off',axis='both', which='major', labelsize=12)
# ax2.tick_params(top='off',bottom='off',axis='both', which='major', labelsize=12)
# ax3.tick_params(top='off',axis='both', which='major', labelsize=12)  # don't put tick labels at the top
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

# tick_spacing_ax2 = 25
# ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_ax2))

# tick_spacing_ax3 = 50
# ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_ax3))

# tick_spacing_x_axis = 2000000
# labels=['0','2M','4M','6M','8M','10M','12M','14M']
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))
# ax.set_xticklabels(labels)

# plt.tick_params(axis='both', which='major', labelsize=12)

# abc=f.text(0.03, 0.5, 'Tweet Processing Throughput',fontproperties=font_axis, ha='center', va='center', rotation='vertical')

# ax.text(0.5, 0.45,'TwiCS-C', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

# ax.text(0.82, 0.6,'TwiCS-CE', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

# ax2.text(0.5, 0.62, 'Twitter NLP',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

# ax3.text(0.15, 0.8, 'Aguilar et al.',ha='center', va='center', transform=ax3.transAxes,FontProperties=font_legend)




# plt.xlabel('Tweet Sentences in Input Stream D6 (NIST 2011)',fontproperties=font_axis)

# # plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
# ax.grid(True)
# ax2.grid(True)
# ax3.grid(True)

# f.savefig("tweet-processing-throughput-NIST-gaguilar.pdf",dpi=1200,bbox_inches='tight',bbox_extra_artists=[abc])

# plt.show()










