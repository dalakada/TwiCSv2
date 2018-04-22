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

# sentence level
# whole_level=[
# [
# 1082.7956538044,
# 1050.6432401884,
# 1053.7217456599,
# 1012.5515266169,
# 1008.6493123934,
# 989.9630010678,
# 952.6808602644,
# 913.4404472567,
# 877.5589883683,
# 850.7370345206,
# 838.5608943759
# ],
# [83.5613470219,
# 86.1817237598,
# 91.4628130584,
# 89.9526238481,
# 86.8751099374,
# 92.2544695707,
# 95.2050934217,
# 97.7485665828,
# 99.7710068994,
# 101.1100325883,
# 101.5450283182

# ],
# [
# 282.5355908803,
# 286.9383684093,
# 304.265311919,
# 304.7142461953,
# 307.9814362368,
# 307.7235153447,
# 303.0300608151,
# 298.1720040698,
# 293.5781982817,
# 290.6551030755,
# 290.0621321868

# ],
# [0.7847898245,
# 0.7890123589,
# 0.7931519113,
# 0.7704950433,
# 0.7660793922,
# 0.771851223,
# 0.7642034553,
# 0.7541908589,
# 0.7508384564,
# 0.7468438025,
# 0.7450788343
# ]
# ]

whole_level=[[624.4496273381776, 593.30218438052, 594.1481509218495, 652.9262676551306, 605.6753390579645, 552.3884725429233, 597.3519978583432, 606.3851824297484, 543.6730550626653, 540.0956312505397, 529.8341210809203],
[48.1899348453864, 48.66714314099523, 51.57192729540457, 58.00438734328955, 52.16690381930308, 51.4769799239461, 59.69570202760152, 64.89014424264954, 61.811010822796185, 64.19032516588791, 64.15994496041087],
[162.9386337256632, 162.03517449871248, 171.56205915928956, 196.48968989695575, 184.9371813615319, 171.70633894746533, 190.00655916273732, 197.940747671422, 181.88014489644576, 184.52417727437214, 183.27209854023903],
[0.45258928748558247, 0.4455582429242619, 0.4472240830561037, 0.49684034801617233, 0.46001656860802365, 0.4306844979493904, 0.47917249084547664, 0.5006677369437788, 0.46516603767974085, 0.47413837483176313, 0.470768660874824]]
tweets_been_processed_list=[173400,
350484,
527834,
682913,
849446,
1028661,
1188145,
1338782,
1500195,
1657711,
1713105
]

tweets_been_processed_list=[100000,
200000,
300000,
400000,
500000,
600000,
700000,
800000,
900000,
1000000,
1035000]

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


ax2.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")
ax3.plot( tweets_been_processed_list, whole_level[1],marker='>' ,markersize=8,linewidth=1, label="OpenCalais")

ax2.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="TwitterNLP")
ax3.plot( tweets_been_processed_list, whole_level[2],marker='x' ,markersize=8,linewidth=1, label="TwitterNLP")

ax3.plot( tweets_been_processed_list, whole_level[3],marker='o' , markersize=8, linewidth=1,label="Stanford")

ax.set_ylim(500,700)  # outliers only
ax2.set_ylim(40, 220)
ax3.set_ylim(0,1)


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

tick_spacing_ax2 = 50
ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_ax2))

tick_spacing_x_axis = 400000
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))

plt.tick_params(axis='both', which='major', labelsize=12)

abc=f.text(0.03, 0.5, 'Tweet Processing Throughput',fontproperties=font_axis, ha='center', va='center', rotation='vertical')

ax.text(0.1, 0.3,'TwiCS', ha='center', va='center', transform=ax.transAxes,FontProperties=font_legend)

ax2.text(0.5, 0.64, 'TwitterNLP',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

ax2.text(0.15, -0.1, 'OpenCalais',ha='center', va='center', transform=ax2.transAxes,FontProperties=font_legend)

ax3.text(0.8, 0.55, 'Stanford',ha='center', va='center', transform=ax3.transAxes,FontProperties=font_legend)



plt.xlabel('Tweet in Input Stream',fontproperties=font_axis2)
# plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
ax2.grid(True)
ax3.grid(True)
ax.grid(True)
# plt.ylim((0.1,1.0))
# plt.legend(loc="lower right",ncol=4,frameon=False,prop=font_legend)
# plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
#            ncol=2,frameon=False,prop=font)
f.savefig("f1_score_us_vs_others7.pdf",dpi=1200,bbox_inches='tight',bbox_extra_artists=[abc])

plt.show()