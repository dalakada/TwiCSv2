import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import copy
import matplotlib.ticker as ticker
import matplotlib
from matplotlib import rc
import matplotlib.font_manager as fm
from matplotlib import collections as matcoll
import warnings

warnings.filterwarnings("ignore")

batchNumber=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
incoming=[90.7, 93.0, 96.2, 87.9, 89.6, 89.1, 79.8, 78.7, 79.2, 89.7, 94.5, 87.0, 88.8, 79.7, 80.0, 80.9, 92.8, 89.6, 87.0, 86.3, 60.1]
phase_II=[90.7, 95.5, 100.3, 90.6, 93.1, 92.3, 82.3, 88.3, 82.7, 94.1, 97.0, 92.4, 91.3, 83.3, 85.9, 86.4, 98.0, 93.4, 90.7, 91.3, 66.0]
phase_II_ritter=[90.7, 94.5, 101.3, 90.26, 93.51, 94.3, 84.3, 89.3, 83.7, 96.1, 100.02, 92.5, 92.3, 83.7, 86.3, 86.2, 98.7, 93.8, 90.4, 91.1, 66.23]
increment=[]

lines = []
for batch in range(1,len(incoming)):
	incoming_memory=incoming[batch]
	memory_w_rescan=phase_II_ritter[batch]
	pair = [(batch, incoming_memory), (batch, memory_w_rescan)]
	lines.append(pair)

	overhead=(memory_w_rescan-incoming_memory)/incoming_memory*100
	increment.append(overhead)
	# print(batch, overhead)

print(sum(incoming))
print(sum(phase_II_ritter)/len(phase_II_ritter))
print(sum(increment)/len(increment))

lines.append(pair)

fontPath = "/Users/satadisha/Downloads/abyssinica-sil/AbyssinicaSIL-R.ttf"
font_axis = fm.FontProperties(fname=fontPath, size=16)

fig, ax = plt.subplots()

params = {
   'text.usetex': False,
    'legend.fontsize': 20,
   'figure.figsize': [40, 400]
   }
matplotlib.rcParams.update(params)

tick_spacing_x_axis = 5
labels=['0','5','10','15','20']
ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing_x_axis))
ax.set_xticklabels(labels)

linecoll = matcoll.LineCollection(lines, color='red', linewidth=2)

ax.plot( batchNumber, phase_II[1:],marker='s' ,markersize=8, color='blue',linewidth=1, label="Incoming+Incomplete")

ax.plot( batchNumber, incoming[1:],marker='p' ,markersize=8, color='blue',linewidth=1, label="Incoming")

ax.add_collection(linecoll)

ax.legend(prop={"size":12})
plt.ylabel('Memory Consumption (Mega Bytes)',fontproperties=font_axis)
plt.xlabel('Incoming Batch of Input Stream D6',fontproperties=font_axis)
# plt.ylabel('Tweet Throughput',fontproperties=font_axis)#prop=20)
ax.grid(True)

plt.show()