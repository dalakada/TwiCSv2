import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import spline
import time


tweets=pd.read_csv("final_incomplete_tweets.csv",sep =',')

tweets_org=tweets[['inverted_loss','entry_vs_tweet_seen_ratio']].groupby(["entry_vs_tweet_seen_ratio"]).mean()

# x = tweets_org['inverted_loss']
# y = tweets_org.index.values
tweets_org=tweets_org.reset_index()

x= tweets_org['entry_vs_tweet_seen_ratio'].tolist()
y = tweets_org['inverted_loss'].tolist()
# print(y)

# Create a canvas to place the subgraphs
canvas = plt.figure()
rect = canvas.patch
rect.set_facecolor('white')


x_sm = np.array(x)
y_sm = np.array(y)

x_smooth = np.linspace(x_sm.min(), x_sm.max(), 100)
y_smooth = spline(x, y, x_smooth)

print(y_sm,y_smooth,y_sm.min(),y_sm.max())
# Define the matrix of 1x1 to place subplots
# Placing the plot1 on 1x1 matrix, at pos 1
sp1 = canvas.add_subplot(1,1,1, axisbg='w')
#sp1.plot(x, y, 'red', linewidth=2)
sp1.plot(x_smooth, y_smooth, 'red', linewidth=1)

# Colorcode the tick tabs 
sp1.tick_params(axis='x', colors='red')
sp1.tick_params(axis='y', colors='red')

# Colorcode the spine of the graph
sp1.spines['bottom'].set_color('r')
sp1.spines['top'].set_color('r')
sp1.spines['left'].set_color('r')
sp1.spines['right'].set_color('r')

# Put the title and labels
sp1.set_title('matplotlib example 1', color='red')
sp1.set_xlabel('matplot x label', color='red')
sp1.set_ylabel('matplot y label', color='red')

# Show the plot/image
plt.tight_layout()
plt.grid(alpha=0.8)
plt.savefig("example6.eps")
plt.show()


