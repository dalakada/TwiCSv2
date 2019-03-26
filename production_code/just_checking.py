import string
import numpy as np
import pandas as pd
import ast
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.text import OffsetFrom
from matplotlib.patches import Ellipse
import trie as trie
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from itertools import groupby
from operator import itemgetter
import collections 



bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_runs.csv",sep =',', keep_default_na=False)

lst=[20,40,60,80,100]

for index,row in bigger_tweet_dataframe.iterrows():

    tweetText=str(row['TweetText'])

    output_reintroduction_theshold_list=[]

    for elem in range(len(lst)):

        threshold=lst[elem]


        multipass_output_list=ast.literal_eval(str(row['output_col_'+str(threshold)]))
        multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
        multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

        output_reintroduction_theshold_list.append(multipass_output_list_flat)

    if not all(collections.Counter(x) == collections.Counter(output_reintroduction_theshold_list[0]) for x in output_reintroduction_theshold_list):
    	for output_list in output_reintroduction_theshold_list:
    		print(output_list)

    	print('================================================')

['jose','chris cornell','potus','morgan','nationalism','religion world tour','rust belt','trumps','spicer']