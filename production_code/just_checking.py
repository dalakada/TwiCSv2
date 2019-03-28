import string
import numpy as np
import pandas as pd
import ast
import os
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



# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_runs_with_annotation.csv",sep =',', keep_default_na=False)
# print(list(bigger_tweet_dataframe.columns.values))
# df1=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_0.csv",sep =',', keep_default_na=False)
# lst=[0,20,40,60,80,100,110]

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])

# for elem in lst:
# 	bigger_tweet_dataframe['output_col_'+str(elem)] = ''
# 	bigger_tweet_dataframe['output_col_'+str(elem)] = bigger_tweet_dataframe['output_col_'+str(elem)].apply(list)
# df1_grouped_df= (df1.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df1_grouped_df['tweetID']=df1_grouped_df['tweetID'].astype(int)
# df1_grouped_df_sorted=(df1_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df1_grouped_df_sorted.tweetID), ['output_col_0']] = df1_grouped_df_sorted.loc[df1_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_20'] = ''
# bigger_tweet_dataframe['output_col_20'] = bigger_tweet_dataframe['output_col_20'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_20.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_20']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_40'] = ''
# bigger_tweet_dataframe['output_col_40'] = bigger_tweet_dataframe['output_col_40'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])

# df3=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_40.csv",sep =',', keep_default_na=False)
# df3_grouped_df= (df3.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df3_grouped_df['tweetID']=df3_grouped_df['tweetID'].astype(int)
# df3_grouped_df_sorted=(df3_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df3_grouped_df_sorted.tweetID), ['output_col_40']] = df3_grouped_df_sorted.loc[df3_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_60'] = ''
# bigger_tweet_dataframe['output_col_60'] = bigger_tweet_dataframe['output_col_60'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])

# df4=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_60.csv",sep =',', keep_default_na=False)
# df4_grouped_df= (df4.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df4_grouped_df['tweetID']=df4_grouped_df['tweetID'].astype(int)
# df4_grouped_df_sorted=(df4_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df4_grouped_df_sorted.tweetID), ['output_col_60']] = df4_grouped_df_sorted.loc[df4_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_80'] = ''
# bigger_tweet_dataframe['output_col_80'] = bigger_tweet_dataframe['output_col_80'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_80'])

# df5=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_80.csv",sep =',', keep_default_na=False)
# df5_grouped_df= (df5.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df5_grouped_df['tweetID']=df5_grouped_df['tweetID'].astype(int)
# df5_grouped_df_sorted=(df5_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df5_grouped_df_sorted.tweetID), ['output_col_80']] = df5_grouped_df_sorted.loc[df5_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print('')
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_80'])


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_100'] = ''
# bigger_tweet_dataframe['output_col_100'] = bigger_tweet_dataframe['output_col_100'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_80'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_100'])

# df6=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_100.csv",sep =',', keep_default_na=False)
# df6_grouped_df= (df6.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df6_grouped_df['tweetID']=df6_grouped_df['tweetID'].astype(int)
# df6_grouped_df_sorted=(df6_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df6_grouped_df_sorted.tweetID), ['output_col_100']] = df6_grouped_df_sorted.loc[df6_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print('')
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_100'])


bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
bigger_tweet_dataframe['output_col_110'] = ''
bigger_tweet_dataframe['output_col_110'] = bigger_tweet_dataframe['output_col_110'].apply(list)
print(list(bigger_tweet_dataframe.columns.values))

print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_80'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_100'])
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_110'])

df7=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_110.csv",sep =',', keep_default_na=False)
df7_grouped_df= (df7.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
df7_grouped_df['tweetID']=df7_grouped_df['tweetID'].astype(int)
df7_grouped_df_sorted=(df7_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
print('')
bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df7_grouped_df_sorted.tweetID), ['output_col_110']] = df7_grouped_df_sorted.loc[df7_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_110'])


# for index,row in bigger_tweet_dataframe.iterrows():
#     tweetID=index
#     print(index)
#     row['output_col_0']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df1[(df1["tweetID"]==tweetID)].iterrows()]
#     row['output_col_20']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df2[(df2["tweetID"]==tweetID)].iterrows()]
#     row['output_col_40']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df3[(df3["tweetID"]==tweetID)].iterrows()]
#     row['output_col_60']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df4[(df4["tweetID"]==tweetID)].iterrows()]
#     row['output_col_80']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df5[(df5["tweetID"]==tweetID)].iterrows()]
#     row['output_col_100']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df6[(df6["tweetID"]==tweetID)].iterrows()]
#     row['output_col_110']=[ast.literal_eval(row1['only_good_candidates']) for index,row1 in  df7[(df7["tweetID"]==tweetID)].iterrows()]

    # output_reintroduction_theshold_list=[]

    # for elem in range(len(lst)):

    #     threshold=lst[elem]


    #     multipass_output_list=ast.literal_eval(str(row['output_col_'+str(threshold)]))
    #     multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
    #     multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

    #     output_reintroduction_theshold_list.append(multipass_output_list_flat)

    # if not all(collections.Counter(x) == collections.Counter(output_reintroduction_theshold_list[0]) for x in output_reintroduction_theshold_list):
    # 	for output_list in output_reintroduction_theshold_list:
    # 		print(output_list)

    # 	print('================================================')
print(list(bigger_tweet_dataframe.columns.values))
bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
os.remove("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
os.rename("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# ['jose','chris cornell','potus','morgan','nationalism','religion world tour','rust belt','trumps','spicer']