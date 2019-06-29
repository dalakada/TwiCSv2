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
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from itertools import groupby
from operator import itemgetter
import collections 
import re


# #---------------------Existing Lists--------------------
# cachedStopWords = stopwords.words("english")
# tempList=["i","and","or","other","another","across","unlike","anytime","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always","although"]
# for item in tempList:
#     if item not in cachedStopWords:
#         cachedStopWords.append(item)
# cachedStopWords.remove("don")
# cachedStopWords.remove("your")
# cachedStopWords.remove("up")
# cachedTitles = ["mr.","mr","mrs.","mrs","miss","ms","sen.","dr","dr.","prof.","president","congressman"]
# prep_list=["in","at","of","on","v."] #includes common conjunction as well
# article_list=["a","an","the"]
# conjoiner=["de"]
# day_list=["sunday","monday","tuesday","wednesday","thursday","friday","saturday","mon","tues","wed","thurs","fri","sat","sun"]
# month_list=["january","february","march","april","may","june","july","august","september","october","november","december","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
# chat_word_list=["nope","gee","hmm","bye","please","retweet","2mrw","2moro","4get","ooh","reppin","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lmfao","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fucking","fuck","f*ck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]

# #string.punctuation.extend('“','’','”')
# #---------------------Existing Lists--------------------
# lst=['I', 'am', 'sick.']
# lst=['We', 'love', 'those^_^', 'Magyar', '::models', '-']

# def capCheck(word):
#     # print(word)
#     # combined_list=[]+cachedStopWords+prep_list+chat_word_list+article_list+conjoiner
#     # p_num=re.compile(r'^[\W]*[0-9]')
#     p_punct=re.compile(r'[\W]+')
    
#     # if word.startswith('@'):
#     #     return False
#     # if word.startswith('#'):
#     #     return False
#     # elif "<Hashtag" in word:
#     #     return False
#     # # elif not (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)).lower():
#     # #     return True
#     # elif (((word.strip('“‘’”')).lstrip(string.punctuation)).rstrip(string.punctuation)) in combined_list:
#     #     # if((word=="The")|(word=="THE")):
#     #     #     return True
#     #     # else:
#     #     return True
#     # elif p_num.match(word):
#     #     return True
#     # else:
#     #     p=re.compile(r'^[\W]*[A-Z]')
#     #     l= p.match(word)
#     #     if l:
#     #         print('==>',word)
#     #         return True
#     #     else:
#     #         l2= p_punct.match(word)
#     #         if l2:
#     #             print(l2,word)
#     #             return True
#     #         else:
#     #             # print(word)
#     #             return False
#     # print(len(p_punct.fullmatch(word)))
#     if not (p_punct.fullmatch(word) is None):

#         return True
#     else:
#         return False

# tweetWordList_cappos = list(map(lambda element : (element[0],element[1]), filter(lambda element : capCheck(element[1]), enumerate(lst))))
# print(tweetWordList_cappos)
#------------------------------------------------ for my PC ------------------------------------------------------

# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/tweets_3k_annotated_output_backup.csv",sep =',', keep_default_na=False)
# print(len(tweets_unpartitoned))
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/tweets_3k_annotated_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/venezuela_output.csv",sep =',', keep_default_na=False)


################---------TWICS INPUTS
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/roevwade.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billdeblasio.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/pikapika.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/ripcity.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billnye.csv",sep =',', keep_default_na=False)

# tweets_unpartitoned=pd.read_csv("deduplicated_test.csv",sep =';')


# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/pikapika_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/ripcity_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billnye_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/roevwade_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/billdeblasio_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/pikapika_output2.csv",sep =',', keep_default_na=False)

################---------RITTER OUTPUTS
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-venezuela-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter_tweets_3k_annotated_output.csv",sep =',', keep_default_na=False)

# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-pikapika-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-ripcity-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-billnye-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-roevwade-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/ritter-billdeblasio-output.csv",sep =',', keep_default_na=False)

################---------OpenCalai OUTPUTS

# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/opencalai_versions/pikapika_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/opencalai_versions/ripcity_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/opencalai_versions/billnye_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/opencalai_versions/roevwade_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/opencalai_versions/billdeblasio_output.csv",sep =',', keep_default_na=False)

################---------STANFORD OUTPUTS
# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_roevwade_mentions.txt","r")
# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_pikapika_mentions.txt","r")
# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_billdeblasio_mentions.txt","r")
# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_billnye_mentions.txt","r")
# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_ripcity_mentions.txt","r")


#------------------------------------------------ for my Mac ------------------------------------------------------


################---------TWICS OUTPUTS
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/pikapika_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/ripcity_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/billnye_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/roevwade_output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/billdeblasio_output.csv",sep =',', keep_default_na=False)

################---------RITTER OUTPUTS
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/my-baseline-setup/ritter-pikapika-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/my-baseline-setup/ritter-ripcity-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/my-baseline-setup/ritter-billnye-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/my-baseline-setup/ritter-roevwade-output.csv",sep =',', keep_default_na=False)
# tweets_unpartitoned=pd.read_csv("/Users/satadisha/Documents/GitHub/my-baseline-setup/ritter-billdeblasio-output.csv",sep =',', keep_default_na=False)

# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_tweets_3k_annotated_mentions.txt","r")

# fp= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/stanford_venezuela_mentions.txt","r")


# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/roevwade.csv",sep =',')
# print(list(tweets_unpartitoned.columns.values))


# tweet_list=[]
# f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/tweets3K.txt","w")

# f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/venezuela.txt","w")

# f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/roevwade.txt","w")

# f= open("/Users/satadisha/Documents/GitHub/pikapika.txt","w")
# f= open("/Users/satadisha/Documents/GitHub/ripcity.txt","w")
# f= open("/Users/satadisha/Documents/GitHub/billnye.txt","w")
# f= open("/Users/satadisha/Documents/GitHub/roevwade.txt","w")
# f= open("/Users/satadisha/Documents/GitHub/billdeblasio.txt","w")

# f= open("/home/satadisha/Desktop/stanford-ner-2016-10-31/deduplicated_test.txt","w")


# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter-corpus-tools-master/twitter-tools-core/20110208.csv",sep =',')
# tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter-corpus-tools-master/twitter-tools-core/20110206.csv",sep =',')

# index_range=range(0,41)

# for dir_index in index_range:
#     directory='/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/1M_'+str(dir_index)+'_input'
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     sub_dir=directory+'/deploy'
#     if not os.path.exists(sub_dir):
#         os.makedirs(sub_dir)

tweets_unpartitoned=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/tweets_1million_for_others.csv",sep =',')

total_count=0
max_length=0
fc=0
# f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/20110208_"+str(fc)+".txt","w")
# f= open("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/nist_inputs/20110206_"+str(fc)+".txt","w")
dir_count=0
dir_text_length=0
directory='/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/1M_'+str(dir_count)+'_input/deploy/'
# f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/tweets_1million_for_others_"+str(fc)+".txt","w")
f=open(directory+"tweets_1million_for_others_"+str(fc)+".txt","w")

file_sum=0

for index, row in tweets_unpartitoned.iterrows():
    tweet_to_include= str(row['TweetText'])+' --eosc\n'
    if(max_length<5000):
        f.write(tweet_to_include)
        max_length+=1
        dir_text_length+=1
    else:
        file_sum+=max_length
        f.close()
        fc+=1
        # f= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/20110208_"+str(fc)+".txt","w")
        # f= open("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/nist_inputs/20110206_"+str(fc)+".txt","w")
        if(dir_text_length<25000):
            f=open(directory+"tweets_1million_for_others_"+str(fc)+".txt","w")
            dir_text_length+=1
        else:
            dir_count+=1
            directory='/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/data/1M_'+str(dir_count)+'_input/deploy/'
            f=open(directory+"tweets_1million_for_others_"+str(fc)+".txt","w")
            dir_text_length=1
        f.write(tweet_to_include)
        max_length=1
    total_count+=1
if((len(tweets_unpartitoned)%5000)!=0):
    file_sum+=max_length
    f.close()

print(str(total_count),len(tweets_unpartitoned),str(file_sum),str(dir_count))


#------------------------------commenting everything from here

# #------------------------------commenting from here
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/tweets_3K_input_2019-04-26_16-49-32-20455/mentions_output.txt","r")
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/venezuela_input_2019-05-10_12-33-16-15380/mentions_output.txt","r")

# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/roevwade_input_2019-05-28_13-16-07-868289/mentions_output.txt","r")
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/billdeblasio_input_2019-05-28_13-50-22-975417/mentions_output.txt","r")
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/pikapika_input_2019-05-28_13-42-00-707381/mentions_output.txt","r")
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/ripcity_input_2019-05-28_13-38-57-902131/mentions_output.txt","r")
# fp= open("/home/satadisha/Desktop/GitProjects/NeuroNER-master/neuroner/output/billnye_input_2019-05-28_13-44-47-648703/mentions_output.txt","r")



# mentions_list = fp.read().split("\n") # Create a list containing all lines
# fp.close() # Close file

# # print(len(mentions_list),len(tweets_unpartitoned))

# output_index=0

# all_annotations=[]
# all_outputs=[]

# tp_counter_outer=0
# fn_counter_outer=0
# fp_counter_outer=0

# #------------------------------commenting to here

# tp_counter_outer_twics=0
# fn_counter_outer_twics=0
# fp_counter_outer_twics=0

# for index, row in tweets_unpartitoned.iterrows():

#     # print(index)

#     all_postitive_reintroduction_counter_inner=0
#     tp_counter_inner=0
#     fn_counter_inner=0
#     fp_counter_inner=0
#     unrecovered_annotated_mention_list=[]

#     all_postitive_reintroduction_counter_inner_twics=0
#     tp_counter_inner_twics=0
#     fn_counter_inner_twics=0
#     fp_counter_inner_twics=0
#     unrecovered_annotated_mention_list_twics=[]

#     annotated_mention_list=[]
#     tweet_level_candidate_list=str(row['mentions_other']).split(';')
#     for tweet_level_candidates in tweet_level_candidate_list:
#         sentence_level_cand_list= tweet_level_candidates.split(',')
#         annotated_mention_list.extend(sentence_level_cand_list)
#     annotated_mention_list=list(map(lambda element: element.lower().strip(),annotated_mention_list))
#     annotated_mention_list=list(filter(lambda element: (element !=''), annotated_mention_list))
#     annotated_mention_list_for_twiCS= copy.deepcopy(annotated_mention_list)


#     # ritter_candidate_list=str(row['Output']).split(',')
#     # ritter_mention_list=list(map(lambda element: element.lower().strip(),ritter_candidate_list))
#     # ritter_mention_list=list(filter(lambda element: (element !=''), ritter_mention_list))
#     # output_mentions_list_twics_flat= copy.deepcopy(ritter_mention_list)

#     calai_candidate_list=str(row['calai_candidates']).split(',')
#     calai_candidate_list=list(map(lambda element: element.lower().strip(),calai_candidate_list))
#     calai_candidate_list=list(filter(lambda element: (element !=''), calai_candidate_list))
#     output_mentions_list_twics_flat= copy.deepcopy(calai_candidate_list)

#     # all_annotations.extend(annotated_mention_list)
#     # all_outputs.extend(output_mentions_list_twics_flat)

#     # output_mentions_list_twics=ast.literal_eval(row['output_mentions'])
#     # # print(output_mentions_list_twics)
#     # # output_mentions_list_twics=[eval(list_str) for list_str in output_mentions_list_twics]
#     # output_mentions_list_twics_flat = [item.lower() for sublist in output_mentions_list_twics for item in sublist]
#     # output_mentions_list_twics_flat=list(filter(lambda element: element !='', output_mentions_list_twics_flat))

#     all_postitive_counter_inner_twics=len(output_mentions_list_twics_flat)

# #------------------------------commenting from here

    # output_mentions_list= mentions_list[output_index].split(',')
    # output_mentions_list=list(map(lambda element: element.lower().strip(),output_mentions_list))
    # output_mentions_list=list(filter(lambda element: element !='', output_mentions_list))

    # all_postitive_counter_inner=len(output_mentions_list)

    # # print(index, annotated_mention_list, output_mentions_list)
    # all_annotations.extend(annotated_mention_list)
    # all_outputs.extend(output_mentions_list)

    # while(annotated_mention_list):
    #     if(len(output_mentions_list)):
    #         annotated_candidate= annotated_mention_list.pop()
    #         if(annotated_candidate in output_mentions_list):
    #             output_mentions_list.pop(output_mentions_list.index(annotated_candidate))
    #             tp_counter_inner+=1
    #         else:
    #             unrecovered_annotated_mention_list.append(annotated_candidate)
    #     else:
    #         unrecovered_annotated_mention_list.extend(annotated_mention_list)
    #         break

    # print(unrecovered_annotated_mention_list)
    # print('--------------------')

    # unrecovered_annotated_mention_list_outer.extend(unrecovered_annotated_mention_list)
    # fn_counter_inner=len(unrecovered_annotated_mention_list)
    # fp_counter_inner=all_postitive_counter_inner- tp_counter_inner

    # tp_counter_outer+=tp_counter_inner
    # fn_counter_outer+=fn_counter_inner
    # fp_counter_outer+=fp_counter_inner

# #------------------------------commenting to here


# #------------------------------commenting from here
#     print(index, annotated_mention_list_for_twiCS, output_mentions_list_twics_flat)

    # all_annotations.extend(annotated_mention_list)
    # all_outputs.extend(output_mentions_list_twics_flat)

    # while(annotated_mention_list_for_twiCS):
    #     if(len(output_mentions_list_twics_flat)):
    #         annotated_candidate= annotated_mention_list_for_twiCS.pop()
    #         if(annotated_candidate in output_mentions_list_twics_flat):
    #             output_mentions_list_twics_flat.pop(output_mentions_list_twics_flat.index(annotated_candidate))
    #             tp_counter_inner_twics+=1
    #         else:
    #             unrecovered_annotated_mention_list_twics.append(annotated_candidate)
    #     else:
    #         unrecovered_annotated_mention_list_twics.extend(annotated_mention_list_for_twiCS)
    #         break

    # print(unrecovered_annotated_mention_list_twics)
    # print('===========================')
    # unrecovered_annotated_mention_list_outer.extend(unrecovered_annotated_mention_list)

    # fn_counter_inner_twics=len(unrecovered_annotated_mention_list_twics)
    # fp_counter_inner_twics=all_postitive_counter_inner_twics- tp_counter_inner_twics

    # tp_counter_outer+=tp_counter_inner_twics
    # fn_counter_outer+=fn_counter_inner_twics
    # fp_counter_outer+=fp_counter_inner_twics

    # tp_counter_outer_twics+=tp_counter_inner_twics
    # fn_counter_outer_twics+=fn_counter_inner_twics
    # fp_counter_outer_twics+=fp_counter_inner_twics
# #------------------------------commenting to here


    # output_index+=1


# #------------------------------commenting from here
# print('tp_counter_outer: ',tp_counter_outer)
# print('fn_counter_outer: ',fn_counter_outer)
# print('fp_counter_outer: ',fp_counter_outer)

# all_annotations=set(all_annotations)
# all_outputs=set(all_outputs)

# tp_counter_outer= len(all_annotations.intersection(all_outputs))
# fp_counter_outer=len(all_outputs-all_annotations)
# fn_counter_outer=len(all_annotations-all_outputs)
# total_mentions=len(all_outputs)
# total_annotation=len(all_annotations)

# print(tp_counter_outer,fp_counter_outer,fn_counter_outer,total_mentions,total_annotation)

# neuroner_precision= tp_counter_outer/(tp_counter_outer+fp_counter_outer)
# neuroner_recall= tp_counter_outer/(tp_counter_outer+fn_counter_outer)

# print('neuroner_precision: ', neuroner_precision)
# print('neuroner_recall: ', neuroner_recall)

# neuroner_f1 = (2*neuroner_precision*neuroner_recall)/(neuroner_precision+neuroner_recall)
# print('neuroner_f1: ',neuroner_f1)

# stanford_precision= tp_counter_outer/(tp_counter_outer+fp_counter_outer)
# stanford_recall= tp_counter_outer/(tp_counter_outer+fn_counter_outer)

# print('stanford_precision: ', stanford_precision)
# print('stanford_recall: ', stanford_recall)

# stanford_f1 = (2*stanford_precision*stanford_recall)/(stanford_precision+stanford_recall)
# print('stanford_f1: ',stanford_f1)

# opencalai_precision= tp_counter_outer/(tp_counter_outer+fp_counter_outer)
# opencalai_recall= tp_counter_outer/(tp_counter_outer+fn_counter_outer)

# print('opencalai_precision: ', opencalai_precision)
# print('opencalai_recall: ', opencalai_recall)

# opencalai_f1 = (2*opencalai_precision*opencalai_recall)/(opencalai_precision+opencalai_recall)
# print('opencalai_f1: ',opencalai_f1)
# #------------------------------commenting to here

# #------------------------------commenting from here
# print('tp_counter_outer_twics: ',tp_counter_outer_twics)
# print('fn_counter_outer_twics: ',fn_counter_outer_twics)
# print('fp_counter_outer_twics: ',fp_counter_outer_twics)

# # ritter_precision= tp_counter_outer_twics/(tp_counter_outer_twics+fp_counter_outer_twics)
# # ritter_recall= tp_counter_outer_twics/(tp_counter_outer_twics+fn_counter_outer_twics)

# ritter_precision= tp_counter_outer/(tp_counter_outer+fp_counter_outer)
# ritter_recall= tp_counter_outer/(tp_counter_outer+fn_counter_outer)

# print('ritter_precision: ', ritter_precision)
# print('ritter_recall: ', ritter_recall)

# twics_precision= tp_counter_outer_twics/(tp_counter_outer_twics+fp_counter_outer_twics)
# twics_recall= tp_counter_outer_twics/(tp_counter_outer_twics+fn_counter_outer_twics)

# twics_precision= tp_counter_outer/(tp_counter_outer+fp_counter_outer)
# twics_recall= tp_counter_outer/(tp_counter_outer+fn_counter_outer)

# print('twics_precision: ', twics_precision)
# print('twics_recall: ', twics_recall)

# ritter_f1 = (2*ritter_precision*ritter_recall)/(ritter_precision+ritter_recall)
# print('ritter_f1: ',ritter_f1)

# twics_f1 = (2*twics_precision*twics_recall)/(twics_precision+twics_recall)
# print('twics_f1: ',twics_f1)
# #------------------------------commenting to here

#------------------------------commenting everything to here

#------------------------------------------------------------------------------------eviction files------------------------------------------------------------------

# output_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# print(list(output_df.columns.values))
# print(len(output_df))

# output_df['output_col_0'] = ''
# output_df['output_col_0'] = output_df['output_col_0'].apply(list)
# print(output_df[output_df.index==2114]['output_col_0'])

# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_eviction_0.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print(len(df2_grouped_df_sorted))
# output_df.loc[output_df.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_0']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(output_df.index),['only_good_candidates']].values
# print(output_df[output_df.index==2114]['output_col_0'])

# output_df.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")



# output_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# print(list(output_df.columns.values))
# print(len(output_df))

# output_df['output_col_10'] = ''
# output_df['output_col_10'] = output_df['output_col_10'].apply(list)
# print(output_df[output_df.index==2114]['output_col_0'])
# print(output_df[output_df.index==2114]['output_col_10'])

# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_eviction_10.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print(len(df2_grouped_df_sorted))
# output_df.loc[output_df.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_10']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(output_df.index),['only_good_candidates']].values
# print(output_df[output_df.index==2114]['output_col_10'])

# output_df.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")



# output_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# print(list(output_df.columns.values))
# print(len(output_df))

# output_df['output_col_20'] = ''
# output_df['output_col_20'] = output_df['output_col_20'].apply(list)

# print(output_df[output_df.index==2114]['output_col_0'])
# print(output_df[output_df.index==2114]['output_col_10'])
# print(output_df[output_df.index==2114]['output_col_20'])

# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_eviction_20.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print(len(df2_grouped_df_sorted))
# output_df.loc[output_df.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_20']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(output_df.index),['only_good_candidates']].values
# print(output_df[output_df.index==2114]['output_col_20'])

# output_df.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")




# output_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# print(list(output_df.columns.values))
# print(len(output_df))

# output_df['output_col_30'] = ''
# output_df['output_col_30'] = output_df['output_col_30'].apply(list)

# print(output_df[output_df.index==2114]['output_col_0'])
# print(output_df[output_df.index==2114]['output_col_10'])
# print(output_df[output_df.index==2114]['output_col_20'])
# print(output_df[output_df.index==2114]['output_col_30'])

# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_eviction_30.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print(len(df2_grouped_df_sorted))
# output_df.loc[output_df.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_30']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(output_df.index),['only_good_candidates']].values
# print(output_df[output_df.index==2114]['output_col_30'])

# output_df.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")




# output_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# print(list(output_df.columns.values))
# print(len(output_df))

# output_df['output_col_40'] = ''
# output_df['output_col_40'] = output_df['output_col_40'].apply(list)

# print(output_df[output_df.index==2114]['output_col_0'])
# print(output_df[output_df.index==2114]['output_col_10'])
# print(output_df[output_df.index==2114]['output_col_20'])
# print(output_df[output_df.index==2114]['output_col_30'])
# print(output_df[output_df.index==2114]['output_col_40'])

# df2=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_eviction_40.csv",sep =',', keep_default_na=False)
# df2_grouped_df= (df2.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df2_grouped_df['tweetID']=df2_grouped_df['tweetID'].astype(int)
# df2_grouped_df_sorted=(df2_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print(len(df2_grouped_df_sorted))
# output_df.loc[output_df.index.isin(df2_grouped_df_sorted.tweetID), ['output_col_40']] = df2_grouped_df_sorted.loc[df2_grouped_df_sorted.tweetID.isin(output_df.index),['only_good_candidates']].values
# print(output_df[output_df.index==2114]['output_col_40'])

# output_df.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")



#------------------------------------------------------------------------------------reintroduction files------------------------------------------------------------------
# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# print(list(bigger_tweet_dataframe.columns.values))
# df1=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_0.csv",sep =',', keep_default_na=False)
# lst=[0,20,40,60,80,100,110]

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])

# for elem in lst:
#   bigger_tweet_dataframe['output_col_'+str(elem)] = ''
#   bigger_tweet_dataframe['output_col_'+str(elem)] = bigger_tweet_dataframe['output_col_'+str(elem)].apply(list)
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


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# bigger_tweet_dataframe['output_col_110'] = ''
# bigger_tweet_dataframe['output_col_110'] = bigger_tweet_dataframe['output_col_110'].apply(list)
# print(list(bigger_tweet_dataframe.columns.values))

# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_0'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_20'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_40'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_60'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_80'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_100'])
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_110'])

# df7=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_110.csv",sep =',', keep_default_na=False)
# df7_grouped_df= (df7.groupby('tweetID', as_index=False).aggregate(lambda x: x.tolist()))
# df7_grouped_df['tweetID']=df7_grouped_df['tweetID'].astype(int)
# df7_grouped_df_sorted=(df7_grouped_df.sort_values(by='tweetID', ascending=True)).reset_index(drop=True)
# print('')
# bigger_tweet_dataframe.loc[bigger_tweet_dataframe.index.isin(df7_grouped_df_sorted.tweetID), ['output_col_110']] = df7_grouped_df_sorted.loc[df7_grouped_df_sorted.tweetID.isin(bigger_tweet_dataframe.index),['only_good_candidates']].values
# print(bigger_tweet_dataframe[bigger_tweet_dataframe.index==2114]['output_col_110'])



#---------------------------------------------------------------------tallying reintroduction outputs among thresholds
# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# lst=[0,20,40,60,80,100,110]
# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# lst=[0,10,20,30,40]

# elem_not_in_no_reintroduction=set()

# for index,row in bigger_tweet_dataframe.iterrows():
#     tweetID=index
#     # print(index)

#     output_reintroduction_theshold_list=[]

#     for elem in range(len(lst)):

#         threshold=lst[elem]

#         multipass_output_list=ast.literal_eval(row['output_col_'+str(threshold)])
#         multipass_output_list=[eval(list_str) for list_str in multipass_output_list]
#         # print(multipass_output_list)
#         multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
#         multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

#         # multipass_output_list=ast.literal_eval(str(row['output_col_'+str(threshold)]))
#         # multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
#         # multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

#         output_reintroduction_theshold_list.append(multipass_output_list_flat)

#     if not all(collections.Counter(x) == collections.Counter(output_reintroduction_theshold_list[0]) for x in output_reintroduction_theshold_list):
#       print(index,output_reintroduction_theshold_list[0])
#       for output_list in output_reintroduction_theshold_list[1:]:
#           print(output_list)
#           difference=set(output_reintroduction_theshold_list[0])-set(output_list)
#           # print(difference)
#           elem_not_in_no_reintroduction|=difference
#           # print('==>',elem_not_in_no_reintroduction)

#       print('================================================')

# print(elem_not_in_no_reintroduction, len(elem_not_in_no_reintroduction))



# print(list(bigger_tweet_dataframe.columns.values))
# bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# ['jose','chris cornell','potus','morgan','nationalism','religion world tour','rust belt','trumps','spicer']


# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)
# print(list(bigger_tweet_dataframe.columns.values))
# bigger_tweet_dataframe['Output'] = bigger_tweet_dataframe['TweetText']
# print(list(bigger_tweet_dataframe.columns.values))

# bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")



# ---------------------------------------------------------annotating and getting Ritter recall

# cachedStopWords = stopwords.words("english")
# tempList=["i","and","or","other","another","across","unlike","anytime","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always"]
# for item in tempList:
#     if item not in cachedStopWords:
#         cachedStopWords.append(item)
# cachedStopWords.remove("don")
# cachedStopWords.remove("your")
# cachedTitles = ["mr.","mr","mrs.","mrs","miss","ms","sen.","dr","dr.","prof.","president","congressman"]
# prep_list=["in","at","of","on","&;"] #includes common conjunction as well
# article_list=["a","an","the"]
# day_list=["sunday","monday","tuesday","wednesday","thursday","friday","saturday","mon","tues","wed","thurs","fri","sat","sun"]
# month_list=["january","february","march","april","may","june","july","august","september","october","november","december","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
# chat_word_list=["nope","gee","hmm","please","4get","ooh","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fuck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]
# string.punctuation=string.punctuation+'…‘’'
# combined_list_here=([]+list(cachedStopWords)+chat_word_list+day_list+month_list+article_list+prep_list)
# combined_list_filtered=list(filter(lambda word: word not in (prep_list+article_list+month_list), combined_list_here))

# def flatten(mylist, outlist,ignore_types=(str, bytes, int)):

#     if mylist !=[]:
#         for item in mylist:
#             #print not isinstance(item, ne.NE_candidate)
#             if isinstance(item, list) and not isinstance(item, ignore_types):
#                 flatten(item, outlist)
#             else:
#                 if type(item)!= int:
#                     item=item.strip(' \t\n\r')
#                 outlist.append(item)
#     return outlist


# def normalize(word):
#     strip_op=word
#     strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip()).lower()
#     strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
#     #strip_op= self.rreplace(self.rreplace(self.rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
#     if strip_op.endswith("'s"):
#         li = strip_op.rsplit("'s", 1)
#         return ''.join(li)
#     elif strip_op.endswith("’s"):
#         li = strip_op.rsplit("’s", 1)
#         return ''.join(li)
#     else:
#         return strip_op


# def getWords(sentence):
#     tempList=[]
#     tempWordList=sentence.split()
#     #print(tempWordList)
#     for word in tempWordList:
#         temp=[]
        
#         if "(" in word:
#             temp=list(filter(lambda elem: elem!='',word.split("(")))
#             if(temp):
#                 temp=list(map(lambda elem: '('+elem, temp))
#         elif ")" in word:
#             temp=list(filter(lambda elem: elem!='',word.split(")")))
#             if(temp):
#                 temp=list(map(lambda elem: elem+')', temp))
#             # temp.append(temp1[-1])
#         elif (("-" in word)&(not word.endswith("-"))):
#             temp1=list(filter(lambda elem: elem!='',word.split("-")))
#             if(temp1):
#                 temp=list(map(lambda elem: elem+'-', temp1[:-1]))
#             temp.append(temp1[-1])
#         elif (("?" in word)&(not word.endswith("?"))):
#             temp1=list(filter(lambda elem: elem!='',word.split("?")))
#             if(temp1):
#                 temp=list(map(lambda elem: elem+'?', temp1[:-1]))
#             temp.append(temp1[-1])
#         elif ((":" in word)&(not word.endswith(":"))):
#             temp1=list(filter(lambda elem: elem!='',word.split(":")))
#             if(temp1):
#                 temp=list(map(lambda elem: elem+':', temp1[:-1]))
#             temp.append(temp1[-1])
#         elif (("," in word)&(not word.endswith(","))):
#             #temp=list(filter(lambda elem: elem!='',word.split(",")))
#             temp1=list(filter(lambda elem: elem!='',word.split(",")))
#             if(temp1):
#                 temp=list(map(lambda elem: elem+',', temp1[:-1]))
#             temp.append(temp1[-1])
#         elif (("/" in word)&(not word.endswith("/"))):
#             temp1=list(filter(lambda elem: elem!='',word.split("/")))
#             if(temp1):
#                 temp=list(map(lambda elem: elem+'/', temp1[:-1]))
#             temp.append(temp1[-1])
#             #print(index, temp)
#         elif "..." in word:
#             #print("here")
#             temp=list(filter(lambda elem: elem!='',word.split("...")))
#             if(temp):
#                 if(word.endswith("...")):
#                     temp=list(map(lambda elem: elem+'...', temp))
#                 else:
#                    temp=list(map(lambda elem: elem+'...', temp[:-1]))+[temp[-1]]
#             # temp.append(temp1[-1])
#         elif ".." in word:
#             temp=list(filter(lambda elem: elem!='',word.split("..")))
#             if(temp):
#                 if(word.endswith("..")):
#                     temp=list(map(lambda elem: elem+'..', temp))
#                 else:
#                     temp=list(map(lambda elem: elem+'..', temp[:-1]))+[temp[-1]]
#             #temp.append(temp1[-1])
#         elif "…" in word:
#             temp=list(filter(lambda elem: elem!='',word.split("…")))
#             if(temp):
#                 if(word.endswith("…")):
#                     temp=list(map(lambda elem: elem+'…', temp))
#                 else:
#                     temp=list(map(lambda elem: elem+'…', temp[:-1]))+[temp[-1]]
#         else:
#             #if word not in string.punctuation:
#             temp=[word]
#         if(temp):
#             tempList.append(temp)
#     tweetWordList=flatten(tempList,[])
#     return tweetWordList


# def get_Candidates(sequence, CTrie,flag):
#     #flag: debug_flag

#     CTtieCandidateList=CTrie.displayTrie("",[])
#     # print('candidate list:', len(CTtieCandidateList))

#     candidateList=[]
#     left=0
#     start_node=CTrie
#     last_cand="NAN"
#     last_cand_substr=""
#     reset=False
#     right=0
#     while (right < len(sequence)):
#         # if(flag):
#         #     print(right)
#         if(reset):
#             start_node=CTrie
#             last_cand_substr=""
#             left=right
#         curr_text=sequence[right][0]
#         curr_pos=[sequence[right][1]]
#         #normalized curr_text
#         curr=normalize(sequence[right][0])
#         cand_str=normalize(last_cand_substr+" "+curr)
#         cand_str_wPunct=(last_cand_substr+" "+curr_text).lower()
#         last_cand_sequence=sequence[left:(right+1)]
#         last_cand_text=' '.join(str(e[0]) for e in last_cand_sequence)
#         last_cand_text_norm=normalize(' '.join(str(e[0]) for e in last_cand_sequence))
#         if(flag):
#             print("==>",cand_str,last_cand_text_norm)
#         if((cand_str==last_cand_text_norm)&((curr in start_node.path.keys())|(curr_text.lower() in start_node.path.keys()))):
#         #if (((curr in start_node.path.keys())&(cand_str==last_cand_text_norm))|(curr_text.lower() in start_node.path.keys())):
#             if flag:
#                 print("=>",cand_str,last_cand_text)
#             reset=False
#             if (curr_text.lower() in start_node.path.keys()):
#                 if (start_node.path[curr_text.lower()].value_valid):
#                     last_cand_pos=[e[1] for e in last_cand_sequence]
#                     last_cand_batch=start_node.path[curr_text.lower()].feature_list[-1]
#                     last_cand=last_cand_text
#                 elif(curr in start_node.path.keys()):
#                     if ((start_node.path[curr].value_valid)):
#                         last_cand_pos=[e[1] for e in last_cand_sequence]
#                         last_cand=last_cand_text
#                         last_cand_batch=start_node.path[curr].feature_list[-1]
#                     else:
#                         if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
#                             #print("hehe",cand_str)
#                             right=left
#                             reset=True
#                 else:
#                     if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
#                         #print("hehe",cand_str)
#                         right=left
#                         reset=True
#             elif ((start_node.path[curr].value_valid)&(cand_str==last_cand_text_norm)):
#                 # if flag:
#                 #     print("==",last_cand_text)
#                 last_cand_pos=[e[1] for e in last_cand_sequence]
#                 last_cand=last_cand_text
#                 last_cand_batch=start_node.path[curr].feature_list[-1]
#             else:
#                 if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
#                     #print("hehe",cand_str)
#                     right=left
#                     reset=True
#             if((curr_text.lower() in start_node.path.keys())&(cand_str==last_cand_text_norm)):
#                 start_node=start_node.path[curr_text.lower()]
#                 last_cand_substr=cand_str_wPunct
#             else:
#                 start_node=start_node.path[curr]
#                 last_cand_substr=cand_str
#         else:
#             #print("=>",cand_str,last_cand_text)
#             if(last_cand!="NAN"):
#                 candidateList.append((last_cand,last_cand_pos,last_cand_batch))
#                 last_cand="NAN"
#                 if(start_node!=CTrie):
#                     start_node=CTrie
#                     last_cand_substr=""
#                     if curr in start_node.path.keys():
#                         # if(flag):
#                         #     print("here",curr)
#                         reset=False
#                         if start_node.path[curr].value_valid:
#                             last_cand_text=curr_text
#                             last_cand_pos=curr_pos
#                             last_cand=last_cand_text
#                             last_cand_batch=start_node.path[curr].feature_list[-1]
#                         left=right
#                         start_node=start_node.path[curr]
#                         last_cand_substr=curr
#                     else:
#                         reset=True
#                 else:
#                     reset=True
#             else:
#                 if(left<right):
#                     # if(flag):
#                     #     print(sequence[(left+1):(right+1)])
#                     #candidateList.extend(get_Candidates(sequence[(left+1):(right+1)], CTrie, flag))
#                     right=left
#                     # if(flag):
#                     #     print("++",right)
#                 reset=True
#         right+=1
#     # if(flag):
#     #     print(last_cand)
#     if(last_cand!="NAN"):
#         candidateList.append((last_cand,last_cand_pos,last_cand_batch))

#     # print('==>',candidateList)
#     return candidateList


# all_entity_candidates=[]

# CTrie=trie.Trie("ROOT")

# tweets_unpartitoned=pd.read_csv("deduplicated_test_output_all_runs.csv",sep =',', keep_default_na=False)

# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations_wRitterOutput.csv",sep =',', keep_default_na=False)

# # annotation_df=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)

# # print(list(bigger_tweet_dataframe.columns.values))
# # print(len(bigger_tweet_dataframe),len(annotation_df))

# # tweetID_list1=bigger_tweet_dataframe['Tweet IDs'].tolist()
# # tweetID_list2=annotation_df['Tweet IDs'].tolist()

# # lst=[tweetID for tweetID in tweetID_list2 if tweetID not in tweetID_list1]
# # print(lst)
# # print(len(lst))

# bigger_tweet_dataframe['annotation'] = ''
# bigger_tweet_dataframe['annotation'] = bigger_tweet_dataframe['annotation'].apply(list)


# #----------------------commenting out since annotations already retrieved
# # using a previously annotated file to set the CTrie
# for index,row in tweets_unpartitoned.iterrows():

#     tweet_in_first_five_hundred=str(row['First_five_hundred'])
#     tweetText=str(row['TweetText'])
#     annotated_mention_list=[]

#     if(tweet_in_first_five_hundred!=''):
        
#         tweet_level_candidate_list=str(row['Annotations']).split(';')
#         for tweet_level_candidates in tweet_level_candidate_list:
#             sentence_level_cand_list= tweet_level_candidates.split(',')
#             annotated_mention_list.extend(sentence_level_cand_list)
#         # for index in range(len(ritter_output)):
#         annotated_mention_list=list(map(lambda element: element.lower().strip(),annotated_mention_list))
#         annotated_mention_list=list(filter(lambda element: element !='', annotated_mention_list))

#         for annotation in annotated_mention_list:
#             if annotation not in all_entity_candidates:
#                 all_entity_candidates.append(annotation)
#             CTrie.setitem_forAnnotation(annotation.split())

#     else:
#         break

# additional_candidates1=['ivanka trump', 'president donald trump', 'jared kushner','bill clinton','president trump','anthony weiner','joe biden','anthony','democrat','john terry','jose','chris cornell','potus','morgan','nationalism','religion world tour','rust belt','trumps','spicer','assange']
# additional_candidates2=['poc', 'luther', 'comey', 'hamas', 'sec of state', 'whitehouse', 'kendrick', 'julian assange', 'asian','andrew breitbart', 'hrc', 'allen', 'yahoo sports', 'potus trump', 'indonesia', 'latin', 'huffpost', 'houston', 'labour party', 'legally blonde', 'sports illustrated', 'ox', 'trump', 'tom', 'high hopes', 'ritz', 'deadline', 'peter', 'trump war room', 'labour', 'washington wizards',  'maggie', 'religion world tour', 'justin bieber', 'sharia', 'jurgen', 'jose',  'james comey', 'tristan', 'hatch', 'johnson', 'tupac', 'blues', 'geno', 'trumps', 'pa', 'carat',  'senate intelligence committee', 'patti', 'pos', "o'reilly", 'j.k. rowling', 'saudi king salman', 'diamond', 'rick', 'latinos', 'colluded', 'hajj', 'buzzfeed news', 'chris', 'michael', 'zlatan', 'observer', 'ceo', 'ann', 'scavino', 'xhaka', 'us presidents', 'alderaan',  'sex addiction', "trump's radar", 'tower', 'ainge', 'wp', 'becker', 'cbsnews', 'blotus', 'americas', 'valencia', 'arabia', 'nbc sports', 'earn', 'striker', 'trumpcare', 'mavs', 'pbo', 'blk', 'gas', 'pichichi', 'wayne bridge', 'airman',  'mcdonald', 'senators', 'colorado', 'isiah', 'larry', 'disney', 'cornel west', 'clas', 'ku', 'scott', 'sf', 'whoopi goldberg', 'sarah palin', 'casino', 'dreaming', 'maxine', 'welsh',  'lords', 'nafta', 'us president', 'jeffery lord', 'shapiro', 'sharif', 'first amendment', 'yahoo', 'stream', 'california democrats',  'us news', 'africa', 'ahca', 'isis-lite',  'michael dorf', 'watch lebron', 'kentucky',  'david lee', 'dc', 'drumpf', 'tru', 'kawhi', 'parker', 'hh', 'ali', 'bush',  'elijah', 'kccinews', 'intel', 'reds', 'byron york', 'state dept', 'copa', 'tump', 'brady', 'madrid', 'aclu', 'andrews', 'finn', 'eibar', 'brad', 'ipo', "trump's administration", 'donald trump memes', 'google', 'infowars',  'weekly standard', 'bbcworld', 'softbank', 'knicks',  'okc', 'mike',  'nbcnews', 'butler', 'k.t',  'anthony weiner', 'fcc', 'matrix', 'gorsuch', 'maya angelou', 'tommy', 'torys', 'bridge', 'demorats', 'gulf', 'gsw', 'dennis wise', 'russian agent', 'brooklyn nets', 'dick gregory', 'turkey', 'kayleigh mcenany', 'hall of presidents', 'msn', 'ray', 'jeffrey lord', 'biko', 'romphim', 'kim', 'afrikan perspectives', 'bleacherreport', 'repubs', 'charles barkley', 'smith', 'gooner', 'white house', 'nwo', 'jeffrey', 'capitol', 'maher', 'sunderland', 'garvey', 'malcolm x sought', 'fairfax', 'salutes', 'breeze', 'arsène wenger', 
# 'cardiff', 'cornell west', 'mcclatchy', 'sc', 'nazis', 'nigerians', 'eveningstandard', 'jackson', 'olympic', 'stamford bridge', 'zucker', 'ford', 'lowry', 'williams', 'tillerson', 'archie', 'ailes', 'the rock', 'ebola', 'hallmark', 'spanish', 'mps', 'lagos', 'cavs', 'suarez', 'truthfeednews', 'sau', 'lw', 'muhammad', 'roland', 'islamaphobic', 'santos', 'coutinho', 'canadian', 'spicer', 'saudia',  'scotland', 'bob woodward', 'rice',  'potus',  'hector bellerin', 'mattis', 'conway', 'chai', 'sean', 'wc', 'sturridge', 'paul pierce', 'greens', 'king abdulaziz medal', 'kingston', 'libertarian', 'roger', 'jack', 'commander in chief', 'thomas', 'stamford', 'brown', 'josh', 'trump organization', 'saudi arabia', 
# 'chuck', 'feds', 'nbatv', 'stein',  'meg', 'jesus', 'flynn','melania trump', 'huffpo', 'penguins', 'jewish', 'merkel', 'greece', 'messi', 'neymar', 'ricky',  'irish', 'klan', 'andrew', 'nd', 'portland', 'stu', 'santa', 'intel agencies', 'adam silver',  'president trump',  'monmouth', 'celtics', 'sa',  'los', 'nationalism',  'space jam',  'nbas', 'republicans', 'david', 'scarborough', 'john mccain', 'tiffany trump', 'titanic', 'eric', 'korver', 'nbc news', 'susan rice', 'howard', 'ki', 'pete', 'arabian', 'rompers', 'harvard study', 'macron', 'sportscenter', 'barkley', 'prophet', 'nsa', 'border infrastructure', 'baker', 'jews', '1b', 'rev',  'ga', 'jc', 'duke', 'democrat', "trump's treasury secretary", 'zidane', 'matt', 'ied', 'abbas', 'kellyanne conway', 'amer', 'bellerin', 'ryan', 'danny', 'supreme court', 'cher', 'kyle', 'western', 'doral', 'ww3', 'apple', 'jarrett', 'barry', 'amazon', 'phil', 'alec baldwin', 'rust belt', 'miller', 'white hou', 'kronke', 'mustafi',   'cristiano',  'first lady', 'eu', 'demoncrats', 'treasury', 'mt', 'liberal resistance', 'nra', 'amy', 'e.p.a', 'anderson pooper', 'texas', 'lance',  'elizabeth', 'kelly', 'usaf', 'stan', 'motd', 'indians', 'john terry', 'blazers', 'travel ban', 'punk', 'cameron', 'baba',
# 'george takei', 'jo', 'tommy milone',  'morals', 'prime minister', 'chuck schumer', 'chris cornell', 'sheiks', 'malcolm', 'romanian', 'norway', 'neocon', 'canadians', 'erdogan', 'champs league', 'brooklyn', 'uk', 'sisi', 'i.t',  'greg', 'allah', 'larry bird', 'kante', 'birmingham', 'fp', 'nba', 'san francisco',  'daily caller', 'yuri kochiyama', 'election day', 'robert', 'sankara', 'crowder', 'chucky', 'indian',  'batshuayi', 'newsflash', 'doj', 'ola ray', 'korean', 'us armed forces', 'westbrook', "handmaid's tale",'rick santorum','joe lieberman','paul craig roberts','kim jong','patti lupone','bernie sanders',
# 'bradley', 'john', 'kevin', 'imam', 'enes kanter', 'alex', 'epshteyn', 'truman', 'nbc',  'muslim brotherhood','associated press', 'enes', 'nawaz', 'kenny', 'zimbabwe', 'russia collusion', 'illegal alien', 'ireland', 'kevin love', 'kyle korver', 'farron', 'saudi', 'chicago', 'malaga', 'group of notre dame', 'oscar', 'snapchat', 'netflix', 'steagall',  'bbc', 'reagan', 'obam', 'sanders', 'atlanta',  'jeb', 'darrell', 'cambridge analytica', 'jaylen brown', 'seth',  'chad brown', 'ivanka', 'malcolm x', 'saddam', 'morgan', 'shannon', 'al', 'jaylen', 'wayne', 'paul', 'america', 'brian', 'kurds', 'island', 'berkeley', 'abc', 'barack', 'stephen','fergie', 'mcconnell', 'police state', 'marine one', 'justice department', 'premier league', 'schmidt', 'n korea', 'mandela', 'omar', 'goldman',  'russ', 'mal', 'liam', 'georginio wijnaldum', 'sheryl', 'michelle obama', 'host bob beckel', 'messiah', 'h.r', 'superhero', 'emirate', 'daniel', 'victor moses', 'charles', 'palin', 'pats', 'steph', 'pippa', 'us state', 'leahy', 'washingtonpost', 'pogba', 'la liga', 'warriors', 'darren wilson', 'issa', 'malcol', 'mich', 'harry potter', 'weekend update', 'saudi king', 'middle east', 'kathy', 'simmons', 'flotus','whoopi', 'congressional republicans', 'superman', 'muslim', 'bulgaria', 'karma', 'wwe', 'pirates of the caribbean', 'godspeed', 'memorial day', 'jabbar', 'jason', 'jew', 'don mattingly', 'donald trump', 'world cup', 'gary', 'hitler', 'obama', 'house of cards', 'bil','tom hardy',  'nick', 'luis enrique', 'france', 'maduro']

# additional_candidate_set= set(additional_candidates1)
# additional_candidate_set.update(additional_candidates2)
# additional_candidates=list(additional_candidate_set)
# print(len(additional_candidates))
# for additional_candidate in additional_candidates:
#     # print(additional_candidate)
#     CTrie.setitem_forAnnotation(additional_candidate.split())

# tp_ritter_counter=0
# fp_ritter_counter=0
# fn_ritter_counter=0

# my_tally_arr_ritter=[]

# for index,row in bigger_tweet_dataframe.iterrows():

#     unrecovered_annotated_mention_list=[]
#     all_postitive_ritter_counter_inner=0
#     tp_ritter_counter_inner=0
#     fp_ritter_counter_inner=0
#     fn_ritter_counter_inner=0

#     annotated_mention_list=[]

#     tweetText=str(row['TweetText'])
#     tweetSentences=list(filter (lambda sentence: len(sentence)>1, tweetText.split('\n')))
#     tweetSentenceList_inter=flatten(list(map(lambda sentText: sent_tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
#     tweetSentenceList=list(filter (lambda sentence: len(sentence)>1, tweetSentenceList_inter))
#     # tweetSentenceList=[]
#     CTtieCandidateList=CTrie.displayTrie("",[])
#     # print('candidate list:')
#     # for annotated_candidate in CTtieCandidateList:
#     #     print(annotated_candidate)
    
#     for sentence in tweetSentenceList:
#         tweetWordList=getWords(sentence)
#         tweetWordList= [(token,idx) for idx,token in enumerate(tweetWordList)]
#         tweetWordList_stopWords=list(filter (lambda word: ((((word[0].strip()).strip(string.punctuation)).lower() in combined_list_filtered)|(word[0].strip() in string.punctuation)|(word[0].startswith('@'))), tweetWordList))
#         # phase 2 candidate tuples without stopwords for a sentence
#         c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords]
#         #c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords ]
#         # print(c)
#         sequences=[]
#         for k, g in groupby(enumerate(c), lambda element: element[0]-int(element[1][1])):
#             sequences.append(list(map(itemgetter(1), g)))

#         # print(sequences)
#         for sequence in sequences:
#             seq_candidate_list=get_Candidates(sequence, CTrie,False)
#             # print('+>',sequence,seq_candidate_list)
#             if(seq_candidate_list):
#                 for candidate_tuple in seq_candidate_list:
#                     candidateText=normalize(candidate_tuple[0])
#                     # if(index=='2174'):
#                     #     print('===>',candidateText)
#                     if(candidate_tuple[0].lower()!='us'):
#                         annotated_mention_list.append(candidateText)

#     # print('-----',index)

#     row['annotation']=annotated_mention_list

#     # annotated_mention_list=ast.literal_eval(str(row['annotation']))

#     ritter_output=list(map(lambda element: element.lower().strip(),str(row['Output']).split(',')))
#     ritter_output=list(filter(lambda element: element !='', ritter_output))
#     all_postitive_ritter_counter_inner=len(ritter_output)
#     print('=>',index, tweetText)
#     print(annotated_mention_list, ritter_output)
#     while(annotated_mention_list):
#         if(len(ritter_output)):
#             annotated_candidate= annotated_mention_list.pop()
#             if(annotated_candidate in ritter_output):
#                 ritter_output.pop(ritter_output.index(annotated_candidate))
#                 tp_ritter_counter_inner+=1
#             else:
#                 unrecovered_annotated_mention_list.append(annotated_candidate)
#                 my_tally_arr_ritter.append(1)
#       # print(ritter_output.pop())
#       # print(ritter_output)
#         else:
#             unrecovered_annotated_mention_list.extend(annotated_mention_list)
#             my_tally_arr_ritter.append(len(annotated_mention_list))
#             break

#     print(unrecovered_annotated_mention_list)
#     # unrecovered_annotated_mention_list_outer_ritter.extend(unrecovered_annotated_mention_list)
#     fn_ritter_counter_inner=len(unrecovered_annotated_mention_list)
#     fp_ritter_counter_inner=all_postitive_ritter_counter_inner- tp_ritter_counter_inner

#     tp_ritter_counter+= tp_ritter_counter_inner
#     fp_ritter_counter+=fp_ritter_counter_inner
#     fn_ritter_counter+=fn_ritter_counter_inner

# print('ritter numbers: ',tp_ritter_counter,fp_ritter_counter,fn_ritter_counter)
# ritter_precision= tp_ritter_counter/(tp_ritter_counter+fp_ritter_counter)
# ritter_recall= tp_ritter_counter/(tp_ritter_counter+fn_ritter_counter)
# print(ritter_recall)

# # ritter numbers:  634489 671892 650167
# # 0.49389797735736257

# ritter numbers:  685694 620687 538329
# 0.5601969897624473

# bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/twitter_nlp-master/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
