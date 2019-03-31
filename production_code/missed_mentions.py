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
import os


cachedStopWords = stopwords.words("english")
tempList=["i","and","or","other","another","across","unlike","anytime","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always"]
for item in tempList:
    if item not in cachedStopWords:
        cachedStopWords.append(item)
cachedStopWords.remove("don")
cachedStopWords.remove("your")
cachedTitles = ["mr.","mr","mrs.","mrs","miss","ms","sen.","dr","dr.","prof.","president","congressman"]
prep_list=["in","at","of","on","&;"] #includes common conjunction as well
article_list=["a","an","the"]
day_list=["sunday","monday","tuesday","wednesday","thursday","friday","saturday","mon","tues","wed","thurs","fri","sat","sun"]
month_list=["january","february","march","april","may","june","july","august","september","october","november","december","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
chat_word_list=["nope","gee","hmm","please","4get","ooh","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fuck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]
string.punctuation=string.punctuation+'…‘’'
combined_list_here=([]+list(cachedStopWords)+chat_word_list+day_list+month_list+article_list+prep_list)
combined_list_filtered=list(filter(lambda word: word not in (prep_list+article_list+month_list), combined_list_here))

def flatten(mylist, outlist,ignore_types=(str, bytes, int)):

    if mylist !=[]:
        for item in mylist:
            #print not isinstance(item, ne.NE_candidate)
            if isinstance(item, list) and not isinstance(item, ignore_types):
                flatten(item, outlist)
            else:
                if type(item)!= int:
                    item=item.strip(' \t\n\r')
                outlist.append(item)
    return outlist


def normalize(word):
    strip_op=word
    strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip()).lower()
    strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
    #strip_op= self.rreplace(self.rreplace(self.rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
    if strip_op.endswith("'s"):
        li = strip_op.rsplit("'s", 1)
        return ''.join(li)
    elif strip_op.endswith("’s"):
        li = strip_op.rsplit("’s", 1)
        return ''.join(li)
    else:
        return strip_op


def getWords(sentence):
    tempList=[]
    tempWordList=sentence.split()
    #print(tempWordList)
    for word in tempWordList:
        temp=[]
        
        if "(" in word:
            temp=list(filter(lambda elem: elem!='',word.split("(")))
            if(temp):
                temp=list(map(lambda elem: '('+elem, temp))
        elif ")" in word:
            temp=list(filter(lambda elem: elem!='',word.split(")")))
            if(temp):
                temp=list(map(lambda elem: elem+')', temp))
            # temp.append(temp1[-1])
        elif (("-" in word)&(not word.endswith("-"))):
            temp1=list(filter(lambda elem: elem!='',word.split("-")))
            if(temp1):
                temp=list(map(lambda elem: elem+'-', temp1[:-1]))
            temp.append(temp1[-1])
        elif (("?" in word)&(not word.endswith("?"))):
            temp1=list(filter(lambda elem: elem!='',word.split("?")))
            if(temp1):
                temp=list(map(lambda elem: elem+'?', temp1[:-1]))
            temp.append(temp1[-1])
        elif ((":" in word)&(not word.endswith(":"))):
            temp1=list(filter(lambda elem: elem!='',word.split(":")))
            if(temp1):
                temp=list(map(lambda elem: elem+':', temp1[:-1]))
            temp.append(temp1[-1])
        elif (("," in word)&(not word.endswith(","))):
            #temp=list(filter(lambda elem: elem!='',word.split(",")))
            temp1=list(filter(lambda elem: elem!='',word.split(",")))
            if(temp1):
                temp=list(map(lambda elem: elem+',', temp1[:-1]))
            temp.append(temp1[-1])
        elif (("/" in word)&(not word.endswith("/"))):
            temp1=list(filter(lambda elem: elem!='',word.split("/")))
            if(temp1):
                temp=list(map(lambda elem: elem+'/', temp1[:-1]))
            temp.append(temp1[-1])
            #print(index, temp)
        elif "..." in word:
            #print("here")
            temp=list(filter(lambda elem: elem!='',word.split("...")))
            if(temp):
                if(word.endswith("...")):
                    temp=list(map(lambda elem: elem+'...', temp))
                else:
                   temp=list(map(lambda elem: elem+'...', temp[:-1]))+[temp[-1]]
            # temp.append(temp1[-1])
        elif ".." in word:
            temp=list(filter(lambda elem: elem!='',word.split("..")))
            if(temp):
                if(word.endswith("..")):
                    temp=list(map(lambda elem: elem+'..', temp))
                else:
                    temp=list(map(lambda elem: elem+'..', temp[:-1]))+[temp[-1]]
            #temp.append(temp1[-1])
        elif "…" in word:
            temp=list(filter(lambda elem: elem!='',word.split("…")))
            if(temp):
                if(word.endswith("…")):
                    temp=list(map(lambda elem: elem+'…', temp))
                else:
                    temp=list(map(lambda elem: elem+'…', temp[:-1]))+[temp[-1]]
        else:
            #if word not in string.punctuation:
            temp=[word]
        if(temp):
            tempList.append(temp)
    tweetWordList=flatten(tempList,[])
    return tweetWordList


def get_Candidates(sequence, CTrie,flag):
    #flag: debug_flag

    CTtieCandidateList=CTrie.displayTrie("",[])
    # print('candidate list:', len(CTtieCandidateList))

    candidateList=[]
    left=0
    start_node=CTrie
    last_cand="NAN"
    last_cand_substr=""
    reset=False
    right=0
    while (right < len(sequence)):
        # if(flag):
        #     print(right)
        if(reset):
            start_node=CTrie
            last_cand_substr=""
            left=right
        curr_text=sequence[right][0]
        curr_pos=[sequence[right][1]]
        #normalized curr_text
        curr=normalize(sequence[right][0])
        cand_str=normalize(last_cand_substr+" "+curr)
        cand_str_wPunct=(last_cand_substr+" "+curr_text).lower()
        last_cand_sequence=sequence[left:(right+1)]
        last_cand_text=' '.join(str(e[0]) for e in last_cand_sequence)
        last_cand_text_norm=normalize(' '.join(str(e[0]) for e in last_cand_sequence))
        if(flag):
            print("==>",cand_str,last_cand_text_norm)
        if((cand_str==last_cand_text_norm)&((curr in start_node.path.keys())|(curr_text.lower() in start_node.path.keys()))):
        #if (((curr in start_node.path.keys())&(cand_str==last_cand_text_norm))|(curr_text.lower() in start_node.path.keys())):
            if flag:
                print("=>",cand_str,last_cand_text)
            reset=False
            if (curr_text.lower() in start_node.path.keys()):
                if (start_node.path[curr_text.lower()].value_valid):
                    last_cand_pos=[e[1] for e in last_cand_sequence]
                    last_cand_batch=start_node.path[curr_text.lower()].feature_list[-1]
                    last_cand=last_cand_text
                elif(curr in start_node.path.keys()):
                    if ((start_node.path[curr].value_valid)):
                        last_cand_pos=[e[1] for e in last_cand_sequence]
                        last_cand=last_cand_text
                        last_cand_batch=start_node.path[curr].feature_list[-1]
                    else:
                        if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
                            #print("hehe",cand_str)
                            right=left
                            reset=True
                else:
                    if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
                        #print("hehe",cand_str)
                        right=left
                        reset=True
            elif ((start_node.path[curr].value_valid)&(cand_str==last_cand_text_norm)):
                # if flag:
                #     print("==",last_cand_text)
                last_cand_pos=[e[1] for e in last_cand_sequence]
                last_cand=last_cand_text
                last_cand_batch=start_node.path[curr].feature_list[-1]
            else:
                if((right==(len(sequence)-1))&(last_cand=="NAN")&(left<right)):
                    #print("hehe",cand_str)
                    right=left
                    reset=True
            if((curr_text.lower() in start_node.path.keys())&(cand_str==last_cand_text_norm)):
                start_node=start_node.path[curr_text.lower()]
                last_cand_substr=cand_str_wPunct
            else:
                start_node=start_node.path[curr]
                last_cand_substr=cand_str
        else:
            #print("=>",cand_str,last_cand_text)
            if(last_cand!="NAN"):
                candidateList.append((last_cand,last_cand_pos,last_cand_batch))
                last_cand="NAN"
                if(start_node!=CTrie):
                    start_node=CTrie
                    last_cand_substr=""
                    if curr in start_node.path.keys():
                        # if(flag):
                        #     print("here",curr)
                        reset=False
                        if start_node.path[curr].value_valid:
                            last_cand_text=curr_text
                            last_cand_pos=curr_pos
                            last_cand=last_cand_text
                            last_cand_batch=start_node.path[curr].feature_list[-1]
                        left=right
                        start_node=start_node.path[curr]
                        last_cand_substr=curr
                    else:
                        reset=True
                else:
                    reset=True
            else:
                if(left<right):
                    # if(flag):
                    #     print(sequence[(left+1):(right+1)])
                    #candidateList.extend(get_Candidates(sequence[(left+1):(right+1)], CTrie, flag))
                    right=left
                    # if(flag):
                    #     print("++",right)
                reset=True
        right+=1
    # if(flag):
    #     print(last_cand)
    if(last_cand!="NAN"):
        candidateList.append((last_cand,last_cand_pos,last_cand_batch))

    # print('==>',candidateList)
    return candidateList

# tweets_unpartitoned=pd.read_csv("deduplicated_test_output_all_runs.csv",sep =',', keep_default_na=False)


# unrecovered_annotated_candidate_list_outer=[]
all_entity_candidates=[]

CTrie=trie.Trie("ROOT")


#----------------------------------------------------------single pass to multi-pass experiment

# ritter_list=[]

# our_counter=0
# our_recovered_counter=0

# our_counter_mention=0
# our_recovered_mention=0

# tp_ritter_counter=0
# fp_ritter_counter=0
# fn_ritter_counter=0


# my_tally_arr_ritter=[]

# # lst=[0,1,2,3,4,5,6]
# lst=[0,1,2,3,4,5,6,7,8,9,10,11,12]

# tp_multipass_counter_arr=[0 for i in range(len(lst))]
# fp_multipass_counter_arr=[0 for i in range(len(lst))]
# fn_multipass_counter_arr=[0 for i in range(len(lst))]

# unrecovered_annotated_mention_list_outer_ritter=[]
# unrecovered_annotated_mention_list_outer_multipass = [[] for i in range(len(lst))]

# # input_df= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence', 'phase1Candidates','start_time','entry_batch'))
# for index,row in tweets_unpartitoned.iterrows():
#     # annotated_candidates=str(row['mentions_other'])
#     unrecovered_annotated_mention_list=[]
#     unrecovered_annotated_mention_list_multipass = [[] for i in range(len(lst))]


#     all_postitive_ritter_counter_inner=0
#     tp_ritter_counter_inner=0
#     fp_ritter_counter_inner=0
#     fn_ritter_counter_inner=0

#     # all_postitive_multipass_counter_inner_arr=[0 for i in range(len(lst))]
#     # tp_multipass_counter_inner_arr=[0 for i in range(len(lst))]
#     # fp_multipass_counter_inner_arr=[0 for i in range(len(lst))]
#     # fn_multipass_counter_inner_arr=[0 for i in range(len(lst))]


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
#         tweetSentences=list(filter (lambda sentence: len(sentence)>1, tweetText.split('\n')))
#         tweetSentenceList_inter=flatten(list(map(lambda sentText: sent_tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
#         tweetSentenceList=list(filter (lambda sentence: len(sentence)>1, tweetSentenceList_inter))
#         # tweetSentenceList=[]
#         CTtieCandidateList=CTrie.displayTrie("",[])
#         # print('candidate list:')
#         # for annotated_candidate in CTtieCandidateList:
#         #     print(annotated_candidate)
        
#         for sentence in tweetSentenceList:
#             tweetWordList=getWords(sentence)
#             tweetWordList= [(token,idx) for idx,token in enumerate(tweetWordList)]
#             tweetWordList_stopWords=list(filter (lambda word: ((((word[0].strip()).strip(string.punctuation)).lower() in combined_list_filtered)|(word[0].strip() in string.punctuation)|(word[0].startswith('@'))), tweetWordList))
#             # phase 2 candidate tuples without stopwords for a sentence
#             c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords]
#             #c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords ]
#             # print(c)
#             sequences=[]
#             for k, g in groupby(enumerate(c), lambda element: element[0]-int(element[1][1])):
#                 sequences.append(list(map(itemgetter(1), g)))

#             # print(sequences)
#             for sequence in sequences:
#                 seq_candidate_list=get_Candidates(sequence, CTrie,False)
#                 # print('+>',sequence,seq_candidate_list)
#                 if(seq_candidate_list):
#                     for candidate_tuple in seq_candidate_list:
#                         candidateText=normalize(candidate_tuple[0])
#                         if(candidate_tuple[0]!='us'):
#                             annotated_mention_list.append(candidateText)

#     # print('-----',index,tweetText)

#     our_counter+=len(annotated_mention_list)
    

# #   # print(ritter_output,annotated_mention_list)
#     annotated_mention_list_tallying_array= [annotated_mention_list.copy() for i in range(len(lst))]

#     ritter_output=list(map(lambda element: element.lower().strip(),str(row['Output']).split(',')))
#     ritter_output=list(filter(lambda element: element !='', ritter_output))
#     all_postitive_ritter_counter_inner=len(ritter_output)
#     # print('=>', annotated_mention_list, ritter_output)
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

#     # print(unrecovered_annotated_mention_list)
#     unrecovered_annotated_mention_list_outer_ritter.extend(unrecovered_annotated_mention_list)
#     fn_ritter_counter_inner=len(unrecovered_annotated_mention_list)
#     fp_ritter_counter_inner=all_postitive_ritter_counter_inner- tp_ritter_counter_inner

#     tp_ritter_counter+= tp_ritter_counter_inner
#     fp_ritter_counter+=fp_ritter_counter_inner
#     fn_ritter_counter+=fn_ritter_counter_inner

#     for elem in lst:

#         all_postitive_multipass_counter_inner=0
#         tp_multipass_counter_inner=0
#         fp_multipass_counter_inner=0
#         fn_multipass_counter_inner=0

#         multipass_output_list=ast.literal_eval(str(row['output_col_'+str(elem)]))
#         multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
#         multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

#         all_postitive_multipass_counter_inner=len(multipass_output_list_flat)

#         annotated_mention_tally_list= annotated_mention_list_tallying_array[elem]

#         # print('==>', elem, annotated_mention_tally_list,multipass_output_list_flat)

#         while(annotated_mention_tally_list):
#             if(len(multipass_output_list_flat)):
#                 annotated_candidate= annotated_mention_tally_list.pop()
#                 if(annotated_candidate in multipass_output_list_flat):
#                     multipass_output_list_flat.pop(multipass_output_list_flat.index(annotated_candidate))
#                     tp_multipass_counter_inner+=1
#                 else:
#                     unrecovered_annotated_mention_list_multipass[elem].append(annotated_candidate)
#             else:
#                 unrecovered_annotated_mention_list_multipass[elem].extend(annotated_mention_tally_list)
#                 break

#           # multipass_output_list=list(filter(lambda element: element !='', multipass_output_list))
#         # print(unrecovered_annotated_mention_list_multipass[elem])
#         unrecovered_annotated_mention_list_outer_multipass[elem].extend(unrecovered_annotated_mention_list_multipass[elem])
#         fn_multipass_counter_inner=len(unrecovered_annotated_mention_list_multipass[elem])
#         fp_multipass_counter_inner=all_postitive_multipass_counter_inner- tp_multipass_counter_inner

#         tp_multipass_counter_arr[elem]+= tp_multipass_counter_inner
#         fp_multipass_counter_arr[elem]+=fp_multipass_counter_inner
#         fn_multipass_counter_arr[elem]+=fn_multipass_counter_inner


        
# #       # unrecovered_annotated_mention_list_outer_ritter.extend(unrecovered_annotated_mention_list)
# #       # for unrecovered_mention in unrecovered_annotated_mention_list:
# #       #   if (unrecovered_mention!=''):
# #       #       if(unrecovered_mention not in unrecovered_annotated_mention_list_outer):
# #       #           unrecovered_annotated_mention_list_outer.append(unrecovered_mention)
# #       #       if(unrecovered_mention.lower() not in unrecovered_annotated_candidate_list_outer):
# #       #           unrecovered_annotated_candidate_list_outer.append(unrecovered_mention.lower())
# #       # print(index, unrecovered_annotated_candidate_list_outer)

# # # print(index)
# # # for candidate in unrecovered_annotated_candidate_list_outer:
# # #     print(candidate)
# # # print(unrecovered_annotated_candidate_list_outer)


# # # candidateList=CTrie.displayTrie("",[])
# # # print('candidate list:', len(candidateList), len(all_entity_candidates))

# print('===',our_counter,sum(my_tally_arr_ritter))
# print('ritter numbers: ',tp_ritter_counter,fp_ritter_counter,fn_ritter_counter)
# ritter_precision= tp_ritter_counter/(tp_ritter_counter+fp_ritter_counter)
# ritter_recall= tp_ritter_counter/(tp_ritter_counter+fn_ritter_counter)

# multipass_precision_arr=[tp_multipass_counter_arr[index]/(tp_multipass_counter_arr[index]+fp_multipass_counter_arr[index]) for index in range(len(tp_multipass_counter_arr))]
# multipass_recall_arr=[tp_multipass_counter_arr[index]/(tp_multipass_counter_arr[index]+fn_multipass_counter_arr[index]) for index in range(len(tp_multipass_counter_arr))]

# print('single-pass: ',ritter_precision,ritter_recall)
# for index in range(len(multipass_precision_arr)):
#     print(index,multipass_precision_arr[index],multipass_recall_arr[index])

# print(len(unrecovered_annotated_mention_list_outer_ritter),len(unrecovered_annotated_mention_list_outer_multipass[0]),len(unrecovered_annotated_mention_list_outer_multipass[1]),len(unrecovered_annotated_mention_list_outer_multipass[2]),len(unrecovered_annotated_mention_list_outer_multipass[3]),
#   len(unrecovered_annotated_mention_list_outer_multipass[4]),len(unrecovered_annotated_mention_list_outer_multipass[5]),len(unrecovered_annotated_mention_list_outer_multipass[6])
#   ,len(unrecovered_annotated_mention_list_outer_multipass[7]),len(unrecovered_annotated_mention_list_outer_multipass[8]),len(unrecovered_annotated_mention_list_outer_multipass[9]),
#   len(unrecovered_annotated_mention_list_outer_multipass[10]),len(unrecovered_annotated_mention_list_outer_multipass[11]),len(unrecovered_annotated_mention_list_outer_multipass[12])
#   )





#--------------------------------------------------------annotating and checking performance for 1Million tweets file

our_counter=0
our_recovered_counter=0

our_counter_mention=0
our_recovered_mention=0

#----------------------commenting out since annotations already retrieved
tweets_unpartitoned=pd.read_csv("deduplicated_test_output_all_runs.csv",sep =',', keep_default_na=False)

# bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv",sep =',', keep_default_na=False)

bigger_tweet_dataframe=pd.read_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv",sep =',', keep_default_na=False)

bigger_tweet_dataframe['annotation'] = ''
bigger_tweet_dataframe['annotation'] = bigger_tweet_dataframe['annotation'].apply(list)

# lst=[0,20,40,60,80,100,110]
lst=[0,10,20,30,40]

all_postitive_reintroduction_counter=0
tp_reintroduction_counter_arr=[0 for i in range(len(lst))]
fp_reintroduction_counter_arr=[0 for i in range(len(lst))]
fn_reintroduction_counter_arr=[0 for i in range(len(lst))]

unrecovered_annotated_mention_list_outer_reintroduction = [[] for i in range(len(lst))]

#----------------------commenting out since annotations already retrieved
# using a previously annotated file to set the CTrie
for index,row in tweets_unpartitoned.iterrows():

    tweet_in_first_five_hundred=str(row['First_five_hundred'])
    tweetText=str(row['TweetText'])
    annotated_mention_list=[]

    if(tweet_in_first_five_hundred!=''):
        
        tweet_level_candidate_list=str(row['Annotations']).split(';')
        for tweet_level_candidates in tweet_level_candidate_list:
            sentence_level_cand_list= tweet_level_candidates.split(',')
            annotated_mention_list.extend(sentence_level_cand_list)
        # for index in range(len(ritter_output)):
        annotated_mention_list=list(map(lambda element: element.lower().strip(),annotated_mention_list))
        annotated_mention_list=list(filter(lambda element: element !='', annotated_mention_list))

        for annotation in annotated_mention_list:
            if annotation not in all_entity_candidates:
                all_entity_candidates.append(annotation)
            CTrie.setitem_forAnnotation(annotation.split())

    else:
        break

additional_candidates1=['ivanka trump', 'president donald trump', 'jared kushner','bill clinton','president trump','anthony weiner','joe biden','anthony','democrat','john terry','jose','chris cornell','potus','morgan','nationalism','religion world tour','rust belt','trumps','spicer','assange']
additional_candidates2=['poc', 'luther', 'comey', 'hamas', 'sec of state', 'whitehouse', 'kendrick', 'julian assange', 'asian','andrew breitbart', 'hrc', 'allen', 'yahoo sports', 'potus trump', 'indonesia', 'latin', 'huffpost', 'houston', 'labour party', 'legally blonde', 'sports illustrated', 'ox', 'trump', 'tom', 'high hopes', 'ritz', 'deadline', 'peter', 'trump war room', 'labour', 'washington wizards',  'maggie', 'religion world tour', 'justin bieber', 'sharia', 'jurgen', 'jose',  'james comey', 'tristan', 'hatch', 'johnson', 'tupac', 'blues', 'geno', 'trumps', 'pa', 'carat',  'senate intelligence committee', 'patti', 'pos', "o'reilly", 'j.k. rowling', 'saudi king salman', 'diamond', 'rick', 'latinos', 'colluded', 'hajj', 'buzzfeed news', 'chris', 'michael', 'zlatan', 'observer', 'ceo', 'ann', 'scavino', 'xhaka', 'us presidents', 'alderaan',  'sex addiction', "trump's radar", 'tower', 'ainge', 'wp', 'becker', 'cbsnews', 'blotus', 'americas', 'valencia', 'arabia', 'nbc sports', 'earn', 'striker', 'trumpcare', 'mavs', 'pbo', 'blk', 'gas', 'pichichi', 'wayne bridge', 'airman',  'mcdonald', 'senators', 'colorado', 'isiah', 'larry', 'disney', 'cornel west', 'clas', 'ku', 'scott', 'sf', 'whoopi goldberg', 'sarah palin', 'casino', 'dreaming', 'maxine', 'welsh',  'lords', 'nafta', 'us president', 'jeffery lord', 'shapiro', 'sharif', 'first amendment', 'yahoo', 'stream', 'california democrats',  'us news', 'africa', 'ahca', 'isis-lite',  'michael dorf', 'watch lebron', 'kentucky',  'david lee', 'dc', 'drumpf', 'tru', 'kawhi', 'parker', 'hh', 'ali', 'bush',  'elijah', 'kccinews', 'intel', 'reds', 'byron york', 'state dept', 'copa', 'tump', 'brady', 'madrid', 'aclu', 'andrews', 'finn', 'eibar', 'brad', 'ipo', "trump's administration", 'donald trump memes', 'google', 'infowars',  'weekly standard', 'bbcworld', 'softbank', 'knicks',  'okc', 'mike',  'nbcnews', 'butler', 'k.t',  'anthony weiner', 'fcc', 'matrix', 'gorsuch', 'maya angelou', 'tommy', 'torys', 'bridge', 'demorats', 'gulf', 'gsw', 'dennis wise', 'russian agent', 'brooklyn nets', 'dick gregory', 'turkey', 'kayleigh mcenany', 'hall of presidents', 'msn', 'ray', 'jeffrey lord', 'biko', 'romphim', 'kim', 'afrikan perspectives', 'bleacherreport', 'repubs', 'charles barkley', 'smith', 'gooner', 'white house', 'nwo', 'jeffrey', 'capitol', 'maher', 'sunderland', 'garvey', 'malcolm x sought', 'fairfax', 'salutes', 'breeze', 'arsène wenger', 
'cardiff', 'cornell west', 'mcclatchy', 'sc', 'nazis', 'nigerians', 'eveningstandard', 'jackson', 'olympic', 'stamford bridge', 'zucker', 'ford', 'lowry', 'williams', 'tillerson', 'archie', 'ailes', 'the rock', 'ebola', 'hallmark', 'spanish', 'mps', 'lagos', 'cavs', 'suarez', 'truthfeednews', 'sau', 'lw', 'muhammad', 'roland', 'islamaphobic', 'santos', 'coutinho', 'canadian', 'spicer', 'saudia',  'scotland', 'bob woodward', 'rice',  'potus',  'hector bellerin', 'mattis', 'conway', 'chai', 'sean', 'wc', 'sturridge', 'paul pierce', 'greens', 'king abdulaziz medal', 'kingston', 'libertarian', 'roger', 'jack', 'commander in chief', 'thomas', 'stamford', 'brown', 'josh', 'trump organization', 'saudi arabia', 
'chuck', 'feds', 'nbatv', 'stein',  'meg', 'jesus', 'flynn','melania trump', 'huffpo', 'penguins', 'jewish', 'merkel', 'greece', 'messi', 'neymar', 'ricky',  'irish', 'klan', 'andrew', 'nd', 'portland', 'stu', 'santa', 'intel agencies', 'adam silver',  'president trump',  'monmouth', 'celtics', 'sa',  'los', 'nationalism',  'space jam',  'nbas', 'republicans', 'david', 'scarborough', 'john mccain', 'tiffany trump', 'titanic', 'eric', 'korver', 'nbc news', 'susan rice', 'howard', 'ki', 'pete', 'arabian', 'rompers', 'harvard study', 'macron', 'sportscenter', 'barkley', 'prophet', 'nsa', 'border infrastructure', 'baker', 'jews', '1b', 'rev',  'ga', 'jc', 'duke', 'democrat', "trump's treasury secretary", 'zidane', 'matt', 'ied', 'abbas', 'kellyanne conway', 'amer', 'bellerin', 'ryan', 'danny', 'supreme court', 'cher', 'kyle', 'western', 'doral', 'ww3', 'apple', 'jarrett', 'barry', 'amazon', 'phil', 'alec baldwin', 'rust belt', 'miller', 'white hou', 'kronke', 'mustafi',   'cristiano',  'first lady', 'eu', 'demoncrats', 'treasury', 'mt', 'liberal resistance', 'nra', 'amy', 'e.p.a', 'anderson pooper', 'texas', 'lance',  'elizabeth', 'kelly', 'usaf', 'stan', 'motd', 'indians', 'john terry', 'blazers', 'travel ban', 'punk', 'cameron', 'baba',
'george takei', 'jo', 'tommy milone',  'morals', 'prime minister', 'chuck schumer', 'chris cornell', 'sheiks', 'malcolm', 'romanian', 'norway', 'neocon', 'canadians', 'erdogan', 'champs league', 'brooklyn', 'uk', 'sisi', 'i.t',  'greg', 'allah', 'larry bird', 'kante', 'birmingham', 'fp', 'nba', 'san francisco',  'daily caller', 'yuri kochiyama', 'election day', 'robert', 'sankara', 'crowder', 'chucky', 'indian',  'batshuayi', 'newsflash', 'doj', 'ola ray', 'korean', 'us armed forces', 'westbrook', "handmaid's tale",'rick santorum','joe lieberman','paul craig roberts','kim jong','patti lupone','bernie sanders',
'bradley', 'john', 'kevin', 'imam', 'enes kanter', 'alex', 'epshteyn', 'truman', 'nbc',  'muslim brotherhood','associated press', 'enes', 'nawaz', 'kenny', 'zimbabwe', 'russia collusion', 'illegal alien', 'ireland', 'kevin love', 'kyle korver', 'farron', 'saudi', 'chicago', 'malaga', 'group of notre dame', 'oscar', 'snapchat', 'netflix', 'steagall',  'bbc', 'reagan', 'obam', 'sanders', 'atlanta',  'jeb', 'darrell', 'cambridge analytica', 'jaylen brown', 'seth',  'chad brown', 'ivanka', 'malcolm x', 'saddam', 'morgan', 'shannon', 'al', 'jaylen', 'wayne', 'paul', 'america', 'brian', 'kurds', 'island', 'berkeley', 'abc', 'barack', 'stephen','fergie', 'mcconnell', 'police state', 'marine one', 'justice department', 'premier league', 'schmidt', 'n korea', 'mandela', 'omar', 'goldman',  'russ', 'mal', 'liam', 'georginio wijnaldum', 'sheryl', 'michelle obama', 'host bob beckel', 'messiah', 'h.r', 'superhero', 'emirate', 'daniel', 'victor moses', 'charles', 'palin', 'pats', 'steph', 'pippa', 'us state', 'leahy', 'washingtonpost', 'pogba', 'la liga', 'warriors', 'darren wilson', 'issa', 'malcol', 'mich', 'harry potter', 'weekend update', 'saudi king', 'middle east', 'kathy', 'simmons', 'flotus','whoopi', 'congressional republicans', 'superman', 'muslim', 'bulgaria', 'karma', 'wwe', 'pirates of the caribbean', 'godspeed', 'memorial day', 'jabbar', 'jason', 'jew', 'don mattingly', 'donald trump', 'world cup', 'gary', 'hitler', 'obama', 'house of cards', 'bil','tom hardy',  'nick', 'luis enrique', 'france', 'maduro']

additional_candidate_set= set(additional_candidates1)
additional_candidate_set.update(additional_candidates2)
additional_candidates=list(additional_candidate_set)
print(len(additional_candidates))
for additional_candidate in additional_candidates:
    # print(additional_candidate)
    CTrie.setitem_forAnnotation(additional_candidate.split())

# # output_df=tweets[['ID', 'HashTags', 'Tweet IDs', 'TweetText']]
# bigger_tweet_dataframe['annotation'] = ''
# bigger_tweet_dataframe['annotation'] = bigger_tweet_dataframe['annotation'].apply(list)

for index,row in bigger_tweet_dataframe.iterrows():
    # annotated_candidates=str(row['mentions_other'])
    
    unrecovered_annotated_mention_list=[]
    unrecovered_annotated_mention_list_reintroduction = [[] for i in range(len(lst))]

    unrecovered_annotated_mention_list_multipass = [[] for i in range(len(lst))]

    annotated_mention_list=[]

    #----------------------commenting out since annotations already retrieved
    tweetText=str(row['TweetText'])
    tweetSentences=list(filter (lambda sentence: len(sentence)>1, tweetText.split('\n')))
    tweetSentenceList_inter=flatten(list(map(lambda sentText: sent_tokenize(sentText.lstrip().rstrip()),tweetSentences)),[])
    tweetSentenceList=list(filter (lambda sentence: len(sentence)>1, tweetSentenceList_inter))
    # tweetSentenceList=[]
    CTtieCandidateList=CTrie.displayTrie("",[])
    # print('candidate list:')
    # for annotated_candidate in CTtieCandidateList:
    #     print(annotated_candidate)
    
    for sentence in tweetSentenceList:
        tweetWordList=getWords(sentence)
        tweetWordList= [(token,idx) for idx,token in enumerate(tweetWordList)]
        tweetWordList_stopWords=list(filter (lambda word: ((((word[0].strip()).strip(string.punctuation)).lower() in combined_list_filtered)|(word[0].strip() in string.punctuation)|(word[0].startswith('@'))), tweetWordList))
        # phase 2 candidate tuples without stopwords for a sentence
        c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords]
        #c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords ]
        # print(c)
        sequences=[]
        for k, g in groupby(enumerate(c), lambda element: element[0]-int(element[1][1])):
            sequences.append(list(map(itemgetter(1), g)))

        # print(sequences)
        for sequence in sequences:
            seq_candidate_list=get_Candidates(sequence, CTrie,False)
            # print('+>',sequence,seq_candidate_list)
            if(seq_candidate_list):
                for candidate_tuple in seq_candidate_list:
                    candidateText=normalize(candidate_tuple[0])
                    # if(index=='2174'):
                    #     print('===>',candidateText)
                    if(candidate_tuple[0].lower()!='us'):
                        annotated_mention_list.append(candidateText)

    print('-----',index)

    row['annotation']=annotated_mention_list

    # annotated_mention_list=ast.literal_eval(str(row['annotation']))

    all_postitive_reintroduction_counter+=len(annotated_mention_list)

    annotated_mention_list_tallying_array= [annotated_mention_list.copy() for i in range(len(lst))]

    for elem in range(len(lst)):

        threshold=lst[elem]

        all_postitive_reintroduction_counter_inner=0
        tp_reintroduction_counter_inner=0
        fp_reintroduction_counter_inner=0
        fn_reintroduction_counter_inner=0

        # multipass_output_list=ast.literal_eval(str(row['output_col_'+str(threshold)]))
        multipass_output_list=ast.literal_eval(row['output_col_'+str(threshold)])
        multipass_output_list=[eval(list_str) for list_str in multipass_output_list]
        # print(multipass_output_list)
        multipass_output_list_flat = [item.lower() for sublist in multipass_output_list for item in sublist]
        multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))

        all_postitive_reintroduction_counter_inner=len(multipass_output_list_flat)

        annotated_mention_tally_list= annotated_mention_list_tallying_array[elem]

        print('==>', elem, annotated_mention_tally_list,multipass_output_list_flat)

        while(annotated_mention_tally_list):
            if(len(multipass_output_list_flat)):
                annotated_candidate= annotated_mention_tally_list.pop()
                if(annotated_candidate in multipass_output_list_flat):
                    multipass_output_list_flat.pop(multipass_output_list_flat.index(annotated_candidate))
                    tp_reintroduction_counter_inner+=1
                else:
                    unrecovered_annotated_mention_list_reintroduction[elem].append(annotated_candidate)
            else:
                unrecovered_annotated_mention_list_reintroduction[elem].extend(annotated_mention_tally_list)
                break

          # multipass_output_list=list(filter(lambda element: element !='', multipass_output_list))
        print(unrecovered_annotated_mention_list_reintroduction[elem])
        unrecovered_annotated_mention_list_outer_reintroduction[elem].extend(unrecovered_annotated_mention_list_reintroduction[elem])
        fn_reintroduction_counter_inner=len(unrecovered_annotated_mention_list_reintroduction[elem])
        fp_reintroduction_counter_inner=all_postitive_reintroduction_counter_inner- tp_reintroduction_counter_inner

        tp_reintroduction_counter_arr[elem]+= tp_reintroduction_counter_inner
        fp_reintroduction_counter_arr[elem]+=fp_reintroduction_counter_inner
        fn_reintroduction_counter_arr[elem]+=fn_reintroduction_counter_inner
        

multipass_precision_arr=[tp_reintroduction_counter_arr[index]/(tp_reintroduction_counter_arr[index]+fp_reintroduction_counter_arr[index]) for index in range(len(tp_reintroduction_counter_arr))]
multipass_recall_arr=[tp_reintroduction_counter_arr[index]/(tp_reintroduction_counter_arr[index]+fn_reintroduction_counter_arr[index]) for index in range(len(tp_reintroduction_counter_arr))]


for index in range(len(multipass_recall_arr)):
    print(tp_reintroduction_counter_arr[index],fn_reintroduction_counter_arr[index],all_postitive_reintroduction_counter)
    print(index,multipass_recall_arr[index])

# bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_runs_with_annotation.csv", sep=',', encoding='utf-8',index=False)
# bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
# os.remove("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")
# os.rename("/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/output_1M_reintroduction_all_reintroduction_runs_with_annotations.csv")

bigger_tweet_dataframe.to_csv("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv", sep=',', encoding='utf-8',index=False)
os.remove("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")
os.rename("/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations_updated.csv","/home/satadisha/Desktop/GitProjects/data/eviction/output_1M_all_eviction_runs_with_annotations.csv")

# #----------------just the plot part

# missed_mention_numbers=[15816, 6222, 6184, 6122, 6090, 6022, 5987, 5964, 5949, 5935, 5929, 5928, 5887, 5872]
# print(list(tweets_unpartitoned.columns.values))
# rest_of_tweets= tweets_unpartitoned[index:]

# fig1 = plt.figure()
# plt.hold(True)
# x_values=[0,0,1,2,3,4,5,6]
# x_values=[0,0,1,2,3,4,5,6,7,8,9,10,11,12]
# y_values=  [len(unrecovered_annotated_mention_list_outer_ritter),len(unrecovered_annotated_mention_list_outer_multipass[0]),len(unrecovered_annotated_mention_list_outer_multipass[1]),len(unrecovered_annotated_mention_list_outer_multipass[2]),len(unrecovered_annotated_mention_list_outer_multipass[3]),
#     len(unrecovered_annotated_mention_list_outer_multipass[4]),len(unrecovered_annotated_mention_list_outer_multipass[5]),len(unrecovered_annotated_mention_list_outer_multipass[6])]

# y_values=[543, 337, 323, 318, 318, 316, 315, 313]
# y_values=[len(unrecovered_annotated_mention_list_outer_ritter)]
# y_values.extend([len(elem) for elem in unrecovered_annotated_mention_list_outer_multipass])

# y_values=[ritter_recall]
# y_values.extend([elem for elem in multipass_recall_arr])


# print(x_values)
# print(y_values)
# for array_index in range(0,18):
#     array_to_plot= eviction_ranking_precision_all_sketch_combined[array_index]

# plt.plot(x_values,y_values)
# plt.annotate(xy=[0,y_values[0]], s='single-pass-Ritter')
# plt.annotate(xy=[0,y_values[1]], s='single-pass-TwiCS')
# plt.yticks(np.arange(min(y_values), max(y_values), .01))
# plt.ylabel('recall value')


# plt.plot(x_values,missed_mention_numbers)
# plt.annotate(xy=[0,missed_mention_numbers[0]], s='single-pass-Ritter')
# plt.annotate(xy=[0,missed_mention_numbers[1]], s='single-pass-TwiCS')
# plt.ylabel('# of False negative mentions')
# plt.xlabel('reintroduction threshold')
# plt.yticks(np.arange(min(missed_mention_numbers)-100, max(missed_mention_numbers)+100, 500))


# f, (ax, ax2) = plt.subplots(2, 1, sharex=True)

# ax.plot(x_values[0:2],missed_mention_numbers[0:2])
# ax.plot(x_values[0],missed_mention_numbers[0], marker="v")
# # ax.set_ylim(8000, 18000) 
# # ax.set_yticks([3000])
# ax.yaxis.set_ticks(np.arange(8000, 18000, 3000))
# ax.annotate(xy=[0,missed_mention_numbers[0]], xytext=(1.5, 15000), s='single-pass-Ritter', arrowprops=dict(arrowstyle='-|>'), horizontalalignment='left', verticalalignment='bottom')

# ax2.plot(x_values[1:],missed_mention_numbers[1:], marker="v")
# # ax2.set_ylim(5500, 6250) 
# # ax2.set_yticks([50])
# ax2.yaxis.set_ticks(np.arange(5500, 6300, 50))
# ax2.plot(x_values[1],missed_mention_numbers[1], marker="v")
# ax2.annotate(xy=[0,missed_mention_numbers[1]], xytext=(1.5, missed_mention_numbers[1]-10), s='single-pass-TwiCS', arrowprops=dict(arrowstyle='-|>'), horizontalalignment='left', verticalalignment='bottom')


# # hide the spines between ax and ax2
# ax.spines['bottom'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # ax.spines['left'].set_visible(True)
# # ax2.spines['left'].set_visible(True)

# ax.xaxis.tick_top()
# # ax.yaxis.tick_left()
# ax.tick_params(labeltop=False)  # don't put tick labels at the top
# ax2.xaxis.tick_bottom()
# # ax2.yaxis.tick_left()

# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
# ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
# ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

# kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

# # ax.set_ylabel('# of False negative mentions')
# f.text(.025, .5, '# of False negative mentions', ha='center', va='center', rotation='vertical')
# ax2.set_xlabel('reintroduction threshold')
# # y_tick_labels=[str(e) for e in np.arange(8000, 18000, 3000)]
# # ax.set_yticklabels(y_tick_labels)
# # plt.setp(ax.get_yticklabels(), visible=True)

# lgd=plt.legend(bbox_to_anchor=(1, 1), loc=9, prop={'size': 4}, borderaxespad=0.)
# plt.title('<--single-to-multi-pass-EMD-->')
# # plt.savefig('single-to-multi-pass-EMD.png', dpi = 900, bbox_extra_artists=(lgd,), bbox_inches='tight')
# plt.show()

