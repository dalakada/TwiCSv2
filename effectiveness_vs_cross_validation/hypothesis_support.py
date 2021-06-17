import re
import string
import csv
import numpy as np
import matplotlib.pyplot as plt
import emoji
import pandas as pd
import re
import copy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import gutenberg
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer

import SatadishaModule_final_trie as phase1
import phase2_Trie_baseline_reintroduction_effectiveness as phase2


list_of_filenames=['broad_twitter_corpus']#'tweets_3k' ,'venezuela','pikapika','ripcity','billnye']#,'roevwade','billdeblasio', 'wnut17_test.annotated','wnut17test.csv'],'wnut17_test.annotated'

path="/Users/satadisha/Documents/GitHub/BIO_annotations/"
path2="/Users/satadisha/Documents/GitHub/"

path3="/Users/satadisha/Documents/GitHub/broad_twitter_corpus/h.conll"

# cap       substr-cap      s-o-s cap       all-cap     non-cap     non disc
all_mentions_syntax=[0,0,0,0,0,0]

# gutenberg_text = ""
# for file_id in gutenberg.fileids():
#     gutenberg_text += gutenberg.raw(file_id)
# tokenizer_trainer = PunktTrainer()
# tokenizer_trainer.INCLUDE_ALL_COLLOCS = True
# tokenizer_trainer.train(gutenberg_text)

# my_sentence_tokenizer = PunktSentenceTokenizer(tokenizer_trainer.get_params())
# my_sentence_tokenizer._params.abbrev_types.add('dr')
# my_sentence_tokenizer._params.abbrev_types.add('c.j')
# my_sentence_tokenizer._params.abbrev_types.add('u.s')
# my_sentence_tokenizer._params.abbrev_types.add('u.s.a')

cachedStopWords = stopwords.words("english")
tempList=["i","and","or","other","another","across","unlike","anytime","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always","although"]
for item in tempList:
    if item not in cachedStopWords:
        cachedStopWords.append(item)
cachedStopWords.remove("don")
# cachedStopWords.remove("your")
# cachedStopWords.remove("up")
cachedTitles = ["mr.","mr","mrs.","mrs","miss","ms","sen.","dr","dr.","prof.","president","congressman"]
prep_list=["in","at","of","on","&;","v."] #includes common conjunction as well
# prep_list=[]
# article_list=[]
article_list=["a","an","the"]
conjoiner=["de"]
day_list=["sunday","monday","tuesday","wednesday","thursday","friday","saturday","mon","tues","wed","thurs","fri","sat","sun"]
month_list=["january","february","march","april","may","june","july","august","september","october","november","december","jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
chat_word_list=["nope","gee","hmm","bye","please","yrs","4get","ooh","ouch","am","tv","ima","psst","thanku","em","qft","ip","icymi","bdsm","ah","http","https","pm","omw","pts","pt","ive","reppin","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lmfao","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fucking","fuck","f*ck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]
string.punctuation=string.punctuation+'…‘’'

def rreplace(s, old, new, occurrence):
    if s.endswith(old):
        li = s.rsplit(old, occurrence)
        return new.join(li)
    else:
        return s

def all_capitalized(candidate):
    strip_op=candidate
    strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip())
    strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
    strip_op= rreplace(rreplace(rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
    prep_article_list=prep_list+article_list+conjoiner
    word_list=strip_op.split()
    for i in range(len(word_list)):
        word=word_list[i]
        if((word[0].isupper())|(word[0].isdigit())):
            continue
        else:
            if(word in prep_article_list):
                if (i!=0):
                    continue
                else:
                    return False
            else:
                return False
    return True


def check_feature_update(candidateText,position):
    #print(candidate_tuple)
    # if(non_discriminative_flag):
    #     return 7
    # candidateText=candidate_tuple[0]
    # position=candidate_tuple[1]
    global all_mentions_syntax
    word_list=candidateText.split()
    val=-99
    if candidateText.islower():
        all_mentions_syntax[4]+=1
        val=4
    elif candidateText.isupper():
        all_mentions_syntax[3]+=1
        val=3
    elif (len(word_list)==1):
        #start-of-sentence-check
        if all_capitalized(candidateText):
            if(int(position[0])==0):
                all_mentions_syntax[2]+=1
                val=2
            else:
                all_mentions_syntax[0]+=1
                val=0
        else:
            all_mentions_syntax[1]+=1
            val=1
    else:
        if(all_capitalized(candidateText)):
            all_mentions_syntax[0]+=1
            val=0
        else:
            all_mentions_syntax[1]+=1
            val=1
    return val

def iscapitalized(tweetWordList):
    for word in tweetWordList:
        if(word[0].islower()):
            return False
    return True
def not_usr_account(mention_position, tuples):

    # word_tag_tuples=list(word_tag_tuples)
    print('==>',tuples)
    print(mention_position)
    retVal=True
    if(mention_position[0]>0):
        word_tag_tuple=tuples[mention_position[0]-1]
        if(word_tag_tuple[0]=='@'):
            retVal=False
    print(retVal)
    return retVal

def get_entities(word_tag_tuples):
    
    mentions=[]
    positionLists=[]
    candidateMention=''
    arg_word_tag_tuples=copy.deepcopy(list(word_tag_tuples))
    #emoji.get_emoji_regexp().sub(u'', candidateMention)
    ind=0
    for ind, tup in enumerate(arg_word_tag_tuples):
        candidate=tup[0]
        tag=tup[1]
        if(tag=='O'):
            if(candidateMention):
                if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
                    mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()
                    if mention_to_add.endswith("'s"):
                        li = mention_to_add.rsplit("'s", 1)
                        mention_to_add=''.join(li)
                    elif mention_to_add.endswith("’s"):
                        li = mention_to_add.rsplit("’s", 1)
                        mention_to_add=''.join(li)
                    else:
                        mention_to_add=mention_to_add
                    if(mention_to_add!=''):
                        # if(not_usr_account(position,arg_word_tag_tuples)):
                        mentions.append(mention_to_add)
                        positionLists.append(position)
            candidateMention=''
            position=[]
        else:
            if (tag=='B'):
                if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
                    mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()
                    if mention_to_add.endswith("'s"):
                        li = mention_to_add.rsplit("'s", 1)
                        mention_to_add=''.join(li)
                    elif mention_to_add.endswith("’s"):
                        li = mention_to_add.rsplit("’s", 1)
                        mention_to_add=''.join(li)
                    else:
                        mention_to_add=mention_to_add
                    if(mention_to_add!=''):
                        # if(not_usr_account(position,arg_word_tag_tuples)):
                        mentions.append(mention_to_add)
                        positionLists.append(position)
                candidateMention=candidate
                position=[ind]
            else:
                candidateMention+=" "+candidate
                position.append(ind)
        # if (tag=='B'):
        #     if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))):
        #         mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip()
        #         if(mention_to_add):
        #             mentions.append(mention_to_add)
        #     candidateMention=candidate
        # else:
        #     candidateMention+=" "+candidate
    if(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()):
        if((not candidateMention.strip().startswith('#'))&(not candidateMention.strip().startswith('@'))&(not candidateMention.strip().startswith('https:'))):
            mention_to_add=emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).strip()
            if(mention_to_add!=''):
                # if(not_usr_account(position,arg_word_tag_tuples)):
                mentions.append(mention_to_add)
                positionLists.append(position)
        # mentions.append(emoji.get_emoji_regexp().sub(u'', candidateMention).strip(string.punctuation).lower().strip())
    # print('extracted mentions:', mentions)
    return mentions,positionLists



################################# EXPERIMENT SUPPORTING HYPOTHESIS E-H2 #################################
for name in list_of_filenames:

    # f= open(path+name, "r")

    f= open(path3, "r")

    file_text=f.read()

    sentences=file_text.split('\n\n')

    sentID=0

    for sentence in sentences:
        word_lines= sentence.split('\n')
        tweetList=[]
        mentionList=[]
        for line in word_lines:
            if(line):
                tabs=line.split('\t')
                word=tabs[0] 
                tag=tabs[1].split('-')[0]
                tweetList.append(word)
                mentionList.append(tag)
        word_tag_tuples=zip(tweetList,mentionList)
        tweetText=' '.join(tweetList)
        tweetText=tweetText.strip()
        # print(sentID,tweetList,mentionList)
        sentID+=1
        
    ### for all other BIO annotated files
    # tweets_unpartitoned=pd.read_csv(path+name+"_BIOannotated_twokenized.csv",sep =',',keep_default_na=False)
    # tweets_unpartitoned=pd.read_csv(path2+name,sep =',',keep_default_na=False)

    # sentIDs=tweets_unpartitoned['Sentence #'].unique().tolist()

    # for row in tweets_unpartitoned.itertuples():

    # for sentID in sentIDs:
    #   sentDF=tweets_unpartitoned[tweets_unpartitoned['Sentence #']==sentID]
    #   tweetList=sentDF['Word'].tolist()
    #   tweetText=' '.join(tweetList)
    #   tweetText=tweetText.strip()
    #   mentionList=sentDF['Tag'].tolist()
    #   word_tag_tuples=zip(tweetList,mentionList)

    #   # print([tup for tup in word_tag_tuples])
    #   print(sentID,tweetText,mentionList)
    ### for all other BIO annotated files
        # tweetText=str(row.TweetText)
        # tweetList=tweetText.split()

        entities, positionLists= get_entities(word_tag_tuples)
        discriminative=False

        if(tweetText.islower()|tweetText.isupper()|iscapitalized(tweetList)):
            discriminative=True

        print(tweetText)

        for ind, entity in enumerate(entities):
            if(discriminative):
                all_mentions_syntax[5]+=1
                retVal=5
            else:
                position=positionLists[ind]
                retVal=check_feature_update(entity,position)
            print(entity,retVal)
            print(all_mentions_syntax)
        print('===')
