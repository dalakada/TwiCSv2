
from nltk.tokenize import word_tokenize
import string
import pandas as pd 
import itertools
import copy
import sys

tweets=pd.read_csv("tweets_1million_for_others.csv",sep =',')
# tweets=tweets[:1000:]

#input comes all lowercase
def makeupperdynamic(input):
    output=[]
    # tokenized_words =word_tokenize(input)
    tokenized_words=input.split()
    # print(tokenized_words)
    number_of_permutations=list(itertools.product([0,1], repeat=len(tokenized_words)))


    for possibility in number_of_permutations:
        output.append(copy.deepcopy(tokenized_words))
    # print(output)

    new=[]
    for idx,possibility in enumerate(number_of_permutations):
        # print(idx)
        for i in range(len(tokenized_words)):
            on_or_off=possibility[i]
            # print(on_or_off)
            if(on_or_off==1):
                # print(i)
                # mert=tokenized_words[i][0].upper()
                # output.append(mert)
                list1=list(output[idx][i])
                list1[0]=list1[0].upper()
                str1 = ''.join(list1)        
                # print(str1)
                output[idx][i]=str1
                # print(output[idx])
            else:
                output[idx][i]

        new.append(copy.deepcopy(output[idx]))
    # print(new)
            
    new_flatten=[]
    for i in new:
        str1 = ' '.join(i)
        new_flatten.append(str1)
    # print(new_flatten)

    new_flatten = new_flatten[:-1]
    # print(tokenized_words)
    # output=tokenized_words
    return new_flatten

# output=makeupperdynamic("ufc light heavyweight champion daniel cormier")
# print(output)





def intersect(a, b):
     return list(set(a) & set(b))

# print(my_list)

# my_anan=['a]
# print(my_list2)
# one_1_c=len(intersect(my_list,my_list2))


file_name="segment"+sys.argv[1]
file = open(file_name,"r") 
my_list2=[]
for line in file: 
    # print (line)
    line_new=line.strip("\n")
    line_new_new=line_new.lower()
    ine_new_new_new=line_new_new.strip()
    my_list2.append(ine_new_new_new)





# print("len brooo",len(my_list),len(my_list2))
flat_tweets=""
tweets_list=[]
for index, row in tweets.iterrows():
    tweetText=str(row['TweetText'])
    tweets_list.append(tweetText)
    # flat_tweets=''.join([flat_tweets,tweetText])
    # flat_tweets=flat_tweets+" "+str(index)+tweetText 
    print(index)

flat_tweets=''.join(tweets_list)
filtered=[]

for idx,candidate in enumerate(my_list2):
    print(idx,sys.argv[1])
    possibilitiess=makeupperdynamic(candidate)
    matched= [val2 for val2 in possibilitiess if val2 in flat_tweets]
    print(matched)
    #match found
    # if(index!=1):
        # print(candidate)
    if(len(matched)>0):
        # print(candidate)
        filtered.append(candidate)
        # print(flat_tweets[index])
        # correct_form=string.capwords(candidate)
        # if(correct_form not in flat_tweets):
        #     print(correct_form,candidate)


file = open("segment_"+sys.argv[1]+"output", "w")

for i in filtered:
    file.write(i+"\n")
# print(filtered)

# print(one_1_c)

# print(list(set(my_list) - set(filtered)))
# print(flat_tweets)