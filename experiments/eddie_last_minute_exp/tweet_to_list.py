import pandas as pd
tweets=pd.read_csv("tweets_1million_for_others.csv",sep =',')
# tweets=tweets.dropna()
tweet_list=tweets['TweetText'].tolist()

tweet_list_filtered=[]
for i in tweet_list:
	if(isinstance(i, str)):
		tweet_list_filtered.append(i)

file = open("tweet_text_only.txt", "w")
for i in tweet_list_filtered:
	# print(i)
	# print(type(i))
	file.write(i+"\n")

# print(len(dedup))
file.close() 