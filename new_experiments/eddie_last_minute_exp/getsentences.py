import pandas as pd 

tweets=pd.read_csv("tweet_base9.csv",sep =',')


sentences = tweets['TweetSentence'].tolist()
thefile = open('sentences.txt', 'w')


for item in sentences:
  thefile.write("%s\n" % item)