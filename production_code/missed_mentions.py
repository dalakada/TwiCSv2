import numpy as np
import pandas as pd

tweets_unpartitoned=pd.read_csv("deduplicated_test_output.csv",sep =',', keep_default_na=False)
annotated_candidate_list=[]
annotated_mention_list=[]
ritter_list=[]

our_counter=0
our_recovered_counter=0
our_counter_mention=0
our_recovered_mention=0
# input_df= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence', 'phase1Candidates','start_time','entry_batch'))
for index,row in tweets_unpartitoned.iterrows():
	# annotated_candidates=str(row['mentions_other'])
	tweet_in_first_five_hundred=str(row['First_five_hundred'])
	if(tweet_in_first_five_hundred!=''):
		ritter_output=str(row['Output']).split(',')
		annotated_mention_list=[]
		tweet_level_candidate_list=str(row['Annotations']).split(';')
		for tweet_level_candidates in tweet_level_candidate_list:
			sentence_level_cand_list= tweet_level_candidates.split(',')
			annotated_mention_list.extend(sentence_level_cand_list)
		# for index in range(len(ritter_output)):
		while(annotated_mention_list):
			# print(ritter_output.pop())
			# print(ritter_output)