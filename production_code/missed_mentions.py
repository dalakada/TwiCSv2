import numpy as np
import pandas as pd

tweets_unpartitoned=pd.read_csv("deduplicated_test_output.csv",sep =',', keep_default_na=False)
unrecovered_annotated_candidate_list_outer=[]
unrecovered_annotated_mention_list_outer=[]
ritter_list=[]

our_counter=0
our_recovered_counter=0
our_counter_mention=0
our_recovered_mention=0
# input_df= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence', 'phase1Candidates','start_time','entry_batch'))
for index,row in tweets_unpartitoned.iterrows():
	# annotated_candidates=str(row['mentions_other'])
	unrecovered_annotated_mention_list=[]
	tweet_in_first_five_hundred=str(row['First_five_hundred'])
	if(tweet_in_first_five_hundred!=''):
		ritter_output=list(map(lambda element: element.strip(),str(row['Output']).split(',')))
		ritter_output=list(filter(lambda element: element !='', ritter_output))
		annotated_mention_list=[]
		tweet_level_candidate_list=str(row['Annotations']).split(';')
		for tweet_level_candidates in tweet_level_candidate_list:
			sentence_level_cand_list= tweet_level_candidates.split(',')
			annotated_mention_list.extend(sentence_level_cand_list)
		# for index in range(len(ritter_output)):
		annotated_mention_list=list(map(lambda element: element.strip(),annotated_mention_list))
		annotated_mention_list=list(filter(lambda element: element !='', annotated_mention_list))
		# print(ritter_output,annotated_mention_list)
		
		while(annotated_mention_list):
			if(len(ritter_output)):
				annotated_candidate= annotated_mention_list.pop()
				if(annotated_candidate in ritter_output):
					ritter_output.pop()
				else:
					unrecovered_annotated_mention_list.append(annotated_candidate)
			# print(ritter_output.pop())
			# print(ritter_output)
			else:
				unrecovered_annotated_mention_list.extend(annotated_mention_list)
				break
		print(index, unrecovered_annotated_mention_list)
		# unrecovered_annotated_mention_list_outer.append(unrecovered_annotated_mention_list)
		for unrecovered_mention in unrecovered_annotated_mention_list:
			if (unrecovered_mention!=''):
				if(unrecovered_mention not in unrecovered_annotated_mention_list_outer):
					unrecovered_annotated_mention_list_outer.append(unrecovered_mention)
				if(unrecovered_mention.lower() not in unrecovered_annotated_candidate_list_outer):
					unrecovered_annotated_candidate_list_outer.append(unrecovered_mention.lower())
		# print(index, unrecovered_annotated_candidate_list_outer)
	else:
		break
# print(index)
# for candidate in unrecovered_annotated_candidate_list_outer:
# 	print(candidate)
# print(unrecovered_annotated_candidate_list_outer)
print(len(unrecovered_annotated_mention_list_outer),len(unrecovered_annotated_candidate_list_outer))
# print(list(tweets_unpartitoned.columns.values))
rest_of_tweets= tweets_unpartitoned[index:]

