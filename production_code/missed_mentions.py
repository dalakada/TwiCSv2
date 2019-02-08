import numpy as np
import pandas as pd
import ast

tweets_unpartitoned=pd.read_csv("deduplicated_test_output_all_runs.csv",sep =',', keep_default_na=False)
# unrecovered_annotated_candidate_list_outer=[]
unrecovered_annotated_mention_list_outer_ritter=[]
unrecovered_annotated_mention_list_outer_multipass = [[] for i in range(2)]


ritter_list=[]

our_counter=0
our_recovered_counter=0
our_counter_mention=0
our_recovered_mention=0



lst=[0,1]
for 
# input_df= pd.DataFrame(columns=('tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence', 'phase1Candidates','start_time','entry_batch'))
for index,row in tweets_unpartitoned.iterrows():
	# annotated_candidates=str(row['mentions_other'])
	unrecovered_annotated_mention_list=[]
	unrecovered_annotated_mention_list_multipass = [[] for i in range(2)]



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
		annotated_mention_list_tallying_array= []

		for elem in lst:
			annotated_mention_list_tallying_array.append(annotated_mention_list)
		
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
		unrecovered_annotated_mention_list_outer_ritter.extend(unrecovered_annotated_mention_list)

		for elem in lst:
			multipass_output_list=ast.literal_eval(str(row['output_col_'+str(elem)]))
			multipass_output_list_flat = [item for sublist in multipass_output_list for item in sublist]
			multipass_output_list_flat=list(filter(lambda element: element !='', multipass_output_list_flat))
			annotated_mention_tally_list= annotated_mention_list_tallying_array[elem]

			while(annotated_mention_tally_list):
				if(len(multipass_output_list_flat)):
					annotated_candidate= annotated_mention_tally_list.pop()
					if(annotated_candidate in multipass_output_list_flat):
						multipass_output_list_flat.pop()
					else:
						unrecovered_annotated_mention_list_multipass[elem].append(annotated_candidate)
				else:
					unrecovered_annotated_mention_list_multipass[elem].extend(annotated_mention_tally_list)
					break

				# multipass_output_list=list(filter(lambda element: element !='', multipass_output_list))
			print(index, elem, unrecovered_annotated_mention_list_multipass[elem])
			unrecovered_annotated_mention_list_outer_multipass[elem].extend(unrecovered_annotated_mention_list_multipass[elem])


		
		# unrecovered_annotated_mention_list_outer_ritter.extend(unrecovered_annotated_mention_list)
		# for unrecovered_mention in unrecovered_annotated_mention_list:
		# 	if (unrecovered_mention!=''):
		# 		if(unrecovered_mention not in unrecovered_annotated_mention_list_outer):
		# 			unrecovered_annotated_mention_list_outer.append(unrecovered_mention)
		# 		if(unrecovered_mention.lower() not in unrecovered_annotated_candidate_list_outer):
		# 			unrecovered_annotated_candidate_list_outer.append(unrecovered_mention.lower())
		# print(index, unrecovered_annotated_candidate_list_outer)
	else:
		break
# print(index)
# for candidate in unrecovered_annotated_candidate_list_outer:
# 	print(candidate)
# print(unrecovered_annotated_candidate_list_outer)
print(len(unrecovered_annotated_mention_list_outer_ritter),len(unrecovered_annotated_candidate_list_outer))
# print(list(tweets_unpartitoned.columns.values))
rest_of_tweets= tweets_unpartitoned[index:]

