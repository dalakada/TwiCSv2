# coding: utf-8
from nltk.corpus import stopwords
import pandas  as pd
import NE_candidate_module as ne
from nltk.tokenize import sent_tokenize, word_tokenize
import string
import copy
import numpy
import math
from itertools import groupby
from operator import itemgetter
from scipy import stats
import SVM as svm
import statistics
import pandas as pd
import time
import datetime
import trie as trie
import re
import pickle

cachedStopWords = stopwords.words("english")
tempList=["i","and","or","other","another","across","were","you","then","still","till","nor","perhaps","otherwise","until","sometimes","sometime","seem","cannot","seems","because","can","like","into","able","unable","either","neither","if","we","it","else","elsewhere","how","not","what","who","when","where","who's","who’s","let","today","tomorrow","tonight","let's","let’s","lets","know","make","oh","via","i","yet","must","mustnt","mustn't","mustn’t","i'll","i’ll","you'll","you’ll","we'll","we’ll","done","doesnt","doesn't","doesn’t","dont","don't","don’t","did","didnt","didn't","didn’t","much","without","could","couldn't","couldn’t","would","wouldn't","wouldn’t","should","shouldn't","souldn’t","shall","isn't","isn’t","hasn't","hasn’t","wasn't","wasn’t","also","let's","let’s","let","well","just","everyone","anyone","noone","none","someone","theres","there's","there’s","everybody","nobody","somebody","anything","else","elsewhere","something","nothing","everything","i'd","i’d","i’m","won't","won’t","i’ve","i've","they're","they’re","we’re","we're","we'll","we’ll","we’ve","we've","they’ve","they've","they’d","they'd","they’ll","they'll","again","you're","you’re","you've","you’ve","thats","that's",'that’s','here’s',"here's","what's","what’s","i’m","i'm","a","so","except","arn't","aren't","arent","this","when","it","it’s","it's","he's","she's","she'd","he'd","he'll","she'll","she’ll","many","can't","cant","can’t","even","yes","no","these","here","there","to","maybe","<hashtag>","<hashtag>.","ever","every","never","there's","there’s","whenever","wherever","however","whatever","always"]
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
chat_word_list=["please","4get","ooh","idk","oops","yup","stfu","uhh","2b","dear","yay","btw","ahhh","b4","ugh","ty","cuz","coz","sorry","yea","asap","ur","bs","rt","lfmao","slfmao","u","r","nah","umm","ummm","thank","thanks","congrats","whoa","rofl","ha","ok","okay","hey","hi","huh","ya","yep","yeah","fyi","duh","damn","lol","omg","congratulations","fuck","wtf","wth","aka","wtaf","xoxo","rofl","imo","wow","fck","haha","hehe","hoho"]
string.punctuation=string.punctuation+'…‘’'



class EntityResolver ():


    def executor(self,TweetBase,CTrie,phase2stopwordList,z_score_threshold,raw_tweets_for_others):


        # SET CB
        candidate_featureBase_DF,data_frame_holder,phase2_candidates_holder,correction_flag=self.set_cb(TweetBase,CTrie,phase2stopwordList,z_score_threshold)
        
        candidate_featureBase_DF.to_csv("candidate_base_new.csv", sep=',', encoding='utf-8')

        # SET TF 
        untrashed_tweets=self.set_tf(data_frame_holder,
            candidate_featureBase_DF,
            phase2_candidates_holder,correction_flag)

        untrashed_tweets.to_csv("phase2output.csv", sep=',', encoding='utf-8')



        ######## EXPERIMENT FUNCTION STARTS #################################
        ########
        ########
        #input: tf, candidate_featureBase_DF 
        #output: incomplete_tweets[candidates_with_label], [good_candidates], [bad_candidates]
        self.set_column_for_candidates_in_incomplete_tweets(candidate_featureBase_DF,untrashed_tweets)
        ########
        ########
        # tp,fp,f1,accuracy calculations.
        # input: tf .[good_candidates],[annotation]
        # output : incomplete tweets.['tp'],['fp'],[f1]
        self.calculate_tp_fp_f1(z_score_threshold,untrashed_tweets)
        ########
        ########
        #SAVE INCOMING TWEETS FOR ANNOTATION FOR OTHERS
        self.raw_tweets_for_others=pd.concat([self.raw_tweets_for_others,raw_tweets_for_others ])
        ########
        ########
        ########
        # tp,fp,f1,accuracy calculations.
        # input: tf .[good_candidates],[annotation]
        # output : incomplete tweets.['tp'],['fp'],[f1]
        self.calculate_tp_fp_f1_for_others(self.raw_tweets_for_others)
        ########
        ########
        ######## EXPERIMENT FUNCTION ENDS ###################################




        # DROP TF
        just_converted_tweets=self.get_complete_tf(untrashed_tweets)
        self.incomplete_tweets=self.get_incomplete_tf(untrashed_tweets)

        #recording tp, fp , f1
        self.accuracy_tuples_prev_batch.append((just_converted_tweets.tp.sum(), just_converted_tweets.total_mention.sum(),just_converted_tweets.fp.sum(),just_converted_tweets.fn.sum()))


        #operations for getting ready for next batch.
        self.incomplete_tweets.drop('2nd Iteration Candidates', axis=1, inplace=True)
        self.counter=self.counter+1

        self.aggregator_incomplete_tweets= self.aggregator_incomplete_tweets.append(self.incomplete_tweets)
        self.just_converted_tweets=self.just_converted_tweets.append(just_converted_tweets)


        self.aggregator_incomplete_tweets.to_csv("all_incompletes.csv", sep=',', encoding='utf-8')


        self.just_converted_tweets.to_csv("all_converteds.csv", sep=',', encoding='utf-8')
        self.incomplete_tweets.to_csv("incomplete_for_last_batch.csv", sep=',', encoding='utf-8')



    def __init__(self):
        self.counter=0
        self.my_classifier= svm.SVM1('training.csv')


    def calculate_tp_fp_f1_generic(self,raw_tweets_for_others,state_of_art):


        unique_tweetIDs=raw_tweets_for_others['tweetID'].unique().tolist()

        column_annot_holder=[]
        column_candidates_holder=[]
        column_tweet_text_holder=[]

        for unique_tweetID in unique_tweetIDs:
            #TO DO : cast to int tweetID column of dataframe.
            group_by_tweet_id_df=raw_tweets_for_others[raw_tweets_for_others.tweetID==unique_tweetID]

            tweet_level_annot=[]
            tweet_level_candidates=[]
            tweet_level_tweets=""
            for index, row in group_by_tweet_id_df.iterrows():
                
                #annotation
                annot=list(row['annotation'])
                tweet_level_annot=annot+tweet_level_annot;

                #candidates from other systems
                tweet_level_candidates=list(row[state_of_art])

                #merging sentences into one tweet.
                sentence_tweet=str(row["TweetSentence"])
                tweet_level_tweets=sentence_tweet+" "+tweet_level_tweets

            column_annot_holder.append(copy.deepcopy(tweet_level_annot))
            column_candidates_holder.append(copy.deepcopy(tweet_level_candidates))
            column_tweet_text_holder.append(copy.deepcopy(tweet_level_tweets))

        # tweet_ids_df=pd.DataFrame(unique_tweetIDs,column_annot_holder,column_candidates_holder, columns=['tweetID','column_annot_holder','column_candidates_holder'])
        tweet_ids_df = pd.DataFrame({'tweetid': unique_tweetIDs,'tweet_text':column_tweet_text_holder, 'annotations': column_annot_holder,"candidates_holder": column_candidates_holder})
       
        # tweet_ids_df=pd.DataFrame(unique_tweetIDs,column_annot_holder,column_candidates_holder, columns=['tweetID','column_annot_holder','column_candidates_holder'])

        good_candidates = column_candidates_holder

        annotations= column_annot_holder

        true_positive_count=0
        false_positive_count=0
        false_negative_count=0
        ambigious_not_in_annotation=0

        true_positive_holder = []
        false_negative_holder=[]
        false_positive_holder=[]
        total_mention_holder=[]
        ambigious_not_in_annotation_holder=[]
        f_measure_holder=[]


        total_mentions=0

        if(len(good_candidates)>0):
            for idx,val in enumerate(annotations):
                total_mentions+=len(val)
                #print(idx,val,true_positives_candidates[idx])
                false_negative_line= [val2 for val2 in val if val2 not in good_candidates[idx]]
                #print(idx,false_negative_line)
                true_positive_line=[val2 for val2 in val if val2 in good_candidates[idx]]

                # ambigious_not_in_annotation_line= [val2 for val2 in ambigious_candidates[idx] if val2 not in val]

                false_positive_line=[val2 for val2 in good_candidates[idx] if val2 not in val]
                #print(idx,false_positive_line)

                
                # print(idx,true_positive_line,'ground truth: ',annotations[idx],'our system: ',good_candidates[idx])
                
                #print(idx+1,'True positive:',true_positive_line)
                true_positive_count+=len(true_positive_line)
                #print(idx+1,'False positive:',false_positive_line)
                false_positive_count+=len(false_positive_line)
                #print(idx+1,'False negative:',false_negative_line)
                false_negative_count+=len(false_negative_line)
                #print(' ')

                true_positive_holder.append(len(true_positive_line))
                false_negative_holder.append(len(false_negative_line))
                false_positive_holder.append(len(false_positive_line))
                # ambigious_not_in_annotation_holder.append(len(ambigious_not_in_annotation_line))
                total_mention_holder.append(len(val))



            #print(total_mentions, true_positive_count,false_positive_count,false_negative_count)
            # print(false_positive_count)
            # print(false_negative_count)
            precision=(true_positive_count)/(true_positive_count+false_positive_count)
            recall=(true_positive_count)/(true_positive_count+false_negative_count)
            f_measure=2*(precision*recall)/(precision+recall)

            if(state_of_art=="ritter_candidates"):
                self.accuracy_vals_ritter.append((f_measure,precision,recall))    
            if(state_of_art=="stanford_candidates"):
                self.accuracy_vals_stanford.append((f_measure,precision,recall))
            if(state_of_art=="calai_candidates"):
                self.accuracy_vals_opencalai.append((f_measure,precision,recall))                            

            # print('z_score:', z_score_threshold , 'precision: ',precision,'recall: ',recall,'f measure: ',f_measure)
            # print('trupe positive: ',tp_count, 'false positive: ',fp_count,'false negative: ', fn_count,'total mentions: ', tm_count)

            tweet_ids_df["tp"+state_of_art]=true_positive_holder
            tweet_ids_df["fn"+state_of_art]=false_negative_holder
            tweet_ids_df['fp'+state_of_art]= false_positive_holder
            
            if(state_of_art=="ritter_candidates"):
                tweet_ids_df.to_csv("ritter_results.csv", sep=',', encoding='utf-8')
    
            if(state_of_art=="stanford_candidates"):
                tweet_ids_df.to_csv("stanford_results.csv", sep=',', encoding='utf-8')




    def calculate_tp_fp_f1_for_others(self,raw_tweets_for_others):

        opencalai="calai_candidates"
        stanford="stanford_candidates"
        ritter="ritter_candidates"

        self.calculate_tp_fp_f1_generic(raw_tweets_for_others,opencalai)
        self.calculate_tp_fp_f1_generic(raw_tweets_for_others,stanford)
        self.calculate_tp_fp_f1_generic(raw_tweets_for_others,ritter)

    #################################
    #input candidate_feature_Base
    #output candidate_feature_Base with ["Z_score"], ["probability"],["class"]
    # no side effect
    #################################
    def classify_candidate_base(self,z_score_threshold,candidate_featureBase_DF):

        # #filtering test set based on z_score
        mert1=candidate_featureBase_DF['cumulative'].as_matrix()
        #frequency_array = np.array(list(map(lambda val: val[0], sortedCandidateDB.values())))
        zscore_array1=stats.zscore(mert1)

        candidate_featureBase_DF['Z_ScoreUnweighted']=zscore_array1
        candidate_featureBase_DF.to_csv("cf_new_with_z_score.csv", sep=',', encoding='utf-8')

        infrequent_candidates=candidate_featureBase_DF[(candidate_featureBase_DF['Z_ScoreUnweighted'] < z_score_threshold) & (candidate_featureBase_DF.length>1)].candidate.tolist()
        candidate_featureBase_DF = candidate_featureBase_DF[candidate_featureBase_DF['Z_ScoreUnweighted'] > z_score_threshold]



        #returns updated candidate_featureBase_DF with ["Z_score"], ["probability"],["class"] attributes.
        return (self.my_classifier.run(candidate_featureBase_DF,z_score_threshold),infrequent_candidates)


    # recall_correction
    def set_partition_dict(self,candidate_featureBase_DF,infrequent_candidates):
        #print(candidate_featureBase_DF)
        ambiguous_bad_candidates=candidate_featureBase_DF[(((candidate_featureBase_DF.status=="a")|(candidate_featureBase_DF.status=="b"))&(candidate_featureBase_DF.length.astype(int)>1))]
        good_candidates=candidate_featureBase_DF[(candidate_featureBase_DF.status=="g")].candidate.tolist()
        flag1=False
        flag2=False
        if(len(ambiguous_bad_candidates)>0):
            ambiguous_bad_candidates['max_column'] =ambiguous_bad_candidates[['cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative']].idxmax(axis=1) 
            ambiguous_bad_candidates_wFilter= ambiguous_bad_candidates[ambiguous_bad_candidates.max_column=='substring-cap']

            #good_candidates=candidate_featureBase_DF[(candidate_featureBase_DF.status=="g")].candidate.tolist()
            #print(len(ambiguous_bad_candidates_wFilter.candidate.tolist()))

            for candidate in ambiguous_bad_candidates_wFilter.candidate.tolist():
                #print(candidate)
                if candidate not in self.partition_dict.keys():
                    substring_candidates=self.get_substring_candidates(candidate.split(),good_candidates,False)
                    if(len(substring_candidates)>0):
                        self.partition_dict[candidate]=substring_candidates

            flag1= True
        if(len(infrequent_candidates)>0):
            #print(len(ambiguous_bad_candidates_wFilter.candidate.tolist()))

            for candidate in infrequent_candidates:
                #print(candidate)
                if candidate not in self.partition_dict.keys():
                    substring_candidates=self.get_substring_candidates(candidate.split(),good_candidates,False)
                    if(len(substring_candidates)>0):
                        self.partition_dict[candidate]=substring_candidates
            flag2= True
        return (flag1|flag2)


    def set_cb(self,TweetBase,CTrie,phase2stopwordList,z_score_threshold):

        #input new_tweets, z_score, Updated candidatebase of phase1
        #output candidate_featureBase_DF, Incomplete_tweets
        candidate_featureBase_DF,data_frame_holder,phase2_candidates_holder = self.extract(TweetBase,CTrie,phase2stopwordList)
        #set ['probabilities'] for candidate_featureBase_DF
        candidate_featureBase_DF,infrequent_candidates= self.classify_candidate_base(z_score_threshold,candidate_featureBase_DF)

        # set readable labels (a,g,b) for candidate_featureBase_DF based on ['probabilities.']
        candidate_featureBase_DF=self.set_readable_labels(candidate_featureBase_DF)

        candidate_featureBase_DF.to_csv("cb_with_prob_label.csv", sep=',', encoding='utf-8')


        correction_flag=self.set_partition_dict(candidate_featureBase_DF,infrequent_candidates)
        # candidate_featureBase_DF.to_csv("cf_new.csv", sep=',', encoding='utf-8')


        #['probability'],['a,g,b']
        return candidate_featureBase_DF,data_frame_holder,phase2_candidates_holder,correction_flag


        #flush out completed tweets
        # input candidate base, looped over tweets (incomplete tweets+ new tweets)
        # output: incomplete tweets (a tags in it.), incomplete_tweets["Complete"]
    def set_tf(self,data_frame_holder,
            candidate_featureBase_DF,
            phase2_candidates_holder,correction_flag):
        return self.set_completeness_in_tweet_frame(data_frame_holder,
            candidate_featureBase_DF,
            phase2_candidates_holder,correction_flag)

    def get_incomplete_tf(self,untrashed_tweets):
        return untrashed_tweets[untrashed_tweets.completeness==False]

    def get_complete_tf(self,untrashed_tweets):
        return untrashed_tweets[untrashed_tweets.completeness==True]

    def compute_seen_tweets_so_far(self,start_batch,end_batch):
        if(start_batch==end_batch):
            sliced_seen_tweets=self.number_of_seen_tweets_per_batch[start_batch]


        sliced_seen_tweets=self.number_of_seen_tweets_per_batch[start_batch:]


        counter=0
        for elem in sliced_seen_tweets:
            counter=counter+elem

        return counter

    #@profile
    def rreplace(self,s, old, new, occurrence):
        if s.endswith(old):
            li = s.rsplit(old, occurrence)
            return new.join(li)
        else:
            return s
    #ME_EXTR=Mention.Mention_Extraction()


    # experiment function
    def set_x_axis(self,just_converted_tweets_for_current_batch):

        self.incomplete_tweets.to_csv("set_x_axis_debug.csv", sep=',', encoding='utf-8')

        self.incomplete_tweets['number_of_seen_tweets'] = self.incomplete_tweets['entry_batch'].apply(lambda x: self.compute_seen_tweets_so_far(x,self.counter))


        self.incomplete_tweets["entry_vs_tweet_seen_ratio"]=self.incomplete_tweets['entry_batch']/self.incomplete_tweets['number_of_seen_tweets']


        #counter_list= 
        self.incomplete_tweets["ratio_entry_vs_current"]=self.incomplete_tweets['entry_batch']/self.counter


        self.incomplete_tweets["current_minus_entry"]=self.counter-self.incomplete_tweets['entry_batch']

        just_converted_tweets_for_current_batch["current_minus_entry"]=self.counter-just_converted_tweets_for_current_batch['entry_batch']

        return just_converted_tweets_for_current_batch



    def set_column_for_candidates_in_incomplete_tweets(self,candidate_featureBase_DF,input_to_eval):
        incomplete_candidates= input_to_eval['2nd Iteration Candidates'].tolist()


        candidate_featureBase_DF= candidate_featureBase_DF.set_index('candidate')

        candidate_with_label_holder=[]
        one_level=[]


        for sentence_level_candidates in incomplete_candidates:
            one_level.clear()
            for candidate in sentence_level_candidates:
                if candidate in candidate_featureBase_DF.index:
                    label=candidate_featureBase_DF.get_value(candidate,'status')
                    one_level.append((candidate,label))
                else:
                    one_level.append((candidate,"na"))


            candidate_with_label_holder.append(copy.deepcopy(one_level))


        input_to_eval["candidates_with_label"]=candidate_with_label_holder
        debug_candidates_label_list= input_to_eval['candidates_with_label'].tolist()
        candidates_filtered_g_labeled=[]
        row_level_candidates=[]

        candidates_filtered_a_labeled=[]
        row_level_a_candidates=[]

        for sentence_level in debug_candidates_label_list:
            row_level_candidates.clear()
            row_level_a_candidates.clear()
            for candidate in sentence_level:
                if(candidate[1]=="g"):
                        row_level_candidates.append(candidate[0])

                if(candidate[1]=="a"):
                        row_level_a_candidates.append(candidate[0])
            candidates_filtered_g_labeled.append(copy.deepcopy(row_level_candidates))
            candidates_filtered_a_labeled.append(copy.deepcopy(row_level_a_candidates))





        input_to_eval["only_good_candidates"]=candidates_filtered_g_labeled
        input_to_eval["ambigious_candidates"]=candidates_filtered_a_labeled



    def calculate_tp_fp_f1(self,z_score_threshold,input_to_eval):



        good_candidates = input_to_eval['only_good_candidates'].tolist()

        annotations= input_to_eval['annotation'].tolist()

        ambigious_candidates=input_to_eval['ambigious_candidates'].tolist()


        true_positive_count=0
        false_positive_count=0
        false_negative_count=0
        ambigious_not_in_annotation=0

        true_positive_holder = []
        false_negative_holder=[]
        false_positive_holder=[]
        total_mention_holder=[]
        ambigious_not_in_annotation_holder=[]
        f_measure_holder=[]


        total_mentions=0

        if(len(good_candidates)>0):
            for idx,val in enumerate(annotations):
                total_mentions+=len(val)
                #print(idx,val,true_positives_candidates[idx])
                false_negative_line= [val2 for val2 in val if val2 not in good_candidates[idx]]
                #print(idx,false_negative_line)
                true_positive_line=[val2 for val2 in val if val2 in good_candidates[idx]]

                ambigious_not_in_annotation_line= [val2 for val2 in ambigious_candidates[idx] if val2 not in val]

                false_positive_line=[val2 for val2 in good_candidates[idx] if val2 not in val]
                #print(idx,false_positive_line)

                
                # print(idx,true_positive_line,'ground truth: ',annotations[idx],'our system: ',good_candidates[idx])
                
                #print(idx+1,'True positive:',true_positive_line)
                true_positive_count+=len(true_positive_line)
                #print(idx+1,'False positive:',false_positive_line)
                false_positive_count+=len(false_positive_line)
                #print(idx+1,'False negative:',false_negative_line)
                false_negative_count+=len(false_negative_line)
                #print(' ')

                true_positive_holder.append(len(true_positive_line))
                false_negative_holder.append(len(false_negative_line))
                false_positive_holder.append(len(false_positive_line))
                ambigious_not_in_annotation_holder.append(len(ambigious_not_in_annotation_line))
                total_mention_holder.append(len(val))



            #print(total_mentions, true_positive_count,false_positive_count,false_negative_count)
            # print(false_positive_count)
            # print(false_negative_count)
            precision=(true_positive_count)/(true_positive_count+false_positive_count)
            recall=(true_positive_count)/(true_positive_count+false_negative_count)
            f_measure=2*(precision*recall)/(precision+recall)

            true_positive_count_IPQ=true_positive_count
            false_positive_count_IPQ = false_positive_count
            false_negative_count_IPQ= false_negative_count
            total_mention_count_IPQ=total_mentions


            tp_count=0
            tm_count=0
            fp_count=0
            fn_count=0

            for idx,tup in enumerate(self.accuracy_tuples_prev_batch):
                # print(idx,tup)
                tp_count+=tup[0]
                tm_count+=tup[1]
                fp_count+=tup[2]
                fn_count+=tup[3]


            tp_count+=true_positive_count_IPQ
            tm_count+=total_mention_count_IPQ
            fp_count+=false_positive_count_IPQ
            fn_count+=false_negative_count_IPQ

            precision=(tp_count)/(tp_count+fp_count)
            recall=(tp_count)/(tp_count+fn_count)
            f_measure=2*(precision*recall)/(precision+recall)



            self.accuracy_vals=(f_measure,z_score_threshold,precision,recall)

            # print('z_score:', z_score_threshold , 'precision: ',precision,'recall: ',recall,'f measure: ',f_measure)
            # print('trupe positive: ',tp_count, 'false positive: ',fp_count,'false negative: ', fn_count,'total mentions: ', tm_count)

            input_to_eval["tp"]=true_positive_holder
            input_to_eval["fn"]=false_negative_holder
            input_to_eval['fp']= false_positive_holder
            input_to_eval["total_mention"]=total_mention_holder

            # input_to_eval["ambigious_not_in_annot"]=ambigious_not_in_annotation_holder
            # input_to_eval["inverted_loss"]=input_to_eval["tp"]/( input_to_eval["fn"]+input_to_eval["ambigious_not_in_annot"])

        return input_to_eval


    def recall_correction(self,phase2_candidates_holder,data_frame_holder):
        corrected_phase2_candidates_holder=[]

        for candidates in phase2_candidates_holder:
            corrected_phase2_candidates=[]
            for idx, candidate in enumerate(candidates):
                if(candidate in self.partition_dict.keys()):
                    #print(candidate, self.partition_dict[candidate])
                    corrected_phase2_candidates.extend(self.partition_dict[candidate])
                else:
                    corrected_phase2_candidates.append(candidate)
            corrected_phase2_candidates_holder.append(copy.deepcopy(corrected_phase2_candidates))

        
        #print(corrected_phase2_candidates_holder)
        data_frame_holder['2nd Iteration Candidates']=corrected_phase2_candidates_holder

        return corrected_phase2_candidates_holder,data_frame_holder                  



    #@profile
    def set_completeness_in_tweet_frame(self,data_frame_holder,candidate_featureBase_DF,phase2_candidates_holder,correction_flag):
        #print(candidate_featureBase_DF.head())

        good_candidates=candidate_featureBase_DF[candidate_featureBase_DF.status=="g"].candidate.tolist()
        bad_candidates=candidate_featureBase_DF[candidate_featureBase_DF.status=="b"].candidate.tolist()

        merged_g_b= bad_candidates+good_candidates

        candidate_featureBase_DF.to_csv("cf_before_labeling_comp.csv", sep=',', encoding='utf-8')
        ambiguous_candidates=candidate_featureBase_DF[candidate_featureBase_DF.status=="a"].candidate.tolist()

        if(correction_flag):
            phase2_candidates_holder,data_frame_holder=self.recall_correction(phase2_candidates_holder,data_frame_holder)

        
        truth_vals=[False if any(x not in merged_g_b for x in list1) else True for list1 in phase2_candidates_holder]


        # truth_vals=[False if any(x in ambiguous_candidates for x in list1) else True for list1 in phase2_candidates_holder]

        # for list1 in phase2_candidates_holder:
        #     if any(x in ambiguous_candidates  for x in list1):
        #         truth_vals.append(False)
        #     else:
        #         truth_vals.append(True)



        #print(truth_vals)
        completeness_series = pd.Series( (v for v in truth_vals) )

        data_frame_holder['completeness']=completeness_series
        data_frame_holder["current_minus_entry"]=self.counter-data_frame_holder['entry_batch']

        data_frame_holder.to_csv("phase2output_with_completeness.csv", sep=',', encoding='utf-8')

        return data_frame_holder



    #@profile
    def set_readable_labels(self,candidate_featureBase_DF):

        #candidate_featureBase_DF['status'] = candidate_featureBase_DF['probability'].apply(lambda x: set(x).issubset(good_candidates))
        candidate_featureBase_DF['status']='ne'
        candidate_featureBase_DF['status'][candidate_featureBase_DF['probability']>=0.75]='g'
        candidate_featureBase_DF['status'][(candidate_featureBase_DF['probability'] > 0.4) & (candidate_featureBase_DF['probability'] < 0.75)] = 'a'
        candidate_featureBase_DF['status'][candidate_featureBase_DF['probability']<=0.4]='b'

        return candidate_featureBase_DF


    #@profile
    def normalize(self,word):
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
        #return strip_op

    #@profile      
    def isSubstring(self,to_increase_element,id_to_incr,comparison_holder,phase1_holder_holder_copy):
        combined_list=comparison_holder[id_to_incr]+phase1_holder_holder_copy[id_to_incr]

        for idx,val in enumerate(comparison_holder[id_to_incr]):
            if((to_increase_element[0] in val[0]) and to_increase_element[0] != val[0]):
                if((to_increase_element[5] in val[5]) and to_increase_element[5] != val[5]):
                    return True
        for idx,val in enumerate(phase1_holder_holder_copy[id_to_incr]):
            if((to_increase_element[0] in val[0]) and to_increase_element[0] != val[0]):
                if((to_increase_element[5] in val[2]) and to_increase_element[5] != val[2]):
                    return True   
                
        return False

    #@profile
    def calculate_pmi(self,big,x1,x2,total):
        big__= float(big/total)
        x1__=float(x1/total)
        x2__=float(x2/total)
        pmi= math.log(big__/(x1__*x2__),2.71828182845)
        pklv=big__*pmi
        #return (1/(1+math.exp(-1*pmi)))
        npmi= pmi/(-1.0*(math.log(big__,2.71828182845)))
        return npmi,pklv
        #return pklv


    def get_substring_candidates(self,candidate_words,good_candidates,whole_check_flag):
        substring_candidates=[]
        last_cand=""
        break_flag=False
        start=0
        for i in range(len(candidate_words)):
            curr=' '.join(candidate_words[0:(i+1)])
            if curr in good_candidates:
                #print("got: ",curr)
                last_cand=curr
            else:
                if i==0:
                    start=i+1
                else:
                    start=i
                    if(last_cand!=""):
                        substring_candidates.append(last_cand)
                    else:
                        substring_candidates.extend(self.get_substring_candidates(candidate_words[0:(i+1)],good_candidates,True))
                break_flag=True
                break
        if(break_flag & (len(candidate_words[start:])>0)):
            substring_candidates.extend(self.get_substring_candidates(candidate_words[start:],good_candidates,True))
        if(whole_check_flag & (not break_flag) & (last_cand!="")):
            substring_candidates.append(last_cand)

        return substring_candidates
    
    #@profile
    def verify(self, subsequence, CTrie):
        return CTrie.__contains__(subsequence)


    #@profile
    def check_sequence(self, sequence, l, CTrie):
        result=[]
        subsequence_length=l
        while(subsequence_length>0):
            shift=len(sequence)-subsequence_length
            verified_subsequence=[]
            verified=False
            for i in range(0,shift+1):
                list1=sequence[i:(i+subsequence_length)]
                text=' '.join(str(e[0]) for e in list1)
                subsequence=(self.normalize(text)).split()
                #print("search for", subsequence)
                if self.verify(subsequence, CTrie):
                    verified_subsequence.append(i)
                    verified_subsequence.append(i+subsequence_length)
                    #print(subsequence)
                    #print(subsequence,[(verified_subsequence[0]-0),(int(sequence[-1][1])-verified_subsequence[1])])
                    verified=True
                    break
            if(verified):
                result.append(sequence[verified_subsequence[0]:verified_subsequence[1]])
                if(verified_subsequence[0]-0)>0:
                    subequence_to_check=sequence[0:verified_subsequence[0]]
                    #since tokens before the starting position of the verified subsequence have already been checked for subsequences of this length
                    partition_length=min(len(subequence_to_check),(subsequence_length-1))
                    #print(subequence_to_check)
                    lst=self.check_sequence(subequence_to_check,partition_length, CTrie)
                    if(lst):
                        result.extend(lst)
                if(int(sequence[-1][1])-verified_subsequence[1])>0:
                    subequence_to_check=sequence[(verified_subsequence[1]):]
                    #since tokens following the end position of the verified subsequence have not been checked for subsequences of this length
                    partition_length=min(len(subequence_to_check),(subsequence_length))
                    #print(subequence_to_check)
                    lst=self.check_sequence(subequence_to_check,partition_length, CTrie)
                    if(lst):
                        result.extend(lst)
                return result
            else:
                subsequence_length-=1
        return result

    def flatten(self,mylist, outlist,ignore_types=(str, bytes, int, ne.NE_candidate)):
    
        if mylist !=[]:
            for item in mylist:
                #print not isinstance(item, ne.NE_candidate)
                if isinstance(item, list) and not isinstance(item, ignore_types):
                    self.flatten(item, outlist)
                else:
                    if isinstance(item,ne.NE_candidate):
                        item.phraseText=item.phraseText.strip(' \t\n\r')
                        item.reset_length()
                    else:
                        if type(item)!= int:
                            item=item.strip(' \t\n\r')
                    outlist.append(item)
        return outlist


    def getWords(self, sentence):
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
        tweetWordList=self.flatten(tempList,[])
        return tweetWordList


    #@profile
    # def get_Candidates(self, sequence, CTrie,flag):
    #     #print(sequence)
    #    #print(sequence)
    #     candidateList=[]
    #     left=0
    #     start_node=CTrie
    #     last_cand="NAN"
    #     last_cand_substr=""
    #     reset=False
    #     for right in range(len(sequence)):
    #         if(reset):
    #             start_node=CTrie
    #             last_cand_substr=""
    #             left=right
    #         curr_text=sequence[right][0]
    #         curr_pos=[sequence[right][1]]
    #         curr=self.normalize(sequence[right][0])
    #         cand_str=self.normalize(last_cand_substr+" "+curr)
    #         last_cand_sequence=sequence[left:(right+1)]
    #         last_cand_text=' '.join(str(e[0]) for e in last_cand_sequence)
    #         last_cand_text_norm=self.normalize(' '.join(str(e[0]) for e in last_cand_sequence))
    #         #print("==>",cand_str,last_cand_text)
    #         if ((curr in start_node.path.keys())&(cand_str==last_cand_text_norm)):
    #             #if flag:
    #             #print("=>",cand_str,last_cand_text)
    #             reset=False
    #             if (start_node.path[curr].value_valid):
    #                 #print(last_cand_text)
    #                 # if flag:
    #                 #     print(last_cand_text)
    #                 last_cand_pos=[e[1] for e in last_cand_sequence]
    #                 last_cand=last_cand_text
    #                 last_cand_batch=start_node.path[curr].feature_list[-1]
    #             start_node=start_node.path[curr]
    #             last_cand_substr=cand_str
    #         else:
    #             #print("=>",cand_str,last_cand_text)
    #             if(last_cand!="NAN"):
    #                 candidateList.append((last_cand,last_cand_pos,last_cand_batch))
    #                 last_cand="NAN"
    #                 if(start_node!=CTrie):
    #                     start_node=CTrie
    #                     last_cand_substr=""
    #                     if curr in start_node.path.keys():
    #                         #print("here",curr)
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
    #                 candidateList.extend(self.get_Candidates(sequence[(left+1):(right+1)], CTrie, flag))
    #                 reset=True
    #     #print(last_cand)
    #     if(last_cand!="NAN"):
    #         candidateList.append((last_cand,last_cand_pos,last_cand_batch))
    #     return candidateList

    def get_Candidates(self, sequence, CTrie,flag):
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
            curr=self.normalize(sequence[right][0])
            cand_str=self.normalize(last_cand_substr+" "+curr)
            last_cand_sequence=sequence[left:(right+1)]
            last_cand_text=' '.join(str(e[0]) for e in last_cand_sequence)
            last_cand_text_norm=self.normalize(' '.join(str(e[0]) for e in last_cand_sequence))
            #print("==>",cand_str,last_cand_text)
            if ((curr in start_node.path.keys())&(cand_str==last_cand_text_norm)):
                # if flag:
                #     print("=>",cand_str,last_cand_text)
                reset=False
                if (start_node.path[curr].value_valid):
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
                        #candidateList.extend(self.get_Candidates(sequence[(left+1):(right+1)], CTrie, flag))
                        right=left
                        # if(flag):
                        #     print("++",right)
                    reset=True
            right+=1
        # if(flag):
        #     print(last_cand)
        if(last_cand!="NAN"):
            candidateList.append((last_cand,last_cand_pos,last_cand_batch))
        return candidateList


        # candidateList=[]
        # left=0
        # start_node=CTrie
        # last_cand="NAN"
        # last_cand_substr=""
        # reset=False
        # for right in range(len(sequence)):
        #     if(reset):
        #         left=right
        #     curr_text=sequence[right][0]
        #     curr_pos=[sequence[right][1]]
        #     curr=self.normalize(sequence[right][0])
        #     cand_str=self.normalize(last_cand_substr+" "+curr)
        #     last_cand_sequence=sequence[left:(right+1)]
        #     last_cand_text=' '.join(str(e[0]) for e in last_cand_sequence)
        #     last_cand_text_norm=self.normalize(' '.join(str(e[0]) for e in last_cand_sequence))
        #     #print("==>",cand_str,last_cand_text)
        #     if ((curr in start_node.path.keys())&(cand_str==last_cand_text_norm)):
        #         #if flag:
        #             #print("=>",cand_str,last_cand_text)
        #         reset=False
        #         if (start_node.path[curr].value_valid):
        #             #print(last_cand_text)
        #             # if flag:
        #             #     print(last_cand_text)
        #             last_cand_pos=[e[1] for e in last_cand_sequence]
        #             last_cand=last_cand_text
        #         start_node=start_node.path[curr]
        #         last_cand_substr=cand_str
        #     else:
        #         #print("=>",cand_str,last_cand_text)
        #         if(last_cand!="NAN"):
        #             candidateList.append((last_cand,last_cand_pos))
        #             last_cand="NAN"
        #             if(start_node!=CTrie):
        #                 start_node=CTrie
        #                 last_cand_substr=""
        #                 if curr in start_node.path.keys():
        #                     #print("here",curr)
        #                     reset=False
        #                     if start_node.path[curr].value_valid:
        #                         last_cand_text=curr_text
        #                         last_cand_pos=curr_pos
        #                         last_cand=curr
        #                     left=right
        #                     start_node=start_node.path[curr]
        #                     last_cand_substr=curr
        #                 else:
        #                     reset=True
        #             else:
        #                 reset=True
        #         else:
        #             candidateList.extend(self.get_Candidates(sequence[(left+1):(right+1)], CTrie, flag))
        #             reset=True
        # #print(last_cand)
        # if(last_cand!="NAN"):
        #     candidateList.append((last_cand,last_cand_pos))
        # return candidateList

    #@profile
    def append_rows(self,df_holder):
    
        df = pd.DataFrame(df_holder)
        #self.data_frame_holder=self.data_frame_holder.append(df,ignore_index=True)
        #self.data_frame_holder=self.data_frame_holder.reset_index(drop=True)
        return df



    #@profile
    def join_token_tuples(self,list_of_tuples):
        #print(string.punctuation)
        combined_str=(' '.join(tuple[0] for tuple in list_of_tuples)).lstrip(string.punctuation).rstrip(string.punctuation).strip()
        combined_pos='*'.join(str(tuple[1]) for tuple in list_of_tuples)
        combined_tuple=(combined_str,combined_pos,list_of_tuples[0][2],list_of_tuples[0][3],list_of_tuples[0][4],list_of_tuples[0][5],list_of_tuples[0][6])
        return combined_tuple



    #@profile
    def all_capitalized(self,candidate):
        strip_op=candidate
        strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip())
        strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
        strip_op= self.rreplace(self.rreplace(self.rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
        prep_article_list=prep_list+article_list+self.phase2stopwordList
        word_list=strip_op.split()
        for i in range(len(word_list)):
            word=word_list[i]
            if(word[0].isupper()):
                continue
            else:
                if(word in prep_article_list):
                    if (i!=0):
                        continue
                    else:
                        False
                else:
                    return False
        return True



    #@profile
    def check_feature_update(self, candidate_tuple,non_discriminative_flag):
        #print(candidate_tuple)
        if(non_discriminative_flag):
            return 7
        candidateText=candidate_tuple[0]
        position=candidate_tuple[1]
        word_list=candidateText.split()
        if candidateText.islower():
            return 6
        elif candidateText.isupper():
            return 5
        elif (len(word_list)==1):
            #start-of-sentence-check
            if self.all_capitalized(candidateText):
                if(int(position[0])==0):
                    return 4
                else:
                    return 2
            else:
                return 3
        else:
            if(self.all_capitalized(candidateText)):
                return 2
            else:
                return 3

    #@profile
    def update_Candidatedict(self,candidate_tuple,non_discriminative_flag):
        candidateText=candidate_tuple[0]
        #print(candidate_tuple)
        normalized_candidate=self.normalize(candidateText)
        feature_list=[]
        if(normalized_candidate in self.CandidateBase_dict.keys()):
            feature_list=self.CandidateBase_dict[normalized_candidate]
        else:
            feature_list=[0]*9
            feature_list[0]=self.counter
            feature_list[1]=len(normalized_candidate.split())
        feature_to_update=self.check_feature_update(candidate_tuple,non_discriminative_flag)
        #print(candidateText,feature_to_update)
        feature_list[feature_to_update]+=1
        feature_list[8]+=1
        self.CandidateBase_dict[normalized_candidate]=feature_list




    #@profile
    def extract(self,TweetBase,CTrie,phase2stopwordList):


        if(self.counter==0):
            #output_queue
            self.data_frame_holder_OQ=pd.DataFrame([], columns=['index', 'entry_batch', 'tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates','annotation','stanford_candidates'])
            self.incomplete_tweets=pd.DataFrame([], columns=['index','entry_batch', 'tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates','annotation','stanford_candidates'])
            self.CandidateBase_dict= {}
            self.partition_dict={}
            self.good_candidates=[]
            self.bad_candidates=[]
            self.aggregator_incomplete_tweets=pd.DataFrame([], columns=['index', 'entry_batch', 'tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates','annotation','stanford_candidates'])
            self.just_converted_tweets=pd.DataFrame([], columns=['index', 'entry_batch', 'tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates','annotation','stanford_candidates'])
            self.data_frame_holder=pd.DataFrame([], columns=['index','entry_batch','tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates'])
            self.raw_tweets_for_others=pd.DataFrame([], columns=['index','entry_batch','tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates'])

            self.accuracy_tuples_prev_batch=[]
            self.accuracy_vals=[]
            
            #### other systems
            self.accuracy_vals_stanford=[]
            self.accuracy_vals_opencalai=[]
            self.accuracy_vals_ritter=[]

            self.number_of_seen_tweets_per_batch=[]
        self.phase2stopwordList=phase2stopwordList
        self.number_of_seen_tweets_per_batch.append(len(TweetBase))


        data_frame_holder=pd.DataFrame([], columns=['index','entry_batch','tweetID', 'sentID', 'hashtags', 'user', 'TweetSentence','phase1Candidates', '2nd Iteration Candidates'])
        phase1_holder_holder=[]
        phase2_candidates_holder=[]
        df_holder=[]
        candidateBase_holder=[]

        #this has to be changed to an append function since IPQ already has incomplete tweets from prev batch  
        immediate_processingQueue = pd.concat([self.incomplete_tweets,TweetBase ])
        immediate_processingQueue.to_csv("impq.csv", sep=',', encoding='utf-8')
        


        print('In Phase 2',len(immediate_processingQueue))
        #immediate_processingQueue=immediate_processingQueue.reset_index(drop=True)
        combined_list_here=([]+list(cachedStopWords)+chat_word_list+day_list+month_list+article_list+prep_list)
        combined_list_filtered=list(filter(lambda word: word not in (prep_list+article_list+month_list+phase2stopwordList), combined_list_here))
        #--------------------------------------PHASE II---------------------------------------------------
        for index, row in immediate_processingQueue.iterrows():

            #phase 1 candidates for one sentence
            phase1_holder=[]

            tweetText=str(row['TweetSentence'])
            #print(tweetText)
            sentID=str(row['sentID'])
            tweetID=str(row['tweetID'])
            phase1Candidates=str(row['phase1Candidates'])
            hashtags=str(row['hashtags'])
            user=str(row['user'])
            batch=int(row['entry_batch'])
            time=str(row['start_time'])



            annotation=list(row['annotation'])

            stanford=list(row['stanford_candidates'])
            non_discriminative_flag=False


            if(phase1Candidates!="nan"):
                phase1Raw=phase1Candidates.split("||")
                phase1Raw = list(filter(None, phase1Raw))


                for entities_with_loc in phase1Raw:
                    entity_to_store=entities_with_loc.split("::")[0]
                    #print(entity_to_store)
                    position=entities_with_loc.split("::")[1]
                    #print(position)
                    phase1_holder.append((entity_to_store,position))

                phase1_holder_holder.append(copy.deepcopy(phase1_holder))
                phase1_holder.clear()

            else:
                non_discriminative_flag=True
                phase1_holder_holder.append([])


            #print(sen_index1)[ ()/,;:!?…-]
            #splitList=tweetText.split()
            '''splitList=re.split('[ ()/,;:!?…-]',tweetText)
            #print(tweetText,splitList)
            #wordlstU=list(filter(lambda word: ((word!="")&(word.strip(string.punctuation).strip().lower() not in cachedStopWords)), splitList))
            splitList=list(map(lambda word: word.strip(), splitList))
            tweetWordList=list(filter(lambda word: word!="", splitList))'''
            #print(tweetWordList)
            tweetWordList=self.getWords(tweetText)
            tweetWordList= [(token,idx) for idx,token in enumerate(tweetWordList)]
            #print(tweetWordList)


            #combined_list_here=([]+list(cachedStopWords)+prep_list+chat_word_list+article_list+day_list+month_list)
            
            tweetWordList_stopWords=list(filter (lambda word: ((((word[0].strip()).strip(string.punctuation)).lower() in combined_list_filtered)|(word[0].strip() in string.punctuation)|(word[0].startswith('@'))), tweetWordList))


            # phase 2 candidate tuples without stopwords for a sentence
            c=[(y[0],str(y[1]),tweetID,sentID,'ne',batch,time) for y  in tweetWordList if y not in tweetWordList_stopWords ]
            #c=[(y[0],str(y[1])) for y  in tweetWordList if y not in tweetWordList_stopWords ]

            
            sequences=[]
            for k, g in groupby(enumerate(c), lambda element: element[0]-int(element[1][1])):
                sequences.append(list(map(itemgetter(1), g)))

            ne_candidate_list=[]
            for sequence in sequences:
                '''print(sequence)
                seq_candidates=self.check_sequence(sequence, len(sequence), CTrie)'''
                #print([y[0] for y in sequence])
                seq_candidate_list=self.get_Candidates(sequence, CTrie,False)
                if(seq_candidate_list):
                    '''seq_candidate_list= list(map(lambda e: self.join_token_tuples(e) ,seq_candidates))
                    print("====",seq_candidate_list)'''

                    
                    for candidate_tuple in seq_candidate_list:
                        #inserts into CandidateBase and updates the correct frequency feature based on Capitalization pattern
                        if not ((float(batch)<self.counter)&(candidate_tuple[-1]<self.counter)):
                        #print(candidate_tuple[0])
                            self.update_Candidatedict(candidate_tuple,non_discriminative_flag)
                    ne_candidate_list.extend(seq_candidate_list)
            
            
            #phase2_candidates='||'.join(e[0] for e in ne_candidate_list)

            phase2_candidates=[self.normalize(e[0]) for e in ne_candidate_list]
            phase2_candidates_holder.append(phase2_candidates)

            #print(phase1Candidates,"====",phase2_candidates)
            dict1 = {'entry_batch':batch, 'tweetID':tweetID, 'sentID':sentID, 'hashtags':hashtags, 'user':user, 'TweetSentence':tweetText, 'phase1Candidates':phase1Candidates,'2nd Iteration Candidates':phase2_candidates,'annotation':annotation,'stanford_candidates':stanford}

            df_holder.append(dict1)
            #-------------------------------------------------------------------END of 1st iteration: RESCAN+CANDIDATE_UPDATION-----------------------------------------------------------

        #df_holder is the immediate processing queue of the current batch converted into a dataframe---> data_frame_holder
        #self.append_rows(df_holder)
        data_frame_holder = pd.DataFrame(df_holder)
        # print(data_frame_holder.head(5))


        #convert the CandidateFeatureBase from a dictionary to dataframe---> CandidateFeatureBaseDF
        candidateBaseHeaders=['candidate', 'batch', 'length','cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative','cumulative']
        candidate_featureBase_DF=pd.DataFrame.from_dict(self.CandidateBase_dict, orient='index')
        candidate_featureBase_DF.columns=candidateBaseHeaders[1:]
        candidate_featureBase_DF.index.name=candidateBaseHeaders[0]
        candidate_featureBase_DF = candidate_featureBase_DF.reset_index(drop=False)


        data_frame_holder.to_csv("phase2output.csv", sep=',', encoding='utf-8')
        return candidate_featureBase_DF,data_frame_holder,phase2_candidates_holder


        # self.aggregator_incomplete_tweets= self.aggregator_incomplete_tweets.append(self.incomplete_tweets)
        # self.just_converted_tweets=self.just_converted_tweets.append(just_converted_tweets_for_current_batch)






    def finish(self):
        return self.accuracy_vals

    def finish_other_systems(self):
        print("*****************************************STANFORD RESULSTSSS***********************")
        for i in self.accuracy_vals_stanford:
            print(i)

        print("*****************************************STANFORD ENDSSSSSSSS***********************")

        return (self.accuracy_vals_stanford,self.accuracy_vals_opencalai,self.accuracy_vals_ritter)


##################UNCOMMENT THIS WHEN YOU'RE DONE // STARTS

        # candidate_featureBase_DF = candidate_featureBase_DF.set_index('candidate')

        # candidate_with_label_holder=[]
        # one_level=[]
        # for sentence_level_candidates in phase2_candidates_holder:
        #     one_level.clear()
        #     for candidate in sentence_level_candidates:
        #         if candidate in candidate_featureBase_DF.index:
        #             label=candidate_featureBase_DF.get_value(candidate,'status')
        #             one_level.append((candidate,label))
        #         else:
        #             one_level.append((candidate,"na"))


        #     candidate_with_label_holder.append(copy.deepcopy(one_level))

        # print(len(data_frame_holder),len(candidate_with_label_holder))

        # data_frame_holder["candidates_with_label"]=candidate_with_label_holder


############################ UNCOMMENT THIS WHEN YOU DONE FINISH



       # print(data_frame_holder["2nd Iteration Candidates"][data_frame_holder.tweetID=='2'].tolist())
        # list1=data_frame_holder["candidates_with_label"][data_frame_holder.tweetID=='2'].tolist()
  
        # list3=[]
        # #print(list1)
        # for i in list1:
        #     for a in i:
        #        # print(a)
        #         list3.append(a)








        # tweetID_holder=data_frame_holder.tweetID.astype(int) 

        # # row_level_candidates=[]
        # # tweet_level_candidates=[]
        # # for i in range(int(tweetID_holder.max())+1):
        # #     list1=data_frame_holder["candidates_with_label"][data_frame_holder.tweetID==str(i)].tolist()

        #     row_level_candidates.clear()
        #     for j in list1:
        #         for a in j:
        #            # print(a)
        #             if(a[1]=="g"):
        #                 row_level_candidates.append(a[0])

        #     tweet_level_candidates.append(copy.deepcopy(row_level_candidates))



        # for i in tweet_level_candidates:
        #     print(i)













