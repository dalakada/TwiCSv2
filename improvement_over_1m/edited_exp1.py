        ###### EXPERIMENT ONLY
        tweets_annot=pd.read_csv("tweets_500.csv",sep =',')
        #tweets_annot=tweets_annot[counter:counter*100:]


        annot_tweet_level_holder=[]
        holder_2_times=[]
        for index, row in tweets_annot.iterrows():
            annot_raw=str(row['mentions_other'])

            split_list=annot_raw.split(";")
            #split_listFilter=list(filter(lambda element: element.strip()!='', split_list))
            split_listFilter=list(filter(None, split_list))


            # filtered_2_times=list(map(lambda element: list(filter(None, element.split(','))), split_listFilter))
            filtered_2_times=list(map(lambda element: list(filter(None, element.split(','))), split_list))
            # print(filtered_2_times)

            #print(filtered_2_times)
            holder_2_times.append(filtered_2_times)


        times_2_holder=[]
        total_mentions=0
        line_holder=[]
        for idx,val in enumerate(holder_2_times):
            for idy, val2 in enumerate(val):
                if(type(val2)==list):
                    # max="0"
                     for idz,val3 in enumerate(val2):
                    #     if(len(max[0])<=len(val3[0])):
                    #         max=val3
                        
                        line_holder.append(self.normalize(val3))
                else:
                    line_holder.append(val2)
            times_2_holder.append(copy.deepcopy(line_holder))
            line_holder.clear()

        for i in range(len(times_2_holder)):
            if not times_2_holder[i]:
                times_2_holder[i]=[]
            elif (times_2_holder[i]==['nan']):
                times_2_holder[i]=[]


        # for i in times_2_holder:
        #     print(i)

        # print(len(times_2_holder))


        ## tweet_level_candidates = ours
        ## times_2_holder = annot

        true_positives_candidates=tweet_level_candidates

        true_positive_count=0
        false_positive_count=0
        false_negative_count=0


        true_positive_holder = []
        false_negative_holder=[]
        total_mention_holder=[]

        for idx,val in enumerate(times_2_holder):
            total_mentions+=len(val)
            #print(idx,val,true_positives_candidates[idx])
            false_negative_line= [val2 for val2 in val if val2 not in true_positives_candidates[idx]]
            #print(idx,false_negative_line)
            true_positive_line=[val2 for val2 in val if val2 in true_positives_candidates[idx]]


            false_positive_line=[val2 for val2 in true_positives_candidates[idx] if val2 not in val]
            #print(idx,false_positive_line)
            # print(idx,false_positive_line,'ground truth: ',times_2_holder[idx],'our system: ',true_positives_candidates[idx])
            
            #print(idx+1,'True positive:',true_positive_line)
            true_positive_count+=len(true_positive_line)
            #print(idx+1,'False positive:',false_positive_line)
            false_positive_count+=len(false_positive_line)
            #print(idx+1,'False negative:',false_negative_line)
            false_negative_count+=len(false_negative_line)
            #print(' ')

            true_positive_holder.append(len(true_positive_line))
            false_negative_holder.append(len(false_negative_line))
            total_mention_holder.append(len(val))




        # print(total_mentions, true_positive_count,false_positive_count,false_negative_count)
        # print(false_positive_count)
        # print(false_negative_count)
        precision=(true_positive_count)/(true_positive_count+false_positive_count)
        recall=(true_positive_count)/(true_positive_count+false_negative_count)
        f_measure=2*(precision*recall)/(precision+recall)
        # print('precision: ',precision,'recall: ',recall,'f measure: ',f_measure)

        data_frame_holder["tp"]=true_positive_holder
        data_frame_holder["fn"]=false_negative_holder
        data_frame_holder["total_mention"]=total_mention_holder

