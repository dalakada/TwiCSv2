import itertools
import string
import re


# good_candidates=['bill','de blasio','new york','bill de blasio']
# good_candidates=['de blasio','new york']

# def normalize(word):
#     strip_op=word
#     strip_op=(((strip_op.lstrip(string.punctuation)).rstrip(string.punctuation)).strip()).lower()
#     strip_op=(strip_op.lstrip('“‘’”')).rstrip('“‘’”')
#     #strip_op= self.rreplace(self.rreplace(self.rreplace(strip_op,"'s","",1),"’s","",1),"’s","",1)
#     if strip_op.endswith("'s"):
#         li = strip_op.rsplit("'s", 1)
#         return ''.join(li)
#     elif strip_op.endswith("’s"):
#         li = strip_op.rsplit("’s", 1)
#         return ''.join(li)
#     else:
#         return strip_op

# def multiSlice(s,cutpoints):
#     k = len(cutpoints)
#     multislices=[]
#     if k == 0:
#         curr_candidate=normalize(' '.join(s))

#         if(curr_candidate in good_candidates):
#             multislices = [curr_candidate]        
#     else:
        
#         curr_candidate=normalize(' '.join(s[:cutpoints[0]]))
#         alt_list=[curr_candidate]
        
#         if(curr_candidate in good_candidates):
#             multislices = [curr_candidate]

#         alt_list.extend(normalize(' '.join(s[cutpoints[i]:cutpoints[i+1]])) for i in range(k-1))
#         multislices.extend(normalize(' '.join(s[cutpoints[i]:cutpoints[i+1]])) for i in range(k-1) if normalize(' '.join(s[cutpoints[i]:cutpoints[i+1]])) in good_candidates)

#         curr_candidate=normalize(' '.join(s[cutpoints[k-1]:]))
#         alt_list.append(curr_candidate)
        
#         if(curr_candidate in good_candidates):
#             multislices.append(curr_candidate)
#         # print('::',alt_list)
#     return multislices

# def allPartitions(s):
#     n = len(s)
#     all_partitions=[]
#     all_partitions_length=[]
#     cuts = list(range(1,n))
#     for k in range(n):
#         # all_partitions_inner=[]
#         partition_list=[]
#         partition_length_list=[]
#         for cutpoints in itertools.combinations(cuts,k):
#             ret_list=multiSlice(s,cutpoints)
#             if(ret_list):
#                 partition_length=sum([len(elem.split()) for elem in ret_list])
#                 # print('==',ret_list,partition_length)
#                 if(partition_length==len(s)):
#                     return ret_list
#                 partition_list.append(ret_list)
#                 partition_length_list.append(partition_length)
#                 # yield ret_list
#         # print('------')
#         if(partition_length_list):
#             max_index=partition_length_list.index(max(partition_length_list))
#             all_partitions.append(partition_list[max_index])
#             all_partitions_length.append(partition_length_list[max_index])
#     # print(all_partitions)
#     max_index=all_partitions_length.index(max(all_partitions_length))
#     # print(all_partitions[max_index])
#     return all_partitions[max_index]

    



# partition = allPartitions(["bill","de","blasio's","new","york"])
# print(partition)

# word='john..jarvis...jake....'
# p=re.compile(r'[.]{2,}')
# # l= p.search(word)
# print(p.finditer(word))


# word="....john..jarvis...jake....j."
# p_dots= re.compile(r'[.]{2,}')

# # start=0
# matched_spans= list(p_dots.finditer(word)) 
# if matched_spans:
#     print('here')
#     temp=[]
#     # match_lst = p_dots.findall(word)
#     #[-1]).span()[1]
#     # print(matched_spans)
#     # whole_string_start=0
#     next_string_start=0
#     for matched_span in matched_spans:
#         matched_start=matched_span.span()[0]
#         this_excerpt=word[next_string_start:matched_start]
#         if(this_excerpt):
#             temp.append(this_excerpt)
#         next_string_start=matched_span.span()[1]
# if(next_string_start<len(word)):
#     last_excerpt=word[next_string_start:]
#     if(last_excerpt):
#         temp.append(last_excerpt)
# print(temp)

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import gutenberg
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
 
# print dir(gutenberg)
print (gutenberg.fileids())
 
text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)
trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
trainer.train(text)
tokenizer = PunktSentenceTokenizer(trainer.get_params())
tokenizer._params.abbrev_types.add('dr')
tokenizer._params.abbrev_types.add('c.j')
tokenizer._params.abbrev_types.add('dr')
tokenizer._params.abbrev_types.add('u.s.a')

txt = "C.J. McCollum Leads Blazers to Game 7 Win on Road  #RipCity"
sentences1= tokenizer.tokenize(txt)

    
tokenizer._params.abbrev_types.add('dr')
tokenizer._params.abbrev_types.add('c.j')

sentences2= tokenizer.tokenize(txt)

print(sentences1)
print('=========================')
print(sentences2)

# for p in parts:
#     print('==',p)