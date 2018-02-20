import pandas as pd 
import warnings
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from scipy import spatial


#entity_synvec_agg=[0.6103390782,0.0017224607,0.0819329593,0.1391210505,0.0210878352,0.1420126381]
ambiguous_synvec_agg=[0.3107914277,0.2406605708,0.0971315279,0.0478697633,0.1069009451,0.1966457653]
#non_entity_synvec_agg=[0.1042614218,0.0083554817,0.0556537306,0.0464007013,0.6816091473,0.1037195173]

candidate_records=pd.read_csv("candidate_base_new_analysis.csv",sep =',')

#-----------------------------------------------------Correlation Coefficients w Scatter Plots-----------------------------------------------------------
temp_candidate_records=candidate_records[(candidate_records['normalized_cap']>0)]
print("Capitalized:", temp_candidate_records['normalized_cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_cap'], temp_candidate_records['probability'], 'r.')
# plt.xlabel('Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

temp_candidate_records=candidate_records[(candidate_records['normalized_non-cap']>0)]
print("Non-capitalized:", temp_candidate_records['normalized_non-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_non-cap'], temp_candidate_records['probability'], 'g.')
# plt.xlabel('Non Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

temp_candidate_records=candidate_records[(candidate_records['normalized_capnormalized_substring-cap']>0)]
print("Substring Capitalized:",	temp_candidate_records['normalized_capnormalized_substring-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_capnormalized_substring-cap'], temp_candidate_records['probability'], 'b.')
# plt.xlabel('Substring Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

temp_candidate_records=candidate_records[(candidate_records['normalized_s-o-sCap']>0)]
print("Start-of-sentence Capitalized:",	temp_candidate_records['normalized_s-o-sCap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_s-o-sCap'], temp_candidate_records['probability'], 'm.')
# plt.xlabel('Start-of-Sentence Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

temp_candidate_records=candidate_records[(candidate_records['normalized_all-cap']>0)]
print("All Capitalized:",	temp_candidate_records['normalized_all-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_all-cap'], temp_candidate_records['probability'], 'c.')
# plt.xlabel('All Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

temp_candidate_records=candidate_records[(candidate_records['normalized_non-discriminative']>0)]
print("Non-discriminative:",	temp_candidate_records['normalized_non-discriminative'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_non-discriminative'], temp_candidate_records['probability'], 'y.')
# plt.xlabel('Non-discriminative frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

entity_candidate_records=candidate_records[(candidate_records.status=="g")]
non_entity_candidate_records=candidate_records[(candidate_records.status=="b")]
ambiguous_candidate_records=candidate_records[(candidate_records.status=="a")]
#print(entity_candidate_records)




#--------------------------------------------------------calculate aggregate vectors-------------------------------------------------------

#entity_vec_agg
# entity_count=0
# entity_synvec_agg=[0.0,0.0,0.0,0.0,0.0]
# for index, row in entity_candidate_records.iterrows():
# 	normalized_cap=(row['cap']+row['all-cap'])/row['cumulative']
# 	entity_synvec_agg[0]+=normalized_cap
# 	normalized_capnormalized_substring_cap=row['substring-cap']/row['cumulative']
# 	entity_synvec_agg[1]+=normalized_capnormalized_substring_cap
# 	normalized_sosCap=row['s-o-sCap']/row['cumulative']
# 	entity_synvec_agg[2]+=normalized_sosCap
# 	normalized_non_cap=row['non-cap']/row['cumulative']
# 	entity_synvec_agg[3]+=normalized_non_cap
# 	normalized_non_discriminative=row['non-discriminative']/row['cumulative']
# 	entity_synvec_agg[4]+=normalized_non_discriminative
# 	entity_count+=1
# entity_synvec_agg=list(map(lambda elem: elem/entity_count, entity_synvec_agg))

# print(entity_synvec_agg)

# #non_entity_vec_agg
# non_entity_count=0
# non_entity_synvec_agg=[0.0,0.0,0.0,0.0,0.0]
# for index, row in non_entity_candidate_records.iterrows():
# 	normalized_cap=(row['cap']+row['all-cap'])/row['cumulative']
# 	non_entity_synvec_agg[0]+=normalized_cap
# 	normalized_capnormalized_substring_cap=row['substring-cap']/row['cumulative']
# 	non_entity_synvec_agg[1]+=normalized_capnormalized_substring_cap
# 	normalized_sosCap=row['s-o-sCap']/row['cumulative']
# 	non_entity_synvec_agg[2]+=normalized_sosCap
# 	normalized_non_cap=row['non-cap']/row['cumulative']
# 	non_entity_synvec_agg[3]+=normalized_non_cap
# 	normalized_non_discriminative=row['non-discriminative']/row['cumulative']
# 	non_entity_synvec_agg[4]+=normalized_non_discriminative
# 	non_entity_count+=1
# non_entity_synvec_agg=list(map(lambda elem: elem/entity_count, non_entity_synvec_agg))

# print(non_entity_synvec_agg)


#--------------------------------Cosine distance computation---------------------------------------------------

# for index, row in entity_candidate_records.iterrows():
#  	candidate_synvec=[(row['normalized_cap']+row['normalized_all-cap']),
#  						row['normalized_capnormalized_substring-cap'],
#  						row['normalized_s-o-sCap'],
#  						row['normalized_non-cap'],
#  						row['normalized_non-discriminative']]
#  	cosine_distance_ent=spatial.distance.cosine(candidate_synvec, entity_synvec_agg)
#  	cosine_distance_non_ent=spatial.distance.cosine(candidate_synvec, non_entity_synvec_agg)
#  	print(row['candidate'],abs(cosine_distance_ent-cosine_distance_non_ent))


# for index, row in non_entity_candidate_records.iterrows():
#  	candidate_synvec=[(row['normalized_cap']+row['normalized_all-cap']),
#  						row['normalized_capnormalized_substring-cap'],
#  						row['normalized_s-o-sCap'],
#  						row['normalized_non-cap'],
#  						row['normalized_non-discriminative']]
#  	cosine_distance_ent=spatial.distance.cosine(candidate_synvec, entity_synvec_agg)
#  	cosine_distance_non_ent=spatial.distance.cosine(candidate_synvec, non_entity_synvec_agg)
#  	print(row['candidate'],abs(cosine_distance_ent-cosine_distance_non_ent))

# for index, row in ambiguous_candidate_records.iterrows():
#  	candidate_synvec=[(row['normalized_cap']+row['normalized_all-cap']),
#  						row['normalized_capnormalized_substring-cap'],
#  						row['normalized_s-o-sCap'],
#  						row['normalized_non-cap'],
#  						row['normalized_non-discriminative']]
#  	cosine_distance_ent=spatial.distance.cosine(candidate_synvec, entity_synvec_agg)
#  	cosine_distance_non_ent=spatial.distance.cosine(candidate_synvec, non_entity_synvec_agg)
#  	print(row['candidate'],abs(cosine_distance_ent-cosine_distance_non_ent))
