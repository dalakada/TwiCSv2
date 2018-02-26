import pandas as pd 
import warnings
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import math
from scipy import spatial
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from pandas.tools.plotting import parallel_coordinates


#entity_synvec_agg=[0.6103390782,0.0017224607,0.0819329593,0.1391210505,0.0210878352,0.1420126381]
ambiguous_synvec_agg=[0.3107914277,0.2406605708,0.0971315279,0.0478697633,0.1069009451,0.1966457653]
#non_entity_synvec_agg=[0.1042614218,0.0083554817,0.0556537306,0.0464007013,0.6816091473,0.1037195173]

candidate_records=pd.read_csv("candidate_base_new_analysis.csv",sep =',')

#-----------------------------------------------------Correlation Coefficients w Scatter Plots-----------------------------------------------------------
# temp_candidate_records=candidate_records[(candidate_records['normalized_cap']>0)]
# print("Capitalized:", temp_candidate_records['normalized_cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_cap'], temp_candidate_records['probability'], 'r.')
# plt.xlabel('Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

# temp_candidate_records=candidate_records[(candidate_records['normalized_non-cap']>0)]
# print("Non-capitalized:", temp_candidate_records['normalized_non-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_non-cap'], temp_candidate_records['probability'], 'g.')
# plt.xlabel('Non Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

# temp_candidate_records=candidate_records[(candidate_records['normalized_capnormalized_substring-cap']>0)]
# print("Substring Capitalized:",	temp_candidate_records['normalized_capnormalized_substring-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_capnormalized_substring-cap'], temp_candidate_records['probability'], 'b.')
# plt.xlabel('Substring Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

# temp_candidate_records=candidate_records[(candidate_records['normalized_s-o-sCap']>0)]
# print("Start-of-sentence Capitalized:",	temp_candidate_records['normalized_s-o-sCap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_s-o-sCap'], temp_candidate_records['probability'], 'm.')
# plt.xlabel('Start-of-Sentence Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

# temp_candidate_records=candidate_records[(candidate_records['normalized_all-cap']>0)]
# print("All Capitalized:",	temp_candidate_records['normalized_all-cap'].corr(temp_candidate_records['probability']))
# plt.plot(temp_candidate_records['normalized_all-cap'], temp_candidate_records['probability'], 'c.')
# plt.xlabel('All Capitalized frequency')
# plt.ylabel('Probability')
# plt.axis([0, 1, 0, 1])
# plt.show()

# temp_candidate_records=candidate_records[(candidate_records['normalized_non-discriminative']>0)]
# print("Non-discriminative:",	temp_candidate_records['normalized_non-discriminative'].corr(temp_candidate_records['probability']))
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


#--------------------------------Multi-dimensional data plot---------------------------------------------------

y=candidate_records['class']
candidate_records['normalized_length']=candidate_records['length']/(candidate_records['length'].max())
x=candidate_records[['normalized_length','normalized_cap','normalized_capnormalized_substring-cap','normalized_s-o-sCap','normalized_all-cap','normalized_non-cap','normalized_non-discriminative']]

#print(candidate_records['normalized_length'])
#--------------Using PCA
# pca = sklearnPCA(n_components=2) #2-dimensional PCA
# transformed = pd.DataFrame(pca.fit_transform(x))
# #print(transformed[y==1])
# plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Entity', c='red')
# # for i, row in transformed[y==1].iterrows():
# # 	#print((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1])
# #     plt.annotate(str(i), ((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1]))

# #print(transformed[y==2])
# plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Ambiguous', c='blue')
# for i, row in transformed[y==2].iterrows():
# 	#print((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1])
#     plt.annotate(str(i), ((transformed[y==2].loc[[i]])[0],(transformed[y==2].loc[[i]])[1]))

# plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Non-Entity', c='lightgreen')
# #print(len(transformed))
# # for index in range(len(transformed)):
# # 	plt.text(transformed[index,0], transformed[index,1], str(index))
# plt.xlabel('Transformed X-axis')
# plt.ylabel('Transformed Y-axis')
# plt.title("PCA plot of Entity Candidates")
# plt.legend()
# #plt.savefig('test-point-visualization-PCA.png', dpi = 600)
# plt.show()


#--------------Using LDA
# lda = LDA(n_components=2) #2-dimensional LDA
# lda_transformed = pd.DataFrame(lda.fit_transform(x, y))

# # Plot all three series
# plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Entity', c='red')
# plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Ambiguous', c='blue')
# plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Non-Entity', c='lightgreen')
# plt.xlabel('Transformed X-axis')
# plt.ylabel('Transformed Y-axis')
# plt.title("LDA plot of Entity Candidates")

# # Display legend and show plot
# plt.legend(loc=3)
# plt.show()


#--------------Using Parallel Coordinates
# y=candidate_records['status']
# # Select features to include in the plot
# plot_feat = ['normalized_cap','normalized_capnormalized_substring-cap','normalized_s-o-sCap','normalized_all-cap','normalized_non-cap','normalized_non-discriminative']

# # Concat classes with the normalized data
# data_norm = pd.concat([x, y], axis=1)

# # Perform parallel coordinate plot
# parallel_coordinates(data_norm, 'status')
# #my_labels=['Entity','Ambiguous','Non-Entity']
# plt.xticks(rotation=90)
# plt.legend()
# plt.title("Entity Candidates on Parallel Feature Coordinates")
# plt.savefig('test-point-visualization-parallel-coordinates.png', dpi = 600)
# plt.show()


#--------------Using t-SNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=500)
transformed = tsne.fit_transform(x)
print(len(transformed[y]))
plt.scatter(transformed[y==1][:, 0], transformed[y==1][:, 1], label='Entity', c='red')
# # for i, row in transformed[y==1].iterrows():
# # 	#print((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1])
# #     plt.annotate(str(i), ((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1]))

# #print(transformed[y==2])
plt.scatter(transformed[y==2][:, 0], transformed[y==2][:, 1], label='Ambiguous', c='blue')
# # for i, row in transformed[y==2].iterrows():
# # 	#print((transformed[y==1].loc[[i]])[0],(transformed[y==1].loc[[i]])[1])
# #     plt.annotate(str(i), ((transformed[y==2].loc[[i]])[0],(transformed[y==2].loc[[i]])[1]))

plt.scatter(transformed[y==3][:, 0], transformed[y==3][:, 1], label='Non-Entity', c='lightgreen')
#print(len(transformed))
# for index in range(len(transformed)):
# 	plt.text(transformed[index,0], transformed[index,1], str(index))

#plt.scatter(transformed[:, 0], transformed[:, 1], c=y,label=['Entity','Ambiguous','Non-Entity'])
plt.xlabel('Transformed X-axis')
plt.ylabel('Transformed Y-axis')
plt.legend()
plt.title("t-SNE plot of Entity Candidates")

#plt.savefig('test-point-visualization-PCA.png', dpi = 600)
plt.show()
