#uncomment till here---> plot starts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.hold(True)
plt.xticks(np.arange(0, 20, 1.0))
filename='sentence-level-estimates-average.csv'
data=pd.read_csv(filename,sep =',', index_col=False)
for i in range(20):
	x_val=list(range(i,20))
	y_val=data[str(i)].tolist()
	#print(len(x_val),len(y_val[i:]))
	plt.plot(x_val,y_val[i:],label=str(i))

# plt.plot([19], [9], linestyle='--', marker='o', label='entry-batch 19',c='blue')
# plt.plot([18,19], [10,8], linestyle='--', marker='o', label='batch 18',c='green')
# plt.plot([17,18,19], [10,8,7], linestyle='--', marker='o', label='batch 17',c='red')
# plt.plot([16,17,18,19], [12,8,6,6], linestyle='-', marker='o', label='batch 16',c='blue')
# plt.plot([15,16,17,18,19], [11,11,11,11,11], linestyle='-', marker='o', label='batch 15',c='green')
# plt.plot([14,15,16,17,18,19], [12,11,10,10,10,10], linestyle='-', marker='o', label='batch 14',c='red')
# plt.plot([13,14,15,16,17,18,19], [11,9,10,10,9,9,9], linestyle=':', marker='o', label='batch 13',c='blue')
# plt.plot([12,13,14,15,16,17,18,19], [16,16,13,11,12,12,12,12], linestyle=':', marker='o', label='batch 12',c='green')
# plt.plot([11,12,13,14,15,16,17,18,19], [13,11,10,12,11,9,8,8,8], linestyle=':', marker='o', label='batch 11',c='red')
# plt.plot([10,11,12,13,14,15,16,17,18,19], [9,8,8,8,8,8,8,8,8,8], linestyle='-.', marker='o', label='batch 10',c='blue')
# plt.plot([9,10,11,12,13,14,15,16,17,18,19], [11,8,7,7,7,7,7,7,7,7,7], linestyle='-.', marker='o', label='batch 9',c='green')
# plt.plot([8,9,10,11,12,13,14,15,16,17,18,19], [13,10,9,9,10,10,9,8,8,8,8,8], linestyle='-.', marker='o', label='batch 8',c='red')
# plt.plot([7,8,9,10,11,12,13,14,15,16,17,18,19], [19,18,16,16,16,15,15,15,15,15,15,15,16], linestyle='--', marker='o', label='batch 7',c='cyan')
# plt.plot([6,7,8,9,10,11,12,13,14,15,16,17,18,19], [15,14,12,12,12,11,11,11,11,11,11,11,11,11], linestyle='--', marker='o', label='batch 6',c='magenta')
# plt.plot([5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [18,16,12,12,10,10,10,10,10,10,10,10,9,9,9], linestyle='--', marker='o', label='batch 5',c='black')
# plt.plot([4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [14,12,12,12,12,12,12,12,12,12,11,10,10,10,10,10], linestyle=':', marker='o', label='batch 4',c='cyan')
# plt.plot([3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [11,8,7,6,5,5,5,6,6,6,6,6,6,6,6,6,5], linestyle=':', marker='o', label='batch 3',c='magenta')
# plt.plot([2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [15,13,13,12,11,12,9,9,9,10,10,10,9,9,9,8,8,8], linestyle=':', marker='o', label='batch 2',c='black')
# plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [20,17,17,15,13,13,13,14,13,13,13,13,13,13,13,12,12,12,12], linestyle='-', marker='o', label='batch 1',c='cyan')
# plt.plot([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], [27,24,22,19,19,19,19,18,18,18,18,18,18,18,18,18,16,16,16,16], linestyle='-', marker='o', label='batch 0',c='magenta')

plt.title("Propagation of Ambiguous Candidates through batches")
plt.ylabel('# of ambiguous candidates')
plt.xlabel('Current Batch')
plt.legend()
plt.show()
#----------------------------------plot code ends---------------------------------------------
# extenstion='.csv'
# sentence='sentence-level-estimates-'
# mention='mention-level-estimates-'
# entity='entity-level-estimates-'
# column_headers=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
# # name=entity+str(0)+extenstion  # for entities
# name=mention+str(0)+extenstion # for mentions
# # name=sentence+str(0)+extenstion  # for sentences
# df_original=pd.read_csv(name,sep =',', index_col=False)
# averaged_data=pd.DataFrame(index=df_original.index,columns=df_original.columns)
# averaged_data = averaged_data.fillna(0)

# for iter in range(10):
# 	# name=entity+str(iter)+extenstion # for entities
# 	name=mention+str(iter)+extenstion # for mentions
# 	# name=sentence+str(iter)+extenstion # for sentences
# 	print(name)
# 	data=pd.read_csv(name,sep =',', index_col=False)
# 	averaged_data=averaged_data.add(data)
# averaged_data= averaged_data/10
# # averaged_data.to_csv(entity+'average'+extenstion) # for entities
# averaged_data.to_csv(mention+'average'+extenstion) # for mentions
# # averaged_data.to_csv(sentence+'average'+extenstion) # for sentences
#----------------------------------averaging code ends---------------------------------------------


# good= [275498,
# 270537,
# 270038,
# 270372,
# 269851,
# 270641,
# 269851,
# 270078,
# 270038,
# 270372]

# bad= [ 708783,
# 720784,
# 723066,
# 718229,
# 719500,
# 718229,
# 722636,
# 722836,
# 722836,
# 721364]

# ambiguous= [ 42873,
# 35833,
# 34050,
# 38553,
# 37803,
# 38284,
# 34667,
# 34240,
# 34280,
# 35418]

# incomplete=[34788,
# 35670,
# 34030,
# 38261,
# 37529,
# 38017,
# 34549,
# 34191,
# 34191,
# 35253]

# print('good: ',str(sum(good)/len(good)))
# print('bad: ',str(sum(bad)/len(bad)))
# print('ambiguous: ',str(sum(ambiguous)/len(ambiguous)))
# print('incomplete: ',str(sum(incomplete)/len(incomplete)))
