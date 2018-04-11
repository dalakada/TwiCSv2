#uncomment till here---> plot starts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# plt.hold(True)
# plt.xticks(np.arange(0, 20, 1.0))
def func(x,a,b1,b2):
	x1,x2=x
	return (a*np.exp(b1*x1+b2*x2))

filename='entity-level-estimates-average.csv'
data=pd.read_csv(filename,sep =',', index_col=False)
x_data=[]
y_data=[]
for i in range(20):
	x_val=list(range(i,20))
	x_val_transformed=[j-i for j in x_val]
	x_tuple_list=[(i,j) for j in x_val]
	x_data.extend(x_tuple_list)
	y_val=data[str(i)].tolist()[i:]
	y_data.extend(y_val)
	#print(len(x_val),len(y_val[i:]))
	y_val_norm=[(y_val[0]-i)/y_val[0] for i in y_val]
	#plt.plot(x_val,y_val_norm,label=str(i))
	# plt.plot(x_val_transformed,y_val_norm,label=str(i))
coeff=curve_fit(func, x_data,y_data)


#plt.title("Propagation of Ambiguous Candidates through batches")
#plt.ylabel('# of ambiguous candidates remaining')
#plt.xlabel('Current Batch')

# plt.title("Propagation of Ambiguous Candidates labeling through batches")
# plt.ylabel('# of ambiguous candidates labelled')
# plt.xlabel('Batches since entry')

# plt.legend()
# plt.show()
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
