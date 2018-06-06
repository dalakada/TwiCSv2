#uncomment till here---> plot starts
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import leastsq,least_squares

ranking_effectiveness_single_sketch=[ 1.0, 0.8333333333333334, 0.8571428571428571, 0.76, 0.7586206896551724, 0.7222222222222222, 0.7441860465116279, 0.7169811320754716, 0.6885245901639344, 0.6818181818181818, 0.6619718309859155, 0.64, 0.6363636363636364, 0.6219512195121951, 0.5714285714285714, 0.5670103092783505, 0.5943396226415094, 0.6, 0.5948275862068966]
ranking_effectiveness_multi_sketch_cosine=[ 1.0, 0.8333333333333334, 0.7857142857142857, 0.68, 0.6896551724137931, 0.6111111111111112, 0.627906976744186, 0.5849056603773585, 0.5573770491803278, 0.5151515151515151, 0.5070422535211268, 0.5066666666666667, 0.5064935064935064, 0.4878048780487805, 0.4725274725274725, 0.4742268041237113, 0.5188679245283019, 0.5272727272727272, 0.5258620689655172]
ranking_effectiveness_multi_sketch_euclidean=[ 1.0, 0.8333333333333334, 0.7857142857142857, 0.68, 0.6551724137931034, 0.5555555555555556, 0.5813953488372093, 0.5471698113207547, 0.5245901639344263, 0.48484848484848486, 0.4788732394366197, 0.48, 0.4935064935064935, 0.4878048780487805, 0.4725274725274725, 0.4845360824742268, 0.5283018867924528, 0.5454545454545454, 0.5431034482758621]
combined_ranking_effectiveness=[ 1.0, 0.8333333333333334, 0.8571428571428571, 0.8, 0.8275862068965517, 0.7777777777777778, 0.7906976744186046, 0.7735849056603774, 0.7704918032786885, 0.7575757575757576, 0.7605633802816901, 0.76, 0.7662337662337663, 0.7560975609756098, 0.7252747252747253, 0.7216494845360825, 0.7452830188679245, 0.7545454545454545, 0.75]
batches= range(1,20)

# for i in range(10,30,5):
# 	print(i)


print(len(ranking_effectiveness_single_sketch),len(ranking_effectiveness_multi_sketch_cosine),len(ranking_effectiveness_multi_sketch_euclidean),len(combined_ranking_effectiveness))

fig = plt.figure()

plt.hold(True)
plt.xticks(np.arange(1, 20, 1.0))

plt.scatter(batches,ranking_effectiveness_single_sketch,label='single sketch, cosine distance')
plt.scatter(batches,ranking_effectiveness_multi_sketch_cosine,label='multiple sketches, cosine distance')
plt.scatter(batches,ranking_effectiveness_multi_sketch_euclidean,label='multiple sketches, euclidean distance')
plt.scatter(batches,combined_ranking_effectiveness,label='multiple sketching techniques combined')

plt.title("Ranking Function Effectiveness")
plt.ylabel('ranking fucntion top-15 effectiveness')
plt.xlabel('Current Batch')
plt.legend()
plt.show()

# def func(x,a,b1,b2):
# 	x1,x2=x
# 	return (a*np.exp(b1*x1+b2*x2))





#model function

# def func(x,a,b):
	
# 	return (a*np.exp(b*x))

# def func(x, a, b, c):
# 	# return a*np.exp(-b*x)+c
# 	return a*x**2 + b*x + c

# def residuals(coeffs,y,x):
# 	return y-func(x,coeffs[0],coeffs[1],coeffs[2])

# filename='entity-level-estimates-average.csv'
# data=pd.read_csv(filename,sep =',', index_col=False)
# fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(20,20) )
# fig = plt.figure()
# ax = fig.gca(projection='3d')

# ax.set_xticks([0, 20, 1.0])
# ax.set_yticks([0, 20, 1.0])


# x_data=[]
# x0=[]
# x1=[]
# y_data=[]

# # for i, ax in enumerate(axes.flat, start=0):
# # 	print(i)
# for i in range(11):
# 	fig, ax = plt.subplots(nrows=1, ncols=1)
# 	ax.set_xticks(np.arange(0, 20, 1.0))
# 	x_val=list(range(i,20))
# 	#x_val_transformed=[j-i for j in x_val]
# 	x0_arr=[j-i for j in x_val]
# 	y_val=data[str(i)].tolist()[i:]
# 	# y_data.extend(y_val)
# 	#print(len(x_val),len(y_val[i:]))
# 	#y_val_norm=[(y_val[0]-i)/y_val[0] for i in y_val]
# 	#plt.plot(x_val,y_val_norm,label=str(i))
# 	x0_arr_truncated=x0_arr[:16]
# 	y_val_truncated=y_val[:16]
# 	# print(x0_arr,y_val)
# 	# print(len(x0_arr_truncated),len(y_val_truncated))
# 	# ax.scatter(x0_arr,y_val)
	
# 	# np_poly=np.polyfit(np.array(x0_arr), np.array(y_val), 3)
# 	# estimated_y = np.polyval(np_poly,np.array(x0_arr))
# 	np_poly=np.polyfit(np.array(x0_arr_truncated), np.array(y_val_truncated), 3)
# 	estimated_y = np.polyval(np_poly,np.array(x0_arr))
# 	print('predictions for batch '+str(i)+':')
# 	print(y_val)
# 	print(estimated_y)
	
	# p0=np.array([y_val[0],1e-4,1], dtype=float)
	# curve_fit_coeff,cov = curve_fit(func, np.array(x0_arr), np.array(y_val))
	# estimated_y=func(np.array(x0_arr),curve_fit_coeff[0],curve_fit_coeff[1],curve_fit_coeff[2])
	# print(cov)

	# res_lsq = least_squares(residuals, p0, args=(np.array(x0_arr), np.array(y_val)))
	# # print(res_lsq)
	# estimated_y=func(np.array(x0_arr),res_lsq.x[0],res_lsq.x[1],res_lsq.x[2])

	# res_robust = least_squares(residuals, p0, loss='soft_l1', f_scale=0.1, args=(np.array(x0_arr), np.array(y_val)))
	# # print(res_lsq)
	# estimated_y=func(np.array(x0_arr),res_robust.x[0],res_robust.x[1],res_robust.x[2])
	
	
	# print(estimated_y)
	# ax.plot(x0_arr,estimated_y)

	# ax.set_title("Propagation of Ambiguous Candidates from batch "+str(i))
	# ax.set_ylabel('# of ambiguous candidates remaining')
	# ax.set_xlabel('Current Batch')

	# plt.show()


# fig.tight_layout()

# plt.show()
	

	# plt.plot(x_val_transformed,y_val_norm,label=str(i))
# 	x0.extend(x0_arr)
# 	x1.extend(x_val)
# x_data=[x0,x1]
# x_data=np.array(x_data)

# coeff, cov=curve_fit(func, x_data,y_data)
# print(coeff)
# estimated_y=func(x_data,coeff[0],coeff[1],coeff[2])
# print(len(x_data[0]),len(x_data[1]),len(estimated_y))
# ax.scatter(x_data[0], x_data[1],estimated_y, alpha=0.2)


# ax.set_xlabel('Entry Batch')

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
