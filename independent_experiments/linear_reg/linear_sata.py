import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd


train = pd.read_csv("candidate_base_new_analysis.csv",delimiter=",",sep='\s*,\s*')



cols = ['s-o-sCap','all-cap',
'non-cap',]


colsRes = ['probability']

trainArr = train.as_matrix(cols) 
trainRes = train.as_matrix(colsRes)


col_mean = np.nanmean(trainArr, axis=0)
inds = np.where(np.isnan(trainArr))
trainArr[inds] = np.take(col_mean, inds[1])


col_mean = np.nanmean(trainRes, axis=0)
inds = np.where(np.isnan(trainRes))
trainRes[inds] = np.take(col_mean, inds[1])




regr = linear_model.LinearRegression()
regr.fit(trainArr, trainRes)
print('Coefficients: \n', regr.coef_)


