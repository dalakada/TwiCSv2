 
# coding: utf-8
import pandas as pd
# import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import autograd.numpy as np
from autograd import elementwise_grad, grad

from scipy import stats

class SVM1():

    def __init__(self,train):

        #train the algorithm once
        self.train = pd.read_csv(train,delimiter=",",sep='\s*,\s*')

        #'''
        self.train['normalized_cap']=self.train['cap']/self.train['cumulative']
        self.train['normalized_capnormalized_substring-cap']=self.train['substring-cap']/self.train['cumulative']
        self.train['normalized_s-o-sCap']=self.train['s-o-sCap']/self.train['cumulative']
        self.train['normalized_all-cap']=self.train['all-cap']/self.train['cumulative']
        self.train['normalized_non-cap']=self.train['non-cap']/self.train['cumulative']
        self.train['normalized_non-discriminative']=self.train['non-discriminative']/self.train['cumulative']

        self.cols = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.return_cols = ['candidate','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.h=1e-5

        self.grad1_cols_plus=['length','normalized_cap_plus_delta',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative']

        self.grad1_cols_minus=['length','normalized_cap_minus_delta',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative']

        self.grad2_cols_plus=['length','normalized_cap',
        'normalized_capnormalized_substring-cap_plus_delta',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative']

        self.grad2_cols_minus=['length','normalized_cap',
        'normalized_capnormalized_substring-cap_minus_delta',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative']

        self.grad3_cols_plus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap_plus_delta',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.grad3_cols_minus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap_minus_delta',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.grad4_cols_plus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap_plus_delta',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.grad4_cols_minus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap_minus_delta',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]

        self.grad5_cols_plus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap_plus_delta',
        'normalized_non-discriminative'
        ]

        self.grad5_cols_minus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap_minus_delta',
        'normalized_non-discriminative'
        ]

        self.grad6_cols_plus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative_plus_delta'
        ]

        self.grad6_cols_minus = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative_minus_delta'
        ]

        #'''

        '''
        self.train['cumulative_red']=self.train['cumulative']-self.train['non-discriminative']

        self.train['normalized_cap']=self.train['cap']/self.train['cumulative_red']
        self.train['normalized_capnormalized_substring-cap']=self.train['substring-cap']/self.train['cumulative_red']
        self.train['normalized_s-o-sCap']=self.train['s-o-sCap']/self.train['cumulative_red']
        self.train['normalized_all-cap']=self.train['all-cap']/self.train['cumulative_red']
        self.train['normalized_non-cap']=self.train['non-cap']/self.train['cumulative_red']


        self.cols = ['length','cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative','cumulative','cumulative_red',
        'normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap'
        ] '''
        
        # print(self.train[(self.train['class']==1)])

        self.colsRes = ['class']

        # self.trainArr = self.train.as_matrix(self.cols) #training array
        # #print(self.trainArr)
        # self.trainRes = self.train.as_matrix(self.colsRes) # training results

        self.trainArr = self.train[self.cols]
        self.trainRes = self.train[self.colsRes].values

        self.clf = svm.SVC(probability=True)
        # self.clf.fit(self.trainArr, self.trainRes) # fit the data to the algorithm

        #--------------only for efficiency experiments
        # self.scaler = StandardScaler()

        # self.clf = CalibratedClassifierCV(base_estimator=svm.SVC(probability=True), cv=5)
        # self.clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=5)
        self.clf.fit(self.trainArr, self.trainRes)

        # X_train = self.scaler.fit_transform(self.trainArr)
        # self.clf.fit(X_train, self.trainRes)

        # self.clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
        # self.clf.fit(self.trainArr, self.trainRes)
        # self.partial_grad_0=grad(self.clf.predict_proba,0)
        # self.partial_grad_1=[elementwise_grad(func[1],1) for func in self.clf.predict_proba(testArr)]
        # self.partial_grad_2=elementwise_grad(self.clf.predict_proba,2)
        # self.partial_grad_3=elementwise_grad(self.clf.predict_proba,3)
        # self.partial_grad_4=elementwise_grad(self.clf.predict_proba,4)
        # self.partial_grad_5=elementwise_grad(self.clf.predict_proba,5)
        # self.partial_grad_6=elementwise_grad(self.clf.predict_proba,6)


    def partial_grad_1(self,x_ambiguous):
        # print(x_ambiguous_arr)
        # return self.partial_grad_1(x_point)
        # func=self.clf.predict_proba(x_ambiguous_arr)[elem]

        x_ambiguous['normalized_cap_plus_delta']=x_ambiguous['normalized_cap']+self.h
        # plus_delta=self.clf.predict_proba(x_ambiguous_arr)[:, 1]
        x_ambiguous['normalized_cap_minus_delta']=x_ambiguous['normalized_cap']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad1_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)[:,1]

        minus_delta_arr= x_ambiguous[self.grad1_cols_minus].values

        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)
        # minus_delta=self.clf.predict_proba(minus_delta_arr)[:,1]

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))

    def partial_grad_2(self,x_ambiguous):

        x_ambiguous['normalized_capnormalized_substring-cap_plus_delta']=x_ambiguous['normalized_capnormalized_substring-cap']+self.h
        x_ambiguous['normalized_capnormalized_substring-cap_minus_delta']=x_ambiguous['normalized_capnormalized_substring-cap']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad2_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)

        minus_delta_arr= x_ambiguous[self.grad2_cols_minus].values
        # minus_delta=self.clf.predict_proba(minus_delta_arr)

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))
        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)
      
    def partial_grad_3(self,x_ambiguous):

        x_ambiguous['normalized_s-o-sCap_plus_delta']=x_ambiguous['normalized_s-o-sCap']+self.h
        x_ambiguous['normalized_s-o-sCap_minus_delta']=x_ambiguous['normalized_s-o-sCap']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad3_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)

        minus_delta_arr= x_ambiguous[self.grad3_cols_minus].values
        # minus_delta=self.clf.predict_proba(minus_delta_arr)

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))
        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)

    def partial_grad_4(self,x_ambiguous):

        x_ambiguous['normalized_all-cap_plus_delta']=x_ambiguous['normalized_all-cap']+self.h
        x_ambiguous['normalized_all-cap_minus_delta']=x_ambiguous['normalized_all-cap']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad4_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)

        minus_delta_arr= x_ambiguous[self.grad4_cols_minus].values
        # minus_delta=self.clf.predict_proba(minus_delta_arr)

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))
        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)

    def partial_grad_5(self,x_ambiguous):

        x_ambiguous['normalized_non-cap_plus_delta']=x_ambiguous['normalized_non-cap']+self.h
        x_ambiguous['normalized_non-cap_minus_delta']=x_ambiguous['normalized_non-cap']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad5_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)

        minus_delta_arr= x_ambiguous[self.grad5_cols_minus].values
        # minus_delta=self.clf.predict_proba(minus_delta_arr)

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))
        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)

    def partial_grad_6(self,x_ambiguous):

        x_ambiguous['normalized_non-discriminative_plus_delta']=x_ambiguous['normalized_non-discriminative']+self.h
        x_ambiguous['normalized_non-discriminative_minus_delta']=x_ambiguous['normalized_non-discriminative']-2*self.h

        plus_delta_arr= x_ambiguous[self.grad6_cols_plus].values
        # plus_delta=self.clf.predict_proba(plus_delta_arr)

        minus_delta_arr= x_ambiguous[self.grad6_cols_minus].values
        # minus_delta=self.clf.predict_proba(minus_delta_arr)

        # return np.divide(np.subtract(plus_delta,minus_delta),(2*self.h))
        return np.concatenate([plus_delta_arr,minus_delta_arr], axis=0)


    def get_partial_derivatives(self,x_ambiguous):
        x_ambiguous['normalized_cap']=x_ambiguous['cap']/x_ambiguous['cumulative']
        x_ambiguous['normalized_capnormalized_substring-cap']=x_ambiguous['substring-cap']/x_ambiguous['cumulative']
        x_ambiguous['normalized_s-o-sCap']=x_ambiguous['s-o-sCap']/x_ambiguous['cumulative']
        x_ambiguous['normalized_all-cap']=x_ambiguous['all-cap']/x_ambiguous['cumulative']
        x_ambiguous['normalized_non-cap']=x_ambiguous['non-cap']/x_ambiguous['cumulative']
        x_ambiguous['normalized_non-discriminative']=x_ambiguous['non-discriminative']/x_ambiguous['cumulative']

        ambiguous_candidate_records_list= x_ambiguous[self.return_cols].values.tolist()

        dataframe_len= len(x_ambiguous)

        # partial_grad_list=[self.partial_grad_1(x_ambiguous),self.partial_grad_2(x_ambiguous),self.partial_grad_3(x_ambiguous),self.partial_grad_4(x_ambiguous),self.partial_grad_5(x_ambiguous),self.partial_grad_6(x_ambiguous)]
        # partial_grad_restacked=np.stack(partial_grad_list, axis=1)

        elementwise_del_list=[self.partial_grad_1(x_ambiguous),self.partial_grad_2(x_ambiguous),self.partial_grad_3(x_ambiguous),self.partial_grad_4(x_ambiguous),self.partial_grad_5(x_ambiguous),self.partial_grad_6(x_ambiguous)]
        elementwise_del_list_concatenated=np.concatenate(elementwise_del_list, axis=0)

        # for elem in elementwise_del_list:
        #     print(len(elem))
        # print(len(elementwise_del_list_concatenated))

        elementwise_del_list_concatenated_output=self.clf.decision_function(elementwise_del_list_concatenated)
        # print(elementwise_del_list_concatenated_output)

        partial_grad_1=np.divide(np.subtract(elementwise_del_list_concatenated_output[(0*dataframe_len):(1*dataframe_len)],elementwise_del_list_concatenated_output[(1*dataframe_len):(2*dataframe_len)]),(2*self.h))
        partial_grad_2=np.divide(np.subtract(elementwise_del_list_concatenated_output[(2*dataframe_len):(3*dataframe_len)],elementwise_del_list_concatenated_output[(3*dataframe_len):(4*dataframe_len)]),(2*self.h))
        partial_grad_3=np.divide(np.subtract(elementwise_del_list_concatenated_output[(4*dataframe_len):(5*dataframe_len)],elementwise_del_list_concatenated_output[(5*dataframe_len):(6*dataframe_len)]),(2*self.h))
        partial_grad_4=np.divide(np.subtract(elementwise_del_list_concatenated_output[(6*dataframe_len):(7*dataframe_len)],elementwise_del_list_concatenated_output[(7*dataframe_len):(8*dataframe_len)]),(2*self.h))
        partial_grad_5=np.divide(np.subtract(elementwise_del_list_concatenated_output[(8*dataframe_len):(9*dataframe_len)],elementwise_del_list_concatenated_output[(9*dataframe_len):(10*dataframe_len)]),(2*self.h))
        partial_grad_6=np.divide(np.subtract(elementwise_del_list_concatenated_output[(10*dataframe_len):(11*dataframe_len)],elementwise_del_list_concatenated_output[(11*dataframe_len):(12*dataframe_len)]),(2*self.h))
        
        partial_grad_list=[partial_grad_1,partial_grad_2,partial_grad_3,partial_grad_4,partial_grad_5,partial_grad_6]
        partial_grad_restacked=np.stack(partial_grad_list, axis=1)

        # print(len(partial_grad_1),len(partial_grad_2),len(partial_grad_3),len(partial_grad_4),len(partial_grad_5),len(partial_grad_6),len(partial_grad_restacked))

        # for elem in range(len(partial_grad_1)):
        #     print(partial_grad_1[elem],partial_grad_2[elem],partial_grad_3[elem],partial_grad_4[elem],partial_grad_5[elem],partial_grad_6[elem])
        #     print(partial_grad_restacked[elem])

        return ambiguous_candidate_records_list, partial_grad_restacked


    def run(self,x_test,z_score_threshold):
    # def run(self,x_test,cumulative_threshold): #for the efficiency_run
        #'''
        x_test['normalized_cap']=x_test['cap']/x_test['cumulative']
        x_test['normalized_capnormalized_substring-cap']=x_test['substring-cap']/x_test['cumulative']
        x_test['normalized_s-o-sCap']=x_test['s-o-sCap']/x_test['cumulative']
        x_test['normalized_all-cap']=x_test['all-cap']/x_test['cumulative']
        x_test['normalized_non-cap']=x_test['non-cap']/x_test['cumulative']
        x_test['normalized_non-discriminative']=x_test['non-discriminative']/x_test['cumulative'] #'''

        '''
        x_test['cumulative_red']=x_test['cumulative']-x_test['non-discriminative']

        x_test['normalized_cap']=x_test['cap']/x_test['cumulative_red']
        x_test['normalized_capnormalized_substring-cap']=x_test['substring-cap']/x_test['cumulative_red']
        x_test['normalized_s-o-sCap']=x_test['s-o-sCap']/x_test['cumulative_red']
        x_test['normalized_all-cap']=x_test['all-cap']/x_test['cumulative_red']
        x_test['normalized_non-cap']=x_test['non-cap']/x_test['cumulative_red']
        '''


        #setting features
        # testArr= x_test.as_matrix(self.cols)
        # #print(testArr)
        # testRes = x_test.as_matrix(self.colsRes)

        testArr = x_test[self.cols]


        # In[ ]:
        # print(x_test[(x_test['cumulative_red']==0.0)])




        # In[65]:

        #clf = svm.SVC(probability=True)
        #clf.fit(trainArr, trainRes) # fit the data to the algorithm


        # In[66]:

        pred_prob=self.clf.predict_proba(testArr)

        # X_test = self.scaler.fit_transform(testArr)
        # pred_prob=self.clf.predict_proba(X_test)


        # In[67]:

        prob_first_column= pred_prob[:, 1]
        # for i in pred_prob:
        #     prob_first_column.append(i[1])
            


        # In[68]:

        #print(x_test_filtered.index.size,len(prob_first_column))


        # In[69]:
        #print(pred_prob) 
        x_test['probability']=prob_first_column


        # In[70]:

        #type(x_test)


        # In[46]:

        #type(x_test)


        # In[71]:

        #x_test_filtered.to_csv("results3.csv", sep=',', encoding='utf-8')


        # In[48]:

        return x_test


'''

# In[109]:

ali.to_csv("Classifier_Results.csv", sep=',', encoding='utf-8')


# In[68]:

pred_class=clf.predict(testArr)
print(pred_class)


# In[69]:

testRes


# In[10]:

count=0


# In[11]:

for i in range(len(pred_class)):
    if pred_class[i]==testRes[i]:
        count+=1


# In[12]:

count


# In[13]:

len(pred_class)


# In[14]:

float(count)/len(pred_class)


# In[22]:

prob_holder=[]
for idx, cl in enumerate(pred_prob):
    prob_holder.append(pred_prob[idx][1])
#x_test.insert(len(x_test.columns),'pred_prob',pred_prob[1])
#print (pred_prob[,1])
#x_test.insert(1, 'bar', df['one'])


# In[23]:

x_test.to_csv("svm_prob.csv", sep=';', encoding='utf-8')



# In[24]:

random_forest_logistic=pd.read_csv("random_forest_logistic.csv",delimiter=";")


# In[25]:

random_forest_logistic


# In[26]:

prob_holder=[]
for idx, cl in enumerate(pred_prob):
    prob_holder.append(pred_prob[idx][1])
#x_test.insert(len(x_test.columns),'pred_prob',pred_prob[1])
#print (pred_prob[,1])
#x_test.insert(1, 'bar', df['one'])


# In[27]:

random_forest_logistic.insert(len(random_forest.columns),'svm_with_prob',prob_holder)
print random_forest_logistic


# In[29]:

random_forest_logistic.to_csv("random_forest_logistic_svm_FINAL.csv", sep=';', encoding='utf-8')


# In[34]:

class_x=0
TP=0
TN=0
FP=0
FN=0

for idx, cl in enumerate(pred_prob):
    #print pred_prob[idx][1]
    #if pred_prob[idx][1]>0.6:
       # class_x=1
    #elif pred_prob[idx][1]<=0.6:
      #  class_x=0
    class_x = pred_class[idx]   

    if (class_x ==testRes[idx]) and class_x==1 :
        TP+=1
    elif (class_x ==testRes[idx]) and class_x==0 :
        TN+=1
    if class_x ==  1 and testRes[idx]==0:
        FP+=1
    if class_x ==  0 and testRes[idx]==1:
        FN+=1


# In[35]:

print TP,TN,FP,FN


# In[ ]:



'''