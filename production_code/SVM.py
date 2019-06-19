 
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import svm

from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


from scipy import stats

class SVM1():
    def __init__(self,train):

        #train the algorithm once
        self.train = pd.read_csv(train,delimiter=",",sep='\s*,\s*')

        self.train['normalized_cap']=self.train['cap']/self.train['cumulative']
        self.train['normalized_capnormalized_substring-cap']=self.train['substring-cap']/self.train['cumulative']
        self.train['normalized_s-o-sCap']=self.train['s-o-sCap']/self.train['cumulative']
        self.train['normalized_all-cap']=self.train['all-cap']/self.train['cumulative']
        self.train['normalized_non-cap']=self.train['non-cap']/self.train['cumulative']
        self.train['normalized_non-discriminative']=self.train['non-discriminative']/self.train['cumulative']


        '''self.cols = ['length','cap','substring-cap','s-o-sCap','all-cap','non-cap','non-discriminative','cumulative',
        'normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]'''
        self.cols = ['length','normalized_cap',
        'normalized_capnormalized_substring-cap',
        'normalized_s-o-sCap',
        'normalized_all-cap',
        'normalized_non-cap',
        'normalized_non-discriminative'
        ]
        self.colsRes = ['class']

        self.trainArr = self.train.as_matrix(self.cols) #training array
        #print(self.trainArr)
        self.trainRes = self.train.as_matrix(self.colsRes) # training results

        # self.clf = svm.SVC(probability=True)
        # self.clf.fit(self.trainArr, self.trainRes) # fit the data to the algorithm

        #--------------only for efficiency experiments
        # self.scaler = StandardScaler()
        self.clf = CalibratedClassifierCV(base_estimator=svm.SVC(probability=True), cv=5)
        # self.clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=5)
        self.clf.fit(self.trainArr, self.trainRes)

        # X_train = self.scaler.fit_transform(self.trainArr)
        # self.clf.fit(X_train, self.trainRes)

      



    def run(self,x_test,z_score_threshold):
    # def run(self,x_test,cumulative_threshold): #for the efficiency_run
        x_test['normalized_cap']=x_test['cap']/x_test['cumulative']
        x_test['normalized_capnormalized_substring-cap']=x_test['substring-cap']/x_test['cumulative']
        x_test['normalized_s-o-sCap']=x_test['s-o-sCap']/x_test['cumulative']
        x_test['normalized_all-cap']=x_test['all-cap']/x_test['cumulative']
        x_test['normalized_non-cap']=x_test['non-cap']/x_test['cumulative']
        x_test['normalized_non-discriminative']=x_test['non-discriminative']/x_test['cumulative']




        #setting features
        testArr= x_test.as_matrix(self.cols)
        #print(testArr)
        testRes = x_test.as_matrix(self.colsRes)


        # In[ ]:




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