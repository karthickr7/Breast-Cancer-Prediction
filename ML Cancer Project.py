#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Prediction using SVM

# # 1. Import necessary packages

# In[3]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score


# # 2. Data Exploration

# In[4]:


# Load dataset
c=pd.read_csv("cancer.csv")


# In[5]:


c


# In[6]:


# Dimension of dataset
c.shape


# In[7]:


c.info()


# In[222]:


c.dtypes.value_counts()


# In[210]:


c.duplicated()


# In[8]:


c.isnull().sum()


# In[9]:


c.head()


# In[10]:


c.tail()


# In[11]:


c.describe()


# In[12]:


c.corr()


# In[13]:


plt.figure(figsize=(25,20))
sns.heatmap(c.corr(), annot=True)


# In[223]:


c.hist()
plt.show()


# # 3. Data Preprocessing

# In[107]:


X=c.drop(['id','Unnamed: 32','diagnosis'], axis=1)


# In[108]:


X.shape


# In[109]:


X.head()


# In[116]:


Y=c['diagnosis']
Y.head()


# In[117]:


print(type(Y))


# In[118]:


Y.value_counts()['M']


# In[119]:


Y.value_counts()['B']


# In[120]:


np.unique(Y)


# In[121]:


np.unique(Y, return_counts=True)


# In[140]:


Y = LabelEncoder().fit_transform(Y) # 1 ='Malignant' and 0='Benign'
# fit_transform --> Fit label encoder and return encoded labels 


# In[142]:


Y


# In[143]:


print(type(Y))


# In[144]:


Y.shape


# In[145]:


np.unique(Y)


# In[146]:


np.unique(Y, return_counts=True)


# # 4. Model Training using SVM classifier

# In[171]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25) # Train:Test = 80:20


# In[172]:


print('Dimension of X train:', X_train.shape)
print('Dimension of X test:', X_test.shape)
print('Dimension of Y train:', Y_train.shape)
print('Dimension of Y test:', Y_test.shape)


# In[173]:


svc_model = SVC()
svc_model.fit(X_train, Y_train)


# In[174]:


Y_predict = svc_model.predict(X_test)
Confusion_Matrix_1 = confusion_matrix(Y_test, Y_predict)
s=sns.heatmap(Confusion_Matrix_1 , annot=True)
s.set(xlabel='Predicted values', ylabel='Actual values') # 1 ='Malignant' and 0='Benign'


# In[175]:


print(classification_report(Y_test, Y_predict))


# In[176]:


Accuracy_score_1 = accuracy_score (Y_test, Y_predict)
print(Accuracy_score_1)


# In[189]:


specificity_1= Confusion_matrix_2[1,1]/(Confusion_matrix_2[1,0]+Confusion_matrix_2[1,1])
print('Specificity score:', specificity_1)  # To correctly classify as Cancer free


# In[213]:


auc_score_1 = roc_auc_score(Y_test, Y_predict)
print('AUC-ROC Score:',auc_score_1)


# # 5. Model Tuning

# In[177]:


#find best hyper parameters using Grid Search CV
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.001], 'kernel':['rbf']}
grid = GridSearchCV(SVC(),param_grid,verbose = 4)
grid.fit(X_train,Y_train)


# In[178]:


grid.best_params_


# In[179]:


grid.best_estimator_


# In[180]:


grid_predictions = grid.predict(X_test)
Confusion_matrix_2 = confusion_matrix(Y_test,grid_predictions)
r=sns.heatmap(Confusion_matrix_2, annot=True)
r.set(xlabel='Predicted values', ylabel='Actual values') # 1 ='Malignant' and 0='Benign'


# In[181]:


print(classification_report(Y_test,grid_predictions))


# In[182]:


Accuracy_score_2 = accuracy_score (Y_test, grid_predictions)
print(Accuracy_score_2)


# In[187]:


specificity_2= Confusion_matrix_2[0,0]/(Confusion_matrix_2[0,1]+Confusion_matrix_2[0,0])
print('Specificity score:', specificity_2) # To correctly classify as Cancer free


# In[212]:


auc_score2 = roc_auc_score(Y_test, grid_predictions)
print('AUC-ROC Score:',auc_score2)


# # 6. Result Comparision

# In[183]:


print('Accuracy before Tuning: ',Accuracy_score_1)
print('Accuracy after Tuning: ',Accuracy_score_2)


# In[214]:


print('AUC-ROC Score before Tuning:',auc_score1)
print('AUC-ROC Score after Tuning:',auc_score2)

