#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
#importing necessary libraries
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[2]:


data = pd.read_csv("C:\\Users\\user\\Documents\\Datasets\\winequality.csv") # Reading the Data
data.head()


# In[3]:


data.describe()


# In[5]:


data.isna().sum()


# In[7]:


X = data.drop(columns = ['quality'])
y = data['quality']


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1
i=1
for column in X:
    if plotnumber<=len(X) :
            ax = plt.subplot(4,4,plotnumber)
            sns.stripplot(y,X[column])
            plotnumber+=1
plt.show()


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(X,y, test_size= 42)


# In[9]:


knn = KNeighborsClassifier()
knn.fit(x_train,y_train)


# In[10]:


knn.score(x_train,y_train)


# In[12]:


y_pred = knn.predict(x_test)
print("The accuracy score is : ", accuracy_score(y_test,y_pred))


# In[39]:


param_grid = { 'algorithm' : ['auto','ball_tree', 'kd_tree', 'brute'],
               'leaf_size' : [i for i in range(2,20)],
               'n_neighbors' : [i for i in range(2,30)]
              }


# In[40]:


gridsearch = GridSearchCV(knn, param_grid,verbose=3)


# In[41]:


gridsearch.fit(x_train,y_train)


# In[42]:


gridsearch.best_params_


# In[43]:


knn = KNeighborsClassifier(algorithm = 'auto', leaf_size =2, n_neighbors =27)


# In[44]:


knn.fit(x_train,y_train)


# In[45]:


knn.score(x_train,y_train)


# In[46]:


knn.score(x_test,y_test)


# In[ ]:




