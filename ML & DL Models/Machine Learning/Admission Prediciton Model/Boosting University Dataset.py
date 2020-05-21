#!/usr/bin/env python
# coding: utf-8

# In[40]:


#importing important libraries
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[5]:


data = pd.read_csv("C:\\Users\\user\\Documents\\Datasets\\Admission_Prediction.csv") # Reading the Data
data.head()


# In[6]:


data.isna().sum()


# In[11]:


data['GRE Score'].fillna(data['GRE Score'].mean(),inplace=True)
data['TOEFL Score'].fillna(data['TOEFL Score'].mean(),inplace=True)
data['University Rating'].fillna(data['University Rating'].mode()[0],inplace=True)


# In[12]:


data.isna().sum()


# In[13]:


x=data.drop(['Chance of Admit','Serial No.'],axis=1)
y=data['Chance of Admit']


# In[51]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1

for column in x:
    if plotnumber<=len(x) :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(y,x[column])
    plotnumber+=1
plt.show()


# In[14]:


x


# In[15]:


y


# In[19]:


scaler=StandardScaler()
scaled_data=scaler.fit_transform(x)


# In[20]:


train_x,test_x,train_y,test_y=train_test_split(scaled_data,y,test_size=0.3,random_state=355)


# In[32]:


model = ensemble.GradientBoostingRegressor()
model.fit(train_x, train_y)


# In[33]:


model.score(train_x,train_y)


# In[34]:


model.score(test_x,test_y)


# In[35]:


from sklearn.metrics import r2_score


# In[38]:


y_pred=model.predict(test_x)


# In[49]:


r2=r2_score(test_y,y_pred)
r2


# In[45]:


def adj_r2(x,y,r2):
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[50]:


adj_r2(test_x,test_y,r2)


# In[ ]:




