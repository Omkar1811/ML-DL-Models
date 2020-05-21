#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 


# In[3]:


boston = datasets.load_boston()


# In[4]:


features = pd.DataFrame(boston.data,columns=boston.feature_names)
target=boston.target


# In[5]:


features


# In[6]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() 
X= scaler.fit_transform(features)


# In[7]:


X


# In[8]:


y=np.round(target)


# In[13]:


y=np.asarray(y)
y


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1

for column in features:
    if plotnumber<=len(features) :
        ax = plt.subplot(4,4,plotnumber)
        sns.stripplot(target,features[column])
    plotnumber+=1
plt.show()


# In[15]:


from sklearn.ensemble.forest import RandomForestRegressor


# In[16]:


rand_clf=RandomForestRegressor()


# In[272]:


x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.30)


# In[273]:


rand_clf.fit(x_train,y_train)


# In[274]:


rand_clf.score(x_train,y_train)


# In[275]:


rand_clf.score(x_test,y_test)

