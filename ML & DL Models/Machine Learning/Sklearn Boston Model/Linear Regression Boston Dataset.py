#!/usr/bin/env python
# coding: utf-8

# In[3]:


#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 


# In[4]:


boston=datasets.load_boston()


# In[5]:


features = pd.DataFrame(boston.data,columns=boston.feature_names)
target=boston.target


# In[6]:


features


# In[7]:


target


# In[8]:


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


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


x_train,x_test,y_train,y_test = train_test_split(features,target,test_size = 0.30,random_state=355)


# In[9]:


lm = LinearRegression(n_jobs=-1,normalize=True)


# In[10]:


lm.fit(x_train,y_train)


# In[11]:


lm.score(x_test,y_test)


# In[12]:


from sklearn.metrics import r2_score


# In[13]:


y_pred=lm.predict(x_test)


# In[14]:


r2_score(y_test,y_pred)


# In[15]:


from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression


# In[16]:


lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)
lasscv.fit(x_train, y_train)


# In[17]:


alpha = lasscv.alpha_
alpha


# In[18]:


lasso_reg = Lasso(alpha)
lasso_reg.fit(x_train, y_train)


# In[19]:


lasso_reg.score(x_test, y_test)


# In[20]:


alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train, y_train)


# In[21]:


ridgecv.alpha_


# In[22]:


ridge_model = Ridge(alpha=ridgecv.alpha_)
ridge_model.fit(x_train, y_train)


# In[23]:


ridge_model.score(x_test, y_test)


# In[24]:


elasticCV = ElasticNetCV(alphas = None, cv =10)

elasticCV.fit(x_train, y_train)


# In[25]:


elasticCV.alpha_


# In[26]:


elasticCV.l1_ratio


# In[27]:


elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.2)
elasticnet_reg.fit(x_train, y_train)


# In[28]:


elasticnet_reg.score(x_test, y_test)

