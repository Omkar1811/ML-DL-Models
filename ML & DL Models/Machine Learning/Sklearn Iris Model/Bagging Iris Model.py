#!/usr/bin/env python
# coding: utf-8

# In[36]:


#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 


# In[37]:


iris=datasets.load_iris()


# In[38]:


X=iris.data
Y=iris.target


# In[60]:


X=pd.DataFrame(X)
X


# In[42]:


Y


# In[61]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=len(X) :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(Y,X[column])
    plotnumber+=1
plt.show()


# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size=.30,random_state=355)


# In[46]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# In[49]:


bag_decision = BaggingClassifier(DecisionTreeClassifier()) 


# In[50]:


bag_decision.fit(X_train, y_train)


# In[51]:


bag_decision.score(X_test, y_test)


# In[52]:


bag_decision.score(X_train,y_train)


# In[55]:


from sklearn.metrics import r2_score


# In[57]:


y_pred=bag_decision.predict(X_test)


# In[58]:


r2_score(y_test,y_pred)


# In[ ]:




