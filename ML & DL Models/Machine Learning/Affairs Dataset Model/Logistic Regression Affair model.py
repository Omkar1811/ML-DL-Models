#!/usr/bin/env python
# coding: utf-8

# In[20]:


#importing important libraries
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import statsmodels.api as sm
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler 
from statsmodels.stats.outliers_influence import variance_inflation_factor 


# In[10]:


data=sm.datasets.fair.load_pandas().data


# In[11]:


data=pd.DataFrame(data)
data


# In[12]:


data


# In[16]:


y=data['affairs']
x=data.drop(['affairs'],axis=1)


# In[17]:


x


# In[18]:


y


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1
i=1
for column in x:
    if plotnumber<=len(x) :
            ax = plt.subplot(4,4,plotnumber)
            sns.stripplot(y,x[column])
            plotnumber+=1
plt.show()


# In[27]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(x)


# In[28]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = x.columns

#let's check the values
vif


# In[33]:


Y=[]
for i in y:
    Y.append(round(i))
Y    


# In[36]:


x_train,x_test,y_train,y_test = train_test_split(X_scaled,Y, test_size= 0.25, random_state = 355)


# In[37]:


log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)


# In[38]:


log_reg.score(x_test,y_test)


# In[50]:


y_pred = log_reg.predict(x_test)


# In[40]:


accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[42]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[43]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[44]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[45]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[46]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[47]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[ ]:




