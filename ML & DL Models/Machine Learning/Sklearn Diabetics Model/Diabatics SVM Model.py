#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
#importing necessary libraries
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[27]:


data = pd.read_csv("C:\\Users\\user\\Documents\\Datasets\\diabetes.csv") # Reading the Data
data.head()


# In[30]:


data=pd.DataFrame(data)
data


# In[62]:


data.isna().sum()


# In[34]:


X=data.drop(data['Outcome'])
Y=data['Outcome']


# In[110]:


plt.figure(figsize=(20,15), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=len(X) :
        ax = plt.subplot(3,3,plotnumber)
        sns.stripplot(Y,X[column])
    plotnumber+=1
plt.show()


# In[37]:


X


# In[40]:


Y


# In[39]:


y=[]
for i in range(766):
    y.append(Y[i])
y


# In[36]:


data.columns


# In[63]:


scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)


# In[64]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# In[65]:


train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.33, random_state=355)


# In[66]:


model=SVC()
model.fit(train_x,train_y)


# In[67]:


model.predict(test_x)


# In[68]:


accuracy_score(test_y,model.predict(test_x))


# In[82]:


param_grid={'C':[i for i in range(1,300)],'gamma':[1,0.5,0.1,0.01,0.001],'decision_function_shape':['ovo', 'ovr']}


# In[83]:


grid= GridSearchCV(SVC(),param_grid, verbose=3, n_jobs=-1)


# In[84]:


grid.fit(train_x,train_y)


# In[85]:


grid.best_params_


# In[86]:


model_new=SVC(C=1,decision_function_shape='ovo', gamma=1)
model_new.fit(train_x,train_y)


# In[87]:


accuracy_score(test_y,model_new.predict(test_x))


# In[100]:


y_pred = model.predict(test_x)


# In[101]:


conf_mat = confusion_matrix(test_y,y_pred)
conf_mat


# In[102]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[103]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[104]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[105]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[106]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[108]:


auc = roc_auc_score(test_y, y_pred)
auc


# In[ ]:




