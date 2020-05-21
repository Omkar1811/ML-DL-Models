#!/usr/bin/env python
# coding: utf-8

# In[26]:


#importing important libraries
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[2]:


df=pd.read_csv("https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv")
df


# In[3]:


df.isna().sum()


# In[4]:


df['Age'].fillna(round(df['Age'].mean()),inplace=True)


# In[5]:


x=df.drop(columns=['Cabin','Embarked','PassengerId','Name','Ticket'])
x


# In[6]:


x.isna().sum()


# In[7]:


x["Sex"].replace({"male": 1, "female": 0}, inplace=True)
Y=x['Survived']
X=x.drop(columns=['Survived'])


# In[8]:


X


# In[24]:


Y


# In[9]:


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


# In[10]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30)


# In[11]:


from sklearn.tree import DecisionTreeClassifier


# In[12]:


clf = DecisionTreeClassifier()


# In[13]:


clf.fit(x_train,y_train)


# In[14]:


clf.score(x_train,y_train)


# In[15]:


py_pred = clf.predict(x_test)


# In[16]:


clf.score(x_test,y_test)


# In[28]:


accuracy_score(y_test,clf.predict(x_test))


# In[33]:


y_pred = clf.predict(x_test)
y_pred


# In[34]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[35]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[36]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[37]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[38]:


Recall = true_positive/(true_positive+false_negative)
Recall


# In[39]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[41]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[42]:


grid_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,32,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']
    
}


# In[43]:


grid_search = GridSearchCV(estimator=clf,
                     param_grid=grid_param,
                     cv=5,
                    n_jobs =-1)


# In[44]:


grid_search.fit(x_train,y_train)


# In[45]:


best_parameters = grid_search.best_params_
print(best_parameters)


# In[46]:


grid_search.best_score_


# In[47]:


clf = DecisionTreeClassifier(criterion = 'gini', max_depth =23, min_samples_leaf= 4, min_samples_split= 6, splitter ='random')
clf.fit(x_train,y_train)


# In[48]:


clf.score(x_test,y_test)


# In[49]:


accuracy_score(y_test,clf.predict(x_test))


# In[50]:


y_pred = clf.predict(x_test)


# In[51]:


conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[52]:


true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]


# In[53]:


Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[54]:


Precision = true_positive/(true_positive+false_positive)
Precision


# In[55]:


F1_Score = 2*(Recall * Precision) / (Recall + Precision)
F1_Score


# In[56]:


auc = roc_auc_score(y_test, y_pred)
auc


# In[ ]:




