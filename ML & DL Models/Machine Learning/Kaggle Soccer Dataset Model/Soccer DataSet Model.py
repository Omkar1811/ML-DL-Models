#!/usr/bin/env python
# coding: utf-8

# In[100]:


#importing important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV 
from sklearn.preprocessing import StandardScaler 
import sqlite3
from math import sqrt
from sklearn import ensemble


# In[57]:


cnx = sqlite3.connect('database.sqlite')
data = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)


# In[58]:


data


# In[59]:


data.isna().sum()


# In[60]:


from sklearn.preprocessing import LabelEncoder 


# In[61]:


print(data['preferred_foot'].mode()[0])


# In[80]:


le = LabelEncoder() 
data['preferred_foot']=data['preferred_foot'].fillna(data['preferred_foot'].mode()[0])
data['overall_rating']=data['overall_rating'].fillna(data['overall_rating'].mode()[0]) 
data['potential ']=data['potential'].fillna(data['potential'].median()) 
data['attacking_work_rate']=data['attacking_work_rate'].fillna(data['attacking_work_rate'].mode()[0]) 
data['defensive_work_rate']=data['defensive_work_rate'].fillna(data['defensive_work_rate'].mode()[0]) 
data['crossing']=data['crossing'].fillna(data['crossing'].mode()[0]) 
data['finishing']=data['finishing'].fillna(data['finishing'].mode()[0]) 
data['heading_accuracy']=data['heading_accuracy'].fillna(data['heading_accuracy'].mode()[0]) 
data['short_passing']=data['short_passing'].fillna(data['short_passing'].mode()[0]) 
data['volleys']=data['volleys'].fillna(data['volleys'].mode()[0]) 
data['dribbling']=data['dribbling'].fillna(data['dribbling'].mode()[0]) 
data['curve']=data['curve'].fillna(data['curve'].mode()[0]) 
data['free_kick_accuracy']=data['free_kick_accuracy'].fillna(data['free_kick_accuracy'].mode()[0]) 
data['long_passing']=data['long_passing'].fillna(data['long_passing'].mode()[0]) 
data['ball_control']=data['ball_control'].fillna(data['ball_control'].mode()[0]) 
data['acceleration']=data['acceleration'].fillna(data['acceleration'].mode()[0]) 
data['sprint_speed']=data['sprint_speed'].fillna(data['sprint_speed'].mode()[0]) 
data['agility']=data['agility'].fillna(data['agility'].mode()[0]) 
data['reactions']=data['reactions'].fillna(data['reactions'].mode()[0]) 
data['balance']=data['balance'].fillna(data['balance'].mode()[0]) 
data['shot_power']=data['shot_power'].fillna(data['shot_power'].mode()[0]) 
data['jumping']=data['jumping'].fillna(data['jumping'].mode()[0]) 
data['strength']=data['strength'].fillna(data['strength'].mode()[0]) 
data['long_shots']=data['long_shots'].fillna(data['long_shots'].mode()[0]) 
data['aggression']=data['aggression'].fillna(data['aggression'].mode()[0]) 
data['interceptions']=data['interceptions'].fillna(data['interceptions'].mode()[0]) 
data['positioning']=data['positioning'].fillna(data['positioning'].mode()[0]) 
data['vision']=data['vision'].fillna(data['vision'].mode()[0]) 
data['penalties']=data['penalties'].fillna(data['penalties'].mode()[0]) 
data['marking']=data['marking'].fillna(data['marking'].mode()[0]) 
data['standing_tackle']=data['standing_tackle'].fillna(data['standing_tackle'].mode()[0]) 
data['sliding_tackle']=data['sliding_tackle'].fillna(data['sliding_tackle'].mode()[0]) 
data['gk_diving']=data['gk_diving'].fillna(data['gk_diving'].mode()[0]) 
data['gk_handling']=data['gk_handling'].fillna(data['gk_handling'].mode()[0]) 
data['gk_kicking']=data['gk_kicking'].fillna(data['gk_kicking'].mode()[0]) 
data['gk_positioning']=data['gk_positioning'].fillna(data['gk_positioning'].mode()[0]) 
data['gk_reflexes']=data['gk_reflexes'].fillna(data['gk_reflexes'].mode()[0]) 
data['stamina']=data['stamina'].fillna(data['stamina'].mode()[0]) 



# In[87]:


data=data.drop(['potential'],axis=1)


# In[88]:


data.isna().sum()


# In[94]:


data['preferred_foot']= le.fit_transform(data['preferred_foot']) 
data['attacking_work_rate']= le.fit_transform(data['attacking_work_rate']) 
data['defensive_work_rate']= le.fit_transform(data['defensive_work_rate']) 

data


# In[104]:


X=data.drop(['overall_rating','date','id','player_fifa_api_id','player_api_id'],axis=1)
Y=data['overall_rating']


# In[105]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.30, random_state= 355)


# In[106]:


model = ensemble.GradientBoostingRegressor()
model.fit(x_train, y_train)


# In[107]:


model.score(x_train,y_train)


# In[108]:


model.score(x_test,y_test)


# In[109]:


from sklearn.metrics import r2_score


# In[111]:


y_pred=model.predict(x_test)


# In[128]:


r2=r2_score(y_test,y_pred)
r2


# In[124]:


def adj_r2(x,y,r2):
    n = x.shape[0]
    p = x.shape[1]
    adjusted_r2 = 1-(1-r2)*(n-1)/(n-p-1)
    return adjusted_r2


# In[129]:


adj_r2(x_test,y_test,r2)


# In[ ]:




