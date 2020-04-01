#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[2]:


d=pd.read_csv('Social_Network_Ads.csv')


# In[3]:


d.head()


# In[4]:


d.info()


# In[8]:


X=d.iloc[:,[2,3]].values
y=d.iloc[:,4].values


# In[9]:


y


# In[6]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[16]:


model=LogisticRegression(random_state=0)


# In[17]:


model.fit(X_train,y_train)


# In[19]:


y_pred=model.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score


# In[22]:


accuracy_score(y_test,y_pred)*100


# In[ ]:



