#!/usr/bin/env python
# coding: utf-8


"""Analysed by Manendra Nath Shukla """



import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import math

tit_data=pd.read_csv("train.csv")





# In[6]:


tit_data.head()


# In[7]:


print("Total no of Passengers in the original data :"+str(len(tit_data.index)))


#  ## Analyzing Data

# In[2]:


sns.countplot(x='Survived',data=tit_data)


# In[8]:


sns.countplot(x="Survived",hue="Sex", data =tit_data)


# In[9]:


sns.countplot(x="Survived",hue="Pclass", data =tit_data)


# In[10]:


tit_data["Age"].plot.hist()


# In[14]:


tit_data["Fare"].plot.hist(bins=20,figsize=(10,5))


# In[15]:


tit_data.info()


# In[16]:


sns.countplot(x="SibSp",data=tit_data)


# In[17]:


sns.countplot(x="Parch",data=tit_data)


# ## Data Wrangling

# In[18]:


tit_data.isnull()


# In[19]:


tit_data.isnull().sum()


# In[23]:


sns.heatmap(tit_data.isnull(),yticklabels=False, cmap="viridis")


# In[24]:


sns.boxplot(x="Pclass",y="Age",data=tit_data)


# In[25]:


tit_data.head()


# In[26]:


tit_data.drop("Cabin",axis=1,inplace=True)


# In[27]:


tit_data.head()


# In[28]:


tit_data.dropna(inplace=True)


# In[32]:


sns.heatmap(tit_data.isnull(),cbar=False)


# In[33]:


tit_data.isnull().sum()


# In[38]:


sex=pd.get_dummies(tit_data["Sex"],drop_first=True)
sex.head()


# In[41]:


embark=pd.get_dummies(tit_data["Embarked"],drop_first=True)
embark.head()


# In[42]:


Pcl=pd.get_dummies(tit_data["Pclass"],drop_first=True)
Pcl.head()


# In[44]:


tit_data=pd.concat([tit_data,sex,embark,Pcl],axis=1)
tit_data.head()


# In[45]:


tit_data.drop(['Sex','Embarked','PassengerId','Name','Ticket'],axis=1,inplace=True)


# In[46]:


tit_data.head()


# In[47]:


tit_data.drop('Pclass',axis=1,inplace=True)


# In[48]:


tit_data.head()


# ## Train Data

# In[50]:


X=tit_data.drop("Survived",axis=1)
y = tit_data["Survived"]


# In[54]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[ ]:





# In[52]:


model=LogisticRegression()


# In[55]:


model.fit(X_train,y_train)


# In[56]:


y_pred=model.predict(X_test)


# In[61]:


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score


# In[58]:


classification_report(y_test,y_pred)


# In[60]:


confusion_matrix(y_test,y_pred)


# In[63]:


accuracy_score(y_test,y_pred)*100


# In[ ]:





# In[ ]:




