#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
plt.rcParams['figure.figsize']=(20.0,10.0)


# In[2]:


d=pd.read_csv("headbrain.csv")


# In[3]:


d.head()


# In[4]:


X=d['Head Size(cm^3)'].values
y=d['Brain Weight(grams)'].values


# In[6]:


mean_x=np.mean(X)
mean_y=np.mean(y)
n=len(X)
num=0
dem=0
for i in range(n):
    num+=(X[i]-mean_x)*(y[i]-mean_y)
    dem+=(X[i]-mean_x)**2
m=num/dem
c=mean_y - (m * mean_x)

print(m,c)


# In[14]:


max_x=np.max(X)+100
min_x=np.min(X)-100

x=np.linspace(min_x,max_x,1000)
Y = m * x + c

plt.plot(x,Y,color="red",label="Regression Line")

plt.scatter(X,y,c="blue",label="Scatter Plot")


plt.xlabel('Head Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.show()

#calculate the r-squared error
ss_t=0
ss_r=0
for i in range(n):
    y_pred=m*X[i]+c
    ss_t+=(y[i]-mean_y)**2
    ss_r+=(y[i]-y_pred)**2
r2=1-(ss_r/ss_t)

print(r2)
