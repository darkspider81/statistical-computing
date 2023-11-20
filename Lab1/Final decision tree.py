#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[2]:


data="ML_DATA.csv"
dta=pd.read_csv(data,sep= ",", header=0)


# In[3]:


dta.head()


# In[25]:


dta["Has_Job"] = dta["Has_Job"].astype(int)
dta["Own_House"] = dta["Own_House"].astype(int)
dta["Credit_Rating"]=dta["Credit_Rating"].astype(int)
dta["Has_Job"]=dta["Has_Job"].astype(int)
dta["Age"]=dta["Age"].astype(int)


# In[26]:


dta


# In[42]:


X=dta.values[:,0:3]
Y=dta.values[:,4]
X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=0.7, random_state=1)
clf_entropy=DecisionTreeClassifier(criterion = "entropy",random_state= 1)
clf_entropy.fit(X_train,y_train)


# In[43]:


y_pred_en = clf_entropy.predict(X_test)
y_pred_en


# In[46]:


print ("Accuracy is ",accuracy_score(y_test,y_pred_en)*100)


# In[ ]:




