#!/usr/bin/env python
# coding: utf-8

# In[282]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[283]:


dta=pd.read_csv("ML_DATA.csv")
dta


# In[284]:


dta.info()


# In[285]:


dta['Class'].value_counts()


# In[286]:


convert = {
    "Age": {"Young": 1, "Middle": 0, "Old": 2},
    "Has_Job": {"False": 0, "True": 1},
    "Own_House": {"False": 0, "True": 1},
    "Credit_Rating": {"Good": 0, "fair": 1, "excellent": 2},
}


# In[287]:


dta.replace(convert, inplace=True)


# In[290]:


dta["Has_Job"] = dta["Has_Job"].astype(int)
dta["Own_House"] = dta["Own_House"].astype(int)


# In[291]:


dta


# In[292]:


xc=['Age','Has_Job','Own_House','Credit_Rating']
y=['No','Yes']
all_inputs = dta[xc]
all_classes = dta["Class"]


# In[293]:


(x_train,x_test,y_train,y_test)=train_test_split(all_inputs,all_classes,train_size=0.7 ) #random_state=1


# In[294]:


clf =DecisionTreeClassifier(random_state=0)


# In[295]:


clf.fit(x_train,y_train)


# In[296]:


score=clf.score(x_test,y_test)
print(score)


# In[ ]:




