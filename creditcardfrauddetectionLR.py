#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


data=pd.read_csv("credit.csv")


# In[53]:


data.head()


# In[54]:


data.tail()


# In[20]:


counts = data.Class.value_counts()
fraud = data.loc[data['Class'] == 1]
normal = data.loc[data['Class'] == 0]


# In[15]:


fraud.count()


# In[16]:


len(fraud)


# In[17]:


len(normal)


# In[21]:


plt.figure(figsize=(8,6))
sns.barplot(x=counts.index, y=counts)
plt.title('Count of Fraud vs Normal Transactions')
plt.ylabel('Count')
plt.xlabel('Class (0:Normal, 1:Fraud)')


# In[60]:


corrmat = data.corr()
fig = plt.figure(figsize = (12, 12))
sns.heatmap(corrmat, vmax = .8, square = True) 
plt.show()


# In[4]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# In[5]:


X = data.iloc[:,:-1]
y = data['Class']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.35)


# In[9]:


clf = linear_model.LogisticRegression(C=1e5)


# In[10]:


clf.fit(X_train, y_train)


# In[24]:


y_pred = np.array(clf.predict(X_test))
y = np.array(y_test)


# In[25]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[29]:


print(confusion_matrix(y, y_pred))


# In[30]:


print(accuracy_score(y, y_pred))


# In[31]:


print(classification_report(y, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




