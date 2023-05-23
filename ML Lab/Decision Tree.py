#!/usr/bin/env python
# coding: utf-8

# Decision Tree

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_csv("D:\\CIT Semester Notes\\6th sem notes\\Machine Learning\\lab\\DT.csv") 


# In[8]:


x = df.drop(['CustomerID', 'Purchased'], axis = 'columns')
x


# In[9]:


y = df['Purchased']
y


# In[10]:


from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()


# In[11]:


x = x.apply(LabelEncoder().fit_transform)
x


# In[12]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# In[13]:


len(x_train)


# In[14]:


len(x_test)


# In[15]:


x_test


# In[16]:


from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x, y)


# In[17]:


model.predict([[1, 1, 2]])


# In[18]:


model.predict(x_test)


# In[19]:


tree.plot_tree(model)


# In[20]:


fig = plt.figure(figsize=(7,7))
h = tree.plot_tree(model, 
                   feature_names = ['Age', 'Gender', 'Income'], class_names =['0', '1'],
                   filled=True)


# In[ ]:




