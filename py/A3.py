#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A3     |     Name : Pratik Pingale     |     Roll No : 19CO056


# ### Importing pandas and numpy libs

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### Reading the dataset and loading into pandas dataframe

# In[3]:


df = pd.read_csv("iris.csv")
df.head()


# ### Basic statistical details of Iris dataset

# In[4]:


'Iris-setosa'
setosa = df['Species'] == 'Iris-setosa'
df[setosa].describe()
'Iris-versicolor'
versicolor = df['Species'] == 'Iris-versicolor'
df[versicolor].describe()
'Iris-virginica'
virginica = df['Species'] == 'Iris-virginica'
df[virginica].describe()


# In[5]:


df.dtypes
df.dtypes.value_counts()

