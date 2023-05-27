#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A1     |     Name : Pratik Pingale     |     Roll No : 19CO056


# In[2]:


# subtask 1 - importing libraries
import pandas as pd
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


# subtask 2 - Dataset url
# Name - COVID -19 Global Reports early March 2022
# URL - https://www.kaggle.com/danielfesalbon/covid-19-global-reports-early-march-2022
# local machine relative address - /covid_19_clean_complete_2022.csv/


# In[4]:


#subtask 3 - Loading dataset
df = pd.read_csv("covid_19_clean_complete_2022.csv")
df.drop('Province/State', axis=1, inplace=True)
df.head()


# In[5]:


# subtask 4 - data preprocessing - detecting NaN values and using describe() function
df.isna().any()
df.describe()


# In[6]:


# shape of dataset (dimensions)
df.shape


# In[7]:


# subtask 5 - data formatting and normalization
df.dtypes


# In[8]:


df['Date'] = pd.to_datetime(df['Date'])
df['Country/Region'] = df['Country/Region'].astype('string')
df.dtypes


# In[9]:


# subtask 6 - handling categorical values
# dropping the categorical variable column
# df['new_col'] = df['some_col'].map({'value_1': 1, 'value_2': 2})
df = df.drop(['WHO Region','Country/Region'], axis=1)
df.head()

