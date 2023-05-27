#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A8  |  Name : Pratik Pingale  |  Roll No : 19CO056


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('titanic')

dataset.head()


# In[3]:


sns.histplot(dataset['fare'], kde=True, linewidth=0);


# In[4]:


sns.jointplot(x='age', y='fare', data=dataset);

