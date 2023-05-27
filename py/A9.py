#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A9  |  Name : Pratik Pingale  |  Roll No : 19CO056


# In[2]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

dataset = sns.load_dataset('titanic')

dataset.head()


# In[3]:


sns.boxplot(x='sex', y='age', data=dataset, hue="survived");


# If we want to see the box plots of forage of passengers of both genders, along with the information about whether or not they survived, we can pass the **survived** as value to the **hue** parameter.
# 
# We can also see the distribution of the passengers who survived. For instance, we can see that among the male passengers, on average more younger people survived as compared to the older ones. Similarly, we can see that the variation among the age of female passengers who did not survive is much greater than the age of the surviving female passengers.
