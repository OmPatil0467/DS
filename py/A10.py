#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A10  |  Name : Pratik Pingale  |  Roll No : 19CO056


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('iris.csv')
df.head()


# ### How many features are there and what are their types (e.g., numeric, nominal)?

# In[3]:


df.info()


# Hence the dataset contains 4 numerical columns and 1 object column

# In[4]:


np.unique(df["Species"])


# In[5]:


df.describe()


# ### Create a histogram for each feature in the dataset.

# In[6]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout = True)

for i in range(4):
    x, y = i // 2, i % 2
    _ = axes[x, y].hist(df[df.columns[i + 1]])
    _ = axes[x, y].set_title(f"Distribution of {df.columns[i + 1][:-2]}")


# ### Create a boxplot for each feature in the dataset.

# In[7]:


data_to_plot = df[df.columns[1:-1]]

fig, axes = plt.subplots(1, figsize=(12,8))
bp = axes.boxplot(data_to_plot)


# If we observe closely for the box 2, interquartile distance is roughly around **0.75** hence the values lying beyond this range of (third quartile + interquartile distance) i.e. roughly around **4.05** will be considered as outliers. Similarly outliers with other boxplots can be found.
