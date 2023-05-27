#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A2     |     Name : Pratik Pingale     |     Roll No : 19CO056


# ### Importing pandas and numpy libs

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# ### Reading the dataset and loading into pandas dataframe

# In[3]:


df = pd.read_csv("Academic-Performance-Dataset.csv")
df


# In[4]:


df.shape


# In[5]:


df.dtypes


# # Handle the Missing value

# ### Handle the Missing value

# In[6]:


df.isna().sum()


# ### Make a list of column having missing value

# In[7]:


cols_with_na = []
for col in df.columns:
    if df[col].isna().any():
        cols_with_na.append(col)

cols_with_na


# ### Fill the missing value using mean for float and int datatypes and for other forword fill

# In[8]:


for col in cols_with_na:
    col_dt = df[col].dtypes
    if (col_dt == 'int64' or col_dt == 'float64'):
        outliers = (df[col] < 0) | (100 < df[col])
        df.loc[outliers, col] = np.nan
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(method='ffill')
df


# ### Correction in Total Marks, Percentage after filling missing value

# In[9]:


df['Total Marks']=df['Phy_marks']+df['Che_marks']+df['EM1_marks']+df['PPS_marks']+df['SME_marks']
df['Percentage']=df['Total Marks']/5

df


# # Outliers Detection

# In[10]:


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9, 6)
df_list = ['Attendence', 'Phy_marks', 'Che_marks', 'EM1_marks', 'PPS_marks', 'SME_marks']
fig, axes = plt.subplots(2, 3)
fig.set_dpi(120)

count=0
for r in range(2):
    for c in range(3):
        _ = df[df_list[count]].plot(kind = 'box', ax=axes[r,c])
        count+=1


# ### Removal of Outliers from Che_marks

# In[11]:


Q1 = df['Che_marks'].quantile(0.25)
Q3 = df['Che_marks'].quantile(0.75)
IQR = Q3 - Q1

Lower_limit = Q1 - 1.5 * IQR
Upper_limit = Q3 + 1.5 * IQR

print(f'Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}, Lower_limit = {Lower_limit}, Upper_limit = {Upper_limit}')


# In[12]:


df[(df['Che_marks'] < Lower_limit) | (df['Che_marks'] > Upper_limit)]


# # Convert Into Normal Distribution

# ### BINNING USING FREQUENy

# In[13]:


def BinningFunction(column, cut_points, labels = None) :
    break_points=[column.min()] + cut_points + [column.max( )]
    print('Gradding According to percentage \n>60 = F \n60-70 = B \n70-80 = A\n80-100 = O')
    return pd.cut(column, bins=break_points, labels=labels, include_lowest=True)


# In[14]:


cut_points=[60, 70, 80]
labels=['F', 'B', 'A', 'O']
df['Grade']=BinningFunction(df['Percentage'], cut_points, labels)

df

