#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Assignment - A4  |  Name : Pratik Pingale  |  Roll No : 19CO056


# # Boston Housing with Linear Regression
# 
# **With this data our objective is create a model using linear regression to predict the houses price**
# 
# The data contains the following columns:
# 
# * **CRIM**:  per capita crime rate by town.
# * **ZN**:    proportion of residential land zoned for lots over 25,000 sq.ft.
# * **INDUS**: proportion of non-retail business acres per town.
# * **CHAS**:  Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# * **NOX**:   nitrogen oxides concentration (parts per 10 million).
# * **RM**:    average number of rooms per dwelling.
# * **AGE**:   proportion of owner-occupied units built prior to 1940.
# * **DIS**:   weighted mean of distances to five Boston employment centres.
# * **RAD**:   index of accessibility to radial highways.
# * **TAX**:   full-value property-TAX rate per $10,000.
# * **PTRATIO**:pupil-teacher ratio by town
# * **BLACK**:  1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# * **LSTAT**:  lower status of the population (percent).
# * **MEDV**:   median value of owner-occupied homes in $$1000s

# **Prepare our enviroment**

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"


# In[3]:


# Importing DataSet and take a look at Data
Boston = pd.read_csv("boston.csv")
Boston.head()


# In[4]:


Boston.info()
Boston.describe()


# In[5]:


Boston.plot.scatter('RM', 'MEDV', figsize=(6, 6));


# In this plot its clearly to see a linear pattern. Wheter more average number of rooms per dwelling, more expensive the median value is.

# In[6]:


plt.subplots(figsize=(10,8))
sns.heatmap(Boston.corr(), cmap = 'coolwarm', annot = True, fmt = '.1f');


# At this heatmap plot, we can do our analysis better than the pairplot.
# 
# Lets focus at the last line, where y = MEDV:
# 
# When shades of Blue: the more Blue color is on X axis, smaller the MEDV. Negative correlation                           
# When light colors: those variables at axis x and y, they dont have any relation. Zero correlation                               
# When shades of Red : the more Red color is on X axis, higher the MEDV. Positive correlation

# # Trainning Linear Regression Model
# **Define X and Y**
# 
# X: Varibles named as predictors, independent variables, features.                                                               
# Y: Variable named as response or dependent variable

# In[7]:


X = Boston[Boston.columns[:-1]]
Y = Boston['MEDV']


# **Import sklearn librarys:**    
# train_test_split, to split our data in two DF, one for build a model and other to validate.                                     
# LinearRegression, to apply the linear regression.

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


# In[9]:


# Split DataSet
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
sc_X = StandardScaler()
X_train_ = sc_X.fit_transform(X_train)
X_test_ = sc_X.transform(X_test)


# In[10]:


print(f'Train Dataset Size - X: {X_train.shape}, Y: {Y_train.shape}')
print(f'Test  Dataset Size - X: {X_test.shape}, Y: {Y_test.shape}')


# In[11]:


# Model Building
lm = LinearRegression()
lm.fit(X_train_, Y_train)
predictions = lm.predict(X_test_)


# In[12]:


# Model Visualization
plt.figure(figsize=(6, 6));
plt.scatter(Y_test, predictions);
plt.xlabel('Y Test');
plt.ylabel('Predicted Y');
plt.title('Test vs Prediction');


# In[13]:


plt.figure(figsize=(6, 6));
sns.regplot(x = X_test['RM'], y = predictions, scatter_kws={'s':5});
plt.scatter(X_test['RM'], Y_test, marker = '+');
plt.xlabel('Average number of rooms per dwelling');
plt.ylabel('Median value of owner-occupied homes');
plt.title('Regression Line Tracing');


# In[14]:


from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, predictions))
print('Mean Square Error:', metrics.mean_squared_error(Y_test, predictions))
print('Root Mean Square Error:', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))


# In[15]:


# Model Coefficients
coefficients = pd.DataFrame(lm.coef_.round(2), X.columns)
coefficients.columns = ['Coefficients']
coefficients


# How to interpret those coefficients:
#     they are in function of MEDV, so 
#     
#     for one unit that NOX increase, the house value decrease 'NOX'*1000 (Negative correlation) money unit.
#     for one unit that RM increase, the house value increase 'RM'*1000 (Positive correlation) money unit.
# 
# *1000 because the MEDV is in 1000
# and this apply to the other variables/coefficients.
#     
