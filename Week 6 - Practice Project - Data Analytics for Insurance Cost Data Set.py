#!/usr/bin/env python
# coding: utf-8

# # Practice Project - Data Analytics for Insurance Cost Data Set
# ## Setup

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


# In[5]:


filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)


# In[6]:


df.head()


#  

# # Task 1: Import the dataset

# Import the dataset into a pandas dataframe. Note that three are currently no headers in the CSV file.
# Print the first 10 rows.

# In[21]:


df = pd.read_csv(filepath,header=None)
df.head(10)


# Add the headers 'age','gender','bmi','no_of_children','smoker','region','charges'

# In[22]:


headers = ['age','gender','bmi','no_of_children','smoker','region','charges']
df.columns = headers


# In[23]:


# Replace '?' entries with 'NaN' values.
df.replace('?',np.nan,inplace=True)


# In[31]:


df.head()


#  

# # Task 2: Data Wrangling

# Use dataframe.info() to identify the columns that have some NaN information. (Null)

# In[27]:


df.info()


# Handle missing data:
# 
# - For continuous attributes (e.g., age), replace missing values with the mean.
# - For categorical attributes (e.g., smoker), replace missing values with the most frequent value.
# - Update the data types of the respective columns.
# - Verify the update using `df.info()`.
# 

# In[40]:


# smoker is a categorical attribute, replace with the most frequent entry
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan,is_smoker,inplace=True)

# age is a continuous variable, replace with mean age
mean_age = df['age'].astype(float).mean(axis=0)
df['age'].replace(np.nan,mean_age,inplace=True)

# Update the data types
df[['age','smoker']] = df[['age','smoker']].astype('int')

print(df.info())


# Also note, that the `charges` column has values which are more than 2 decimal places long. Update the `charges` column such that all values are rounded to nearest 2 decimal places. Verify conversion by printing the first 5 values of the updated dataframe.
# 

# In[43]:


df[['charges']] = np.round(df[['charges']],2)
print(df.head())


#  

# # Task 3: Explanatory Data Analysis (EDA)

# Implement the regression plot for charges with respect to bmi.

# In[47]:


sns.regplot(x='bmi',y='charges',data=df,line_kws={"color":"red"})
plt.ylim(0,)


# Implement the box plot for charges with respect to smoker

# In[51]:


sns.boxplot(x='smoker',y='charges',data=df)
plt.ylim(0,)


# Print the correlation matrix for the dataset.

# In[53]:


df.corr()


#  

# # Task 4: Model Development

# Fit a linear regression model that may be used to predict the charges value, just by using the smoker attribute of the dataset. Print the R^2 score of this model.

# In[59]:


lr = LinearRegression()
X = df[['smoker']]
Y = df['charges']
lr.fit(X,Y)
print("The R-squared is:",lr.score(X,Y))


# Fit a linear regression model that may be used to predict the charges value, just by using all other attributes of the dataset. Print the R^2 score of this model. You should see an improvement in the performance.

# In[65]:


Z = df[['age','gender','bmi','no_of_children','smoker','region']]
lr.fit(Z,Y)
print("R-squared is better with an score of:",lr.score(Z,Y))


# Create a training pipeline that uses `StandardScaler()`, `PolynomialFeatures()` and `LinearRegression()` to create a model that can predict the `charges` value using all the other attributes of the dataset. There should be even further improvement in the performance.
# 

# In[72]:


Input=[('scale',StandardScaler()),('polynomial',PolynomialFeatures()),('model',LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
print(r2_score(Y,ypipe))


#  

# # Task 5: Model Refinement

# Split the data into training and testing subsets, assuming that 20% of the data will be reserved for testing.

# In[76]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(Z,Y,train_size=0.2,random_state=0)


# Initialize a Ridge regressor that used hyperparameter $ \alpha = 0.1 $. Fit the model using training data subset. Print the $ R^2 $ score for the testing data.

# In[80]:


# x_train, x_test, y_train, y_test hold same values as in previous cells
RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
yhat = RidgeModel.predict(x_test)
print(r2_score(y_test,yhat))


# Apply polynomial transformation to the training parameters with degree=2. Use this transformed feature set to fit the same regression model, as above, using the training subset. Print the $ R^2 $ score for the testing subset.
# 

# In[83]:


# x_train, x_test, y_train, y_test hold same values as in previous cells
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr,y_train)
y_hat=RidgeModel.predict(x_test_pr)
print(r2_score(y_test,y_hat))

