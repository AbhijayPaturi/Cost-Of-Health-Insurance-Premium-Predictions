#!/usr/bin/env python
# coding: utf-8

# In[5]:


pip install seaborn


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[3]:


### read the dataset 
health_insurance_df = pd.read_csv("../health_insurance_premium_data.csv")


# In[4]:


### Explore the dataset
health_insurance_df.shape


# In[5]:


health_insurance_df.columns


# In[6]:


health_insurance_df.info()


# In[7]:


health_insurance_df.describe()


# In[8]:


# check for null values 
health_insurance_df.isna().sum()


# In[9]:


### feature engineering
# identify the independent and dependent variables
independent_variables = health_insurance_df.loc[:, 'age':'region']
dependent_variables = health_insurance_df.loc[:, 'charges']


# In[10]:


dependent_variables.head()


# In[11]:


independent_variables.head()


# In[12]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False, drop='first')
dummy_encoded_df = pd.DataFrame(onehotencoder.fit_transform(independent_variables[['sex', 'smoker', 'region']]))
onehotencoder.categories_


# In[13]:


dummy_encoded_df.head()


# In[14]:


dummy_encoded_df.columns


# In[15]:


dummy_encoded_df.columns = ["sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]


# In[16]:


dummy_encoded_df.columns


# In[17]:


dummy_encoded_df.head()


# In[18]:


health_insurance_df.head()


# In[19]:


# # del test_df
# health_insurance_df
# # test_df.reset_index(drop=True, inplace=True)
# # dummy_encoded_df.reset_index(drop=True, inplace=True)
# test_df.head()


# In[20]:


health_insurance_df = pd.concat([dummy_encoded_df, health_insurance_df], axis=1)
health_insurance_df.drop(['region', 'sex', 'smoker'], axis=1, inplace=True)
health_insurance_df.head()


# In[21]:


from sklearn.preprocessing import StandardScaler

for num_col in health_insurance_df[['age','bmi']]:
  SS = StandardScaler()
  health_insurance_df[num_col] = SS.fit_transform(health_insurance_df[[num_col]])
health_insurance_df.head()


# In[22]:


# check for multicollinearity
sns.heatmap(health_insurance_df.corr(), annot=True)


# In[23]:


independent_variables = health_insurance_df.loc[:, 'sex_male':'children']
dependent_variables = health_insurance_df.loc[:, 'charges']
independent_variables.head()


# In[24]:


dependent_variables.head()


# In[25]:


### Building the LinearRegression model 
# split the data into training and testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent_variables,dependent_variables,test_size=0.2,random_state=12)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[26]:


# fit the LinearRegression model to the training data
from sklearn.linear_model import LinearRegression
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)


# In[27]:


# predict with the test data
y_pred = linear_regression_model.predict(X_test)


# In[28]:


# identify the coefficients of the linear regression equation
linear_regression_model.coef_


# In[29]:


results_df = pd.DataFrame({'Actual': y_test,'Predicted' : y_pred})
results_df


# In[30]:


from sklearn import metrics
import numpy as np
  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 


# In[31]:


from sklearn.metrics import mean_squared_error
rsme = mean_squared_error(y_test, y_pred, squared=False)
rsme


# In[32]:


normalized_rsme = rsme/(dependent_variables.max()-dependent_variables.min())
normalized_rsme


# In[33]:


# R^2 value
from sklearn.metrics import r2_score
r2_accuracy_score=r2_score(y_test, y_pred)*100
r2_accuracy_score


# In[34]:


# plotting the results
sns.lmplot(x="Actual", y="Predicted", data=results_df, ci=None, line_kws={'color': 'red'})


# In[38]:


# adjusted R2
adjusted_r2 = 1 - (1-linear_regression_model.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
adjusted_r2
# linear_regression_model.score(X_test, y_test)


# In[ ]:




