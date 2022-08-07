#!/usr/bin/env python
# coding: utf-8

# In[219]:


pip install seaborn


# In[220]:


import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# In[221]:


### read the dataset 
health_insurance_df = pd.read_csv("../health_insurance_premium_data.csv")


# In[222]:


### Explore the dataset (exploratory data analysis/EDA)
health_insurance_df.shape


# In[223]:


health_insurance_df.columns


# In[224]:


health_insurance_df.info()


# In[225]:


health_insurance_df.describe()


# In[226]:


# check for null values 
health_insurance_df.isna().sum()


# In[227]:


# Age feature
age_descriptive_stats = health_insurance_df['age'].describe()
age_descriptive_stats


# In[228]:


health_insurance_df['age'].value_counts().sort_index()


# In[229]:


sns.set_style('whitegrid')
sns.histplot(x="age", data=health_insurance_df, bins=47)
plt.title("Distribution of Age Feature")
plt.show()


# In[230]:


sns.lmplot(x="age", y="charges", data=health_insurance_df, ci=None, line_kws={"color":"red"})
plt.title("Correlation of Age with Charges")
plt.show()


# In[231]:


# make correlation slightly more linear
health_insurance_df['age^2'] = health_insurance_df['age'] ** (2)
sns.lmplot(x="age^2", y="charges", data=health_insurance_df, ci=None, line_kws={"color":"red"})
plt.title("Correlation of Age with Charges")
plt.show()


# In[232]:


# BMI (body mass index) feature 
health_insurance_df['bmi'].describe()


# In[233]:


health_insurance_df['bmi'].value_counts().sort_index()


# In[234]:


# BMI (body mass index) feature 
sns.histplot(x="bmi", data=health_insurance_df)
plt.title("Distribution of BMI (body mass index) Feature")
plt.show()


# In[235]:


sns.lmplot(x="bmi", y="charges", data=health_insurance_df, ci=None, line_kws={"color":"red"})
plt.title("Correlation of BMI (body mass index) with Charges")
plt.show()


# In[236]:


# Children feature 
health_insurance_df['children'].describe()


# In[237]:


health_insurance_df['children'].value_counts().sort_index()


# In[238]:


sns.histplot(x="children", data=health_insurance_df)
plt.title("Distribution of Children Feature")
plt.show()


# In[239]:


# Smoker Feature
health_insurance_df['smoker'].describe()


# In[240]:


smoker_values = health_insurance_df['smoker'].value_counts()
smoker_values


# In[241]:


plt.pie(smoker_values, labels=smoker_values.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=['orange', 'lightblue'], textprops = {'color':'black', 'fontsize':15, 'fontweight':'bold'}, pctdistance=0.6, 
        labeldistance=1.2)
plt.title("Smoker Feature Pie Chart", fontsize=18)
plt.show()


# In[242]:


# Sex Feature
health_insurance_df['sex'].describe()


# In[243]:


sex_values = health_insurance_df['sex'].value_counts()
sex_values


# In[244]:


plt.pie(sex_values, labels=sex_values.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=['orange', 'lightblue'], textprops = {'color':'black', 'fontsize':15, 'fontweight':'bold'}, pctdistance=0.6, 
        labeldistance=1.2)
plt.title("Sex Feature Pie Chart", fontsize=18)
plt.show()


# In[245]:


# Region Feature
health_insurance_df['region'].describe()


# In[246]:


region_values = health_insurance_df['region'].value_counts()
region_values


# In[247]:


plt.pie(region_values, labels=region_values.index, autopct='%1.1f%%', shadow=True, startangle=90, colors=['orange', 'lightblue', 'pink', 'lightgreen'], textprops = {'color':'black', 'fontsize':15, 'fontweight':'bold'}, pctdistance=0.6, 
        labeldistance=1.2)
plt.title("Region Feature Pie Chart", fontsize=18)
plt.show()


# In[248]:


# Charges Values
health_insurance_df['charges'].describe()


# In[249]:


sns.boxplot(x="charges", data=health_insurance_df, color='orange')
plt.title("Distribution of Charge Values", fontsize=16)
plt.show()


# In[250]:


sns.stripplot(x='region', y='charges', data=health_insurance_df, hue='smoker');
plt.title("Distribution of Charge Values Based on Sex & Region", fontsize=16)
plt.show()


# In[251]:


sns.boxplot(x='region', y='charges', data=health_insurance_df, hue='sex')
plt.title("Distribution of Charge Values Based on Sex & Region", fontsize=16)
plt.show()


# In[252]:


### feature engineering
# identify the independent and dependent variables
independent_variables = health_insurance_df[['age^2', 'sex', 'bmi', 'children', 'smoker', 'region']]
dependent_variables = health_insurance_df.loc[:, 'charges']


# In[253]:


dependent_variables.head()


# In[254]:


independent_variables.head()


# In[255]:


# one-hot econding (converting categorical variables)
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(sparse=False, drop='first')
dummy_encoded_df = pd.DataFrame(onehotencoder.fit_transform(independent_variables[['sex', 'smoker', 'region']]))
onehotencoder.categories_


# In[256]:


dummy_encoded_df.head()


# In[257]:


dummy_encoded_df.columns


# In[258]:


# changing the column names of the dataframe with dummy variables 
dummy_encoded_df.columns = ["sex_male", "smoker_yes", "region_northwest", "region_southeast", "region_southwest"]


# In[259]:


dummy_encoded_df.columns


# In[260]:


dummy_encoded_df.head()


# In[261]:


health_insurance_df.head()


# In[262]:


# concating the two dataframes
health_insurance_df = pd.concat([dummy_encoded_df, health_insurance_df], axis=1)
health_insurance_df.drop(['region', 'sex', 'smoker', 'age'], axis=1, inplace=True)
health_insurance_df.head()


# In[263]:


# scaling the data 
from sklearn.preprocessing import StandardScaler

for num_col in health_insurance_df[['age^2','bmi', 'children', 'charges']]:
  SS = StandardScaler()
  health_insurance_df[num_col] = SS.fit_transform(health_insurance_df[[num_col]])
health_insurance_df.head()


# In[264]:


# resplitting data
independent_variables = health_insurance_df[['sex_male', 'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest', 'bmi', 'children', 'age^2']]
dependent_variables = health_insurance_df.loc[:, 'charges']
independent_variables.head()


# In[265]:


dependent_variables.head()


# In[266]:


# check for multicollinearity
sns.heatmap(health_insurance_df.corr(), annot=True, robust=True, cmap='mako', linewidth=0.01, linecolor='w')
plt.title("Heatmap Displaying Correlations")
plt.show()


# In[267]:


### Building the LinearRegression model 
# split the data into training and testing data 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(independent_variables,dependent_variables,test_size=0.2,random_state=12)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[268]:


from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge
kf = KFold(n_splits=5, shuffle=True, random_state=10)
param_grid = {"alpha": np.arange(0.001, 1, 10)}
ridge = Ridge()
ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
ridge_cv.fit(X_train, y_train)
print(ridge_cv.best_params_, ridge_cv.best_score_)
y_pred = ridge_cv.predict(X_test)
r2 = ridge_cv.score(X_test, y_test)
r2
# 0.7118438306721093


# In[269]:


# DataFrame with the actual and predicted charges
results_df = pd.DataFrame({'Actual': y_test,'Predicted' : y_pred})
results_df.head()


# In[270]:


# Evaluating the performance of the model
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
rsme = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', rsme) 
print('R^2:', r2)
adjusted_r2 = 1 - (1-ridge_cv.score(X_test, y_test))*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
print('Adjusted R^2:', adjusted_r2) 


# In[271]:


# plotting the results
sns.lmplot(x="Actual", y="Predicted", data=results_df, ci=None, line_kws={'color': 'red'})
plt.title("Actual vs Predicted Charges Values")
plt.show()

