#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd 


# In[3]:


import matplotlib.pyplot as plt 


# In[4]:


from sklearn import metrics 


# In[5]:


import seaborn as sb


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 


# In[7]:


from sklearn.preprocessing import PolynomialFeatures


# In[8]:


df = pd.read_csv('kc_house_data.csv', encoding= "ISO-8859-1")


# In[9]:


df


# In[10]:


df.drop('date',1,inplace=True)


# In[11]:


df.isnull().sum()


# # Linear Regression
# 

# In[12]:


y=df[['price']]


# In[13]:


y.shape


# In[14]:


x=df.drop('price',axis=1)


# In[15]:


x.shape


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=30)


# In[17]:


model = LinearRegression()


# In[18]:


model.fit(x_train, y_train)


# In[19]:


predictions=model.predict(x_test)


# In[20]:


print("MSE :", metrics.mean_squared_error(y_test,predictions))


# In[21]:


print("R squared :",metrics.r2_score(y_test,predictions))


# In[22]:


x=df[['sqft_living']]
y=df[['price']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=30)
model = LinearRegression()
model.fit(x_train, y_train)
pred=model.predict(x_test)
print("MSE :", metrics.mean_squared_error(y_test,predictions))
print("R squared :",metrics.r2_score(y_test,predictions))


# In[23]:


plt.scatter(x,y,color='r')
plt.title("linear regression")
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.plot(x,model.predict(x),color='k')
plt.show()


# In[24]:


x=df[['sqft_living']]
y=df[['grade']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=30)
model = LinearRegression()
model.fit(x_train, y_train)
predictions=model.predict(x_test)
print("MSE :", metrics.mean_squared_error(y_test,predictions))
print("R squared :",metrics.r2_score(y_test,predictions))


# In[25]:


plt.scatter(x,y,color='r')
plt.title("linear regression")
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.plot(x,model.predict(x),color='k')
plt.show()


# In[26]:


x=df[['sqft_living']]
y=df[['bathrooms']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=30)
model = LinearRegression()
model.fit(x_train, y_train)
pred=model.predict(x_test)
print("MSE :", metrics.mean_squared_error(y_test,predictions))
print("R squared :",metrics.r2_score(y_test,predictions))


# In[27]:


plt.scatter(x,y,color='r')
plt.title("linear regression")
plt.xlabel('True values')
plt.ylabel('Predictions')
plt.plot(x,model.predict(x),color='k')
plt.show()


# # Multiple Linear Regression

# In[28]:


X= df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above','sqft_basement','yr_built', 'yr_renovated','zipcode','yr_built']].values
Y = df ['price'].values


# In[29]:


plt.figure(figsize=(10,5))
plt.tight_layout()
sb.distplot(df['price'])


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[31]:


LR = LinearRegression()  
LR.fit(X_train, y_train)


# In[32]:


y_pred = LR.predict(X_test)


# # Polynomial Regression

# In[33]:


X= df[['bedrooms', 'bathrooms', 'sqft_living']].values
Y = df ['price'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
LR = LinearRegression()  
poly = PolynomialFeatures(degree = 3) 
X_poly_fit = poly.fit_transform(x_train) 
LR.fit(X_poly_fit,y_train) 


# In[34]:


x_test=poly.fit_transform(x_test)
predect=LR.predict(x_test)


# In[35]:


plt.scatter(x,y,color='r')
plt.title("poly regression")
plt.plot(x,LR.predict(poly.fit_transform(x)),color='k')
plt.show()

