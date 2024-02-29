#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF CAR SALES PROJECT 

# 1.IMPORTING THE LIBRARIES:

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import plot_tree


# 2.LOADING THE DATASET:

# In[2]:


data = pd.read_csv("car details v4.csv")
data


# 3.FINDING THE CORRELATION BETWEEN VARIABLES:
# This helps us to find the variables which are highly correlated and which are not correlated.
# Correlation plot gives the mapping of relationships between the variables.

# In[3]:


plt.figure(figsize=(20, 15))
correlations = data.corr()
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()


# From the above correlation plot,it is identified that the price of the car is dependent with length, width and fuel tank capacity.

# In[4]:


data = data[["Model", "Kilometer", "Fuel Type", 
             "Engine", "Height", "Width", 
             "Length", "Max Power", "Seating Capacity", 
             "Fuel Tank Capacity", "Price"]]


# 4.RESETTING THE INDEX:

# In[5]:


X=data.loc[:,["Model", "Kilometer", "Fuel Type", 
             "Engine", "Height", "Width", 
             "Length", "Max Power", "Seating Capacity", 
             "Fuel Tank Capacity", "Price"]]


# 5.SUBSETTING ALL CATEGORICAL VALUES:

# In[6]:


data_categorical = X.select_dtypes(include=['object'])


# 6.CONVERTING TO DUMMIES:

# In[7]:


data_dummies = pd.get_dummies(data_categorical, drop_first=True)


# 7.DROP CATEGORICAL VALUES AND CONCAT DUMMY VARIABLES:

# In[8]:


X = X.drop(list(data_categorical.columns), axis=1)


# 8.SCALING THE FEATURES OF THE TRAINING DATASET:

# In[9]:


from sklearn.preprocessing import scale
# storing column names in cols, since column names are (annoyingly) lost after 
# scaling (the df is converted to a numpy array)
cols = X.columns
X = pd.DataFrame(scale(X))
X.columns = cols
X.columns


# 9. TARGET VARIABLE - PRICE:

# In[11]:


Y = data["Price"]


# 10.SPLITTING THE DATASET:

# In[12]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# 11.FILL THE NULL VALUES:

# In[13]:


X_train.fillna(0,inplace=True)
X_test.fillna(0,inplace=True)


# 12.DECISION TREE REGREESSOR:

# In[14]:


model = DecisionTreeRegressor()
model.fit(X_train, Y_train)
predictions = model.predict(X_test)


# 13.PREDICTING THE TEST DATA:

# In[15]:


predictions = model.predict(X_test)


# 14.MODEL SCORE:

# In[16]:


from sklearn.metrics import mean_absolute_error
model.score(X_test, predictions)


# 15.PREDICTION ON NEW DATA:

# In[34]:


new_data = pd.DataFrame({
     "Kilometer":[2440000] ,
     "Height":[1706], "Width":[1821], 
     "Length":[4598] , 
     "Seating Capacity":[5], "Fuel Tank Capacity":[51], "Price":[3800000]
})


# Predict using the trained model
prediction = model.predict(new_data)

# 5. Evaluate the prediction
print("Predicted output:", prediction)


# 16.PLOTTING THE DECISION TREE:

# In[37]:


plt.figure(figsize=(20,10))
plot_tree(model, filled=True)
plt.show()


# In[ ]:




