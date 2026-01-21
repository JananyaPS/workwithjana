#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER LOAN PREDICTION PROJECT

# 1.IMPORT THE LIBRARIES:

# In[2]:


import pandas as pd
import numpy as np                     
import seaborn as sns                 
import matplotlib.pyplot as plt 
import seaborn as sn                   
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                       
warnings.filterwarnings("ignore")


# 2.IMPORT THE DATASET:

# In[3]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


# 3.CHECHKING THE FEATURES OF THE COLUMN:

# In[4]:


train.columns


# In[5]:


test.columns


# It can be identified that "Subscribed"is the target variable as it is not present in the test dataset.

# 4.PRINTING THE FIRST FIVE ROWS OF TRAIN DATASET:

# In[6]:


train.head()


# 5.CONVERTING TO NUMERICAL VARIABLES:

# In[7]:


train['subscribed'].replace('no', 0,inplace=True)
train['subscribed'].replace('yes', 1,inplace=True)


# 6.PLOTTING CORRELATION PLOT:

# In[8]:


corr = train.corr()
mask = np.array(corr)
mask[np.tril_indices_from(mask)] = False
fig,ax= plt.subplots()
fig.set_size_inches(20,10)
sn.heatmap(corr, mask=mask,vmax=.9, square=True,annot=True, cmap="YlGnBu")


# It is clear from the correlation plot that the duration of the call is highly correlated with the target variable.
# 

# 7.CHECKING FOR NULL VALUES:

# In[9]:


train.isnull().sum()


# 8.MODEL BUILDING:

# In[10]:


target = train['subscribed']
train = train.drop('subscribed',1)


# 9.APPLYING DUMMIES ON THE TRAIN DATASET:

# In[11]:


train = pd.get_dummies(train)


# 10.SPLITTING THE DATASET:

# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train, X_val, y_train, y_val = train_test_split(train, target, test_size = 0.2, random_state=12)


# 11.APPLYING LOGISTIC REGRESSION:

# In[14]:


from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()


# 12.FITTING THE DATASETS INTO MODEL:

# In[15]:


lreg.fit(X_train,y_train)


# 13.PREDICTION OF DATASET:

# In[16]:


prediction = lreg.predict(X_val)


# 14.CALCULATING ACCURACY SCORE:

# In[17]:


from sklearn.metrics import accuracy_score
accuracy_score(y_val, prediction)


# 15.GENERATION OF NEW DATA:

# In[22]:


new_data = np.random.rand(10, 52)


# 16.PREDICTION ON NEW DATA:

# In[25]:


predictions = lreg.predict(new_data)
print(predictions)


# Thus, the customer is not likely to get a loan is the prediction done by the model when provided new data.

# In[ ]:




