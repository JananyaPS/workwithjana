#!/usr/bin/env python
# coding: utf-8

# # EMAIL SPAM CLASSIFICATION

# 1.IMPORT THE LIBRARIES:

# In[39]:


import numpy as np 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# 2.IMPORT THE DATASET:

# In[40]:


data=pd.read_csv("enronSpamSubset.csv")


# 3.PRINT THE FIRST FIVE ROWS OF THE DATASET:

# In[41]:


data.head()


# 4.IDENTIFY THE TARGET VARIABLE:

# In[42]:


X= data["Body"]
y= data["Label"]


# 5.SPILTING OF THE DATASET:

# In[43]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2,random_state=42)


# 6.CONVERTING TEXT INTO NUMERIC FORM:

# In[44]:


vector=CountVectorizer()


# CountVectorizer() is a class in scikit-learn that transforms a collection of text documents into a numerical matrix of word or token counts

# 7.SCALING THE VALUES:

# In[45]:


X_train=vector.fit_transform(X_train)
X_test=vector.transform(X_test)


# 8.APPLYING THE MODEL:

# In[46]:


model=MultinomialNB()


# 9.FITTING THE DATA INTO MODEL:

# In[47]:


model.fit(X_train,y_train)


# 10.PREDICTION ON TEST SET:

# In[48]:


pred=model.predict(X_test)


# In[49]:


pred


# 11.FINDING THE ACCURACY:

# In[50]:


accuracy=model.score(X_test,y_test)


# In[51]:


print(accuracy)


# 12.PREDICTION ON NEW DATASET:

# In[52]:


new_emails=[
    "Congrats dude! You're eligible to claim bonus",
    "Hey,we have an important meeting tomorrow"
    ]


# In[53]:


new_emails_transformed=vector.transform(new_emails)


# In[54]:


new_predictions= model.predict(new_emails_transformed)


# In[55]:


new_predictions


# Thus, the multinomial model Naive Bayes classifier classified accurately the mails to into spam(1) and not spam(0) with a high accuracy of 98%.

# In[ ]:




