#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#this is to import text file with delimiters (note that the original raw data was separated by commas) 
#and the file is assigned the name data -  you can use any name you want
#Note that if you have strings as the header , it will come up as NaN. so you have to rename the columns
path = "C:/Users/Olanrewaju.Balogun/Desktop/python/practice.txt"
data = np.genfromtxt ('C:/Users/Olanrewaju.Balogun/Desktop/python/practice.txt', delimiter =',')
#create a data frame 
df = pd.DataFrame(data)
df.head(5)


# In[3]:


#drop the first row if it has NaN 
df.dropna (subset=[1], axis = 0 , inplace = True) #this is to drop a row/column with NaN value
df.head(5)


# In[4]:


#rename the headers 
headers = ["Exam 1","Exam 2","Admitted"]
#this assigns the header names
df.columns=headers
df.head(5)


# In[5]:


#to check if there are missing/blank values in the data frame  df. 'True" response means missing value, while "False" means no missing value. 
#This is just a check do not rename df
missing_data = df.isnull()
missing_data.head(5)

for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("")


# In[35]:


#to check if all the columns contain numbers. False values indicate columns with any-non numeric values.
df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all())


# In[6]:


#change admitted column to string 
#df[["Admitted"]] = df[["Admitted"]].astype("object")
#df.head(5)


# In[7]:



from sklearn.model_selection import train_test_split


# In[8]:


x_data = X = df[['Exam 1', 'Exam 2']]
y_data = y = df['Admitted']


# In[9]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)


# In[10]:


print("number of test samples :", X_test.shape[0])
print("number of training samples:",X_train.shape[0])


# In[11]:


from sklearn.linear_model import LogisticRegression


# In[12]:


logmodel = LogisticRegression(solver='lbfgs', multi_class='auto')
logmodel.fit(X_train,y_train)


# In[28]:


#this shows accuracy of the model
accuracy = logmodel.score(X_test, y_test)
print(accuracy*100)


# In[14]:


X_test


# In[15]:


y_test


# In[16]:


testpredictions = logmodel.predict(X_test)


# In[31]:


from sklearn.metrics import classification_report


# In[32]:


#this shows the accuracy per prediction
print(classification_report(y_test,testpredictions))


# In[34]:


red = logmodel.predict([[45,71]])
red


# In[ ]:




