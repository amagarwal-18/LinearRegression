
# coding: utf-8

# In[97]:


import os
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


# In[98]:


os.getcwd()

os.chdir('/Users/amitagarwal/Desktop/Amit/game-of-thrones/30Sep')
df = pd.read_csv('train.csv')
df.head()


# In[99]:


df.describe()


# In[100]:


correlation_values = df.select_dtypes(include=[np.number]).corr()
correlation_values


# In[101]:


correlation_values[["SalePrice"]]


# In[102]:


def threashold(value,df_temp):
     df_temp = df_temp[["SalePrice"]][(df_temp["SalePrice"]>=value)|(df_temp["SalePrice"]<=-value)]
     return df_temp.index
    
df_feature = threashold(0.6,correlation_values)

correlation_values_feature = df[df_feature].select_dtypes(include=[np.number]).corr()
correlation_values_feature


# In[118]:


X = df[["OverallQual","TotalBsmtSF","GrLivArea","GarageArea","GarageCars","1stFlrSF"]]
y=df["SalePrice"]


# In[119]:


X_train, X_test, y_train, y_test = tts(X,y, test_size = 0.3, random_state = 42)


# In[120]:


reg = LinearRegression()


# In[121]:


reg.fit(X_train,y_train)


# In[122]:


y_pred = reg.predict(X_test)


# In[123]:


reg.score(X_test,y_test)


# In[124]:


rmse = np.sqrt(mean_squared_error(y_test,y_pred))
rmse

