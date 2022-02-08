#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import json
#import requests
#from config import api_key
import yfinance as yf

#from pandas_datareader import data as pdr
#yf.pdr_override() 


# In[2]:


#endpoint = "/v8/finance/spark"
#url = "https://yfapi.net" + endpoint


# In[3]:


#querystring = {"symbols":"AAPL,BTC-USD,EURUSD=X"}

#headers = {
#    'x-api-key': api_key
#    }

#response = requests.request("GET", url, headers=headers, params=querystring)

#print(response.text)


# In[4]:


#msft = yf.Ticker("MSFT")
#msft.info
#hist = msft.history(period="max")
#print(hist)


# In[5]:


#data = pdr.get_data_yahoo("MSFT", start="2017-01-01", end="2022-01-31")
#data


# In[6]:


# Read the raw data
path='Resources/key_stats_yahoo.csv'
df = pd.read_csv(path, index_col=False)


# In[7]:


df


# In[8]:


df = df.drop(columns=['Unnamed: 0'])


# In[9]:


#Ticker Array
tickers = df['Ticker'].unique()
tickers


# In[10]:


#List of number columns
num_col = df.dtypes[df.dtypes == 'float64'].index.tolist()
num_col


# In[11]:


# data types
df.dtypes


# In[12]:


#Replace NaN with Mean
#ticker='a'
for ticker in tickers:
    m = df[df['Ticker'] == ticker][num_col].mean()
    mask = df[df['Ticker'] == ticker]
    mask[num_col] = mask[num_col].fillna(value = m)
    df[df['Ticker'] == ticker] = mask
df


# In[13]:


#Check remaining Nans
df.isnull().sum()


# In[14]:


#df[df['DE Ratio'].isnull()]


# In[15]:


#drop remaining Nans
df.dropna(axis = 'rows', how='any',inplace=True)
df


# In[16]:


df.isnull().sum()


# In[17]:


df.describe()


# In[18]:


#Remove outliers
#from scipy import stats
for i in num_col:
    df = df[np.abs(df[i]-df[i].mean()) <= (3*df[i].std())]

df = df[df['Price']<10000]
#cdf[(np.abs(stats.zscore(cdf)) < 3).all(axis=0)]
#cdf[(np.abs(stats.zscore(cdf[0])) < 3)]
df.describe()


# In[19]:


#export clean data
df.to_csv('Resources/clean_data.csv', index=False)


# In[20]:


# Import dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler
import pandas as pd
import tensorflow as tf
#turn off the warnings
import warnings
warnings.filterwarnings('ignore')

# Import checkpoint dependencies
import os
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error
import sklearn
import matplotlib.pyplot as plt


# In[21]:


#import clean data
cdf=pd.read_csv('Resources/clean_data.csv')
cdf


# In[22]:


cdf = cdf.drop(columns = ['Date','Ticker','DE Ratio',
 #'Trailing P/E',
 'Price/Sales',
 'Price/Book',
 'Profit Margin',
 'Operating Margin',
 #'Return on Assets',
 #'Return on Equity',
 'Revenue Per Share',
 'Market Cap',
 'Enterprise Value',
 #'Forward P/E',
 #'PEG Ratio',
 'Enterprise Value/Revenue',
 #'Enterprise Value/EBITDA',
 'Revenue',
 'Gross Profit',
 'EBITDA',
 'Net Income Avl to Common ',
 #'Diluted EPS',
 #'Earnings Growth',
 'Revenue Growth',
 'Total Cash',
 'Total Cash Per Share',
 'Total Debt',
 'Current Ratio',
 'Book Value Per Share',
 'Cash Flow',
 'Beta'
                         ])


# In[23]:


cdf.dtypes


# In[24]:


# Split our preprocessed data into our features and target arrays
y = cdf["Price"].values
X = cdf.drop(["Price"],1).values

# Split the preprocessed data into a training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[25]:


# Create a StandardScaler instances
scaler = StandardScaler()

# Fit the StandardScaler
X_scaler = scaler.fit(X_train)

# Scale the data
X_train_scaled = X_scaler.transform(X_train)
X_test_scaled = X_scaler.transform(X_test)
X_train_scaled


# In[54]:


# Define the model - deep neural net, i.e., the number of input features and hidden nodes for each layer.
number_input_features = len(X_train_scaled[0])
hidden_nodes_layer1 = 80
hidden_nodes_layer2 = 30
hidden_nodes_layer3 = 10

nn = tf.keras.models.Sequential()

# First hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer1, input_dim=number_input_features, activation="relu"))

# Second hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer2, activation="selu"))

# Third hidden layer
nn.add(tf.keras.layers.Dense(units=hidden_nodes_layer3, activation="selu"))

# Output layer
nn.add(tf.keras.layers.Dense(units=1, activation="selu"))

# Check the structure of the model
nn.summary()


# In[55]:


train_size = len(X_train_scaled)
train_size 


# In[56]:


batch_size = 32
steps_per_epoch = train_size / batch_size
save_period = 5
save_freq = int(save_period * steps_per_epoch)


# In[57]:


# Compile the model
nn.compile(loss="mean_absolute_error", optimizer="adam")


# In[58]:


# Train the model
fit_model = nn.fit(X_train_scaled, y_train, epochs=50, batch_size = batch_size)


# In[59]:


pred = X_test[:100]#.reshape(-1, 1)
pred = pred.reshape(-1,cdf.shape[1]-1)
pred.shape


# In[60]:


ee = nn.predict(pred)
for rr in range(len(pred)):
    print(f"Prediction: {ee[rr]}, Actual: {y_test[:100][rr]}")


# In[61]:


# Evaluate the model using the test data
model_loss = nn.evaluate(X_test_scaled,y_test,verbose=2)
print(f"Loss: {model_loss}")


# In[62]:


# Export the model to HDF5 file
nn.save("Model/AlphabetSoupCharity_Optimization.h5")


# In[63]:


# WebSite
from splinter import Browser
from webdriver_manager.chrome import ChromeDriverManager


# In[66]:


# Set up Splinter
executable_path = {'executable_path': ChromeDriverManager().install()}
browser = Browser('chrome', **executable_path, headless=True)


# In[65]:


url = 'https://redplanetscience.com'
browser.visit(url)
# Optional delay for loading the page
browser.is_element_present_by_css('div.list_text', wait_time=1)


# In[37]:


if __name__ == "__main__":
    app.run()