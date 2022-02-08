# Final_Project

## Collaborators 

**Alexander, Lydia** - Database

**Sabir, Muhammad** - Github & Vizualization

**Levine, Lev** - Machine Learning

## Overview

### Selected topic

  * Stock Market Prediction

### Reason why they selected their topic

  * Wanted to work on something useful and found this interesting

### Description of data sorces

  - Yahoo Finance dataset on Kaggle  [https://www.kaggle.com/jerryhans/key-statistics-yahoo-finance-stocks-of-2003-2013](https://www.kaggle.com/jerryhans/key-statistics-yahoo-finance-stocks-of-2003-2013)
  - Yahoo Finance API [https://www.yahoofinanceapi.com/](https://www.yahoofinanceapi.com/)
  - yfinance Library [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)

### Questions they hope to answer with the data

  * Predict prices for next 15-30 days

## Machine Learning Model

### Model Overview 

A provisional Deep Learning Regression Model has been developed (see **Yahoo_Finance.ipynb**)

Target Variable: **Stock Price** (Price)

Initial Feature Variables: 

- **Price per Earnings** (Trailing P/E) 
- **Earnings per Share** (Diluted EPS)

The model is using the **3 hidden layers**:

- hidden_nodes_layer1 = **80 neurons**
- hidden_nodes_layer2 = **30 neurons**
- hidden_nodes_layer3 = **10 neurons**

The **SELU** activation function is used on all the **Hidden layers**. 

The **Linear** activation function is used on all the **Output layer**. 

The initial model is run **50 epochs** to train and is generating **Loss: 20299.24**

### Data preprosessing

- Dropped columns that did not contribute to the prediction
- Replaced NaN (null) values with mean values for each stock ticker
- Removed remaining rows that contained NaN values 
- Removed rows with outliers beyong 2 standard deviation from the mean
- Scaled the training data to Mean = 0 and STD = 1 (with Standard Scaler)

### Deep Learning Model Optimization Next Steps

- Attempt to cluster the stocks into several categories with unsupervised ML 
- Develop and fit DL models for each cluster individually 
- Experienemt with activation functions, number of hidden layers, model feature selection, number of epochs, etc

### Overall User Experience (Concept)

#### A. Explore individual stock ticker 

1. Enter stock ticker
2. The backend will pull the most recent feature data and target variable data from Yahoo Finance
3. The DL Model will calculate the predicted stock price base on the most current feature data
4. The predicted and current stock price will be displayed for the user
5. If predicted stock price > current stock price, a BUY recommendation generated and vise versa

#### B. Discover investment opportunity stocks

1. Click the "Analyze the Market" button
2. Pull the current feature data and target variable data for all the tickers in the array (>300)
3. Run the DL model to predict prices for each stock
4. Display predicted and current prices sorted in the descending order of the investment opportunity
5. Provide a recommendation on which stocks to invest
