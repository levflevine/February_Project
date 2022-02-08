import pandas as pd
import numpy as np
import json
import requests
from config import api_key
import yfinance as yf

def data_etl():
    # Read the raw data
    path='Resources/key_stats_yahoo.csv'
    df = pd.read_csv(path, index_col=False)

    df = df.drop(columns=['Unnamed: 0'])

    #Ticker Array
    tickers = df['Ticker'].unique()

    #List of number columns
    num_col = df.dtypes[df.dtypes == 'float64'].index.tolist()

    #Replace NaN with Mean
    for ticker in tickers:
        m = df[df['Ticker'] == ticker][num_col].mean()
        mask = df[df['Ticker'] == ticker]
        mask[num_col] = mask[num_col].fillna(value = m)
        df[df['Ticker'] == ticker] = mask

    #drop remaining Nans
    df.dropna(axis = 'rows', how='any',inplace=True)

    #Remove outliers
    #from scipy import stats
    for i in num_col:
        df = df[np.abs(df[i]-df[i].mean()) <= (3*df[i].std())]

    df = df[df['Price']<10000]

    #export clean data
    df.to_csv('Resources/clean_data.csv', index=False)
    #return None

def data_analysis():
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

    #import clean data
    cdf0 = pd.read_csv('Resources/clean_data.csv')

    cdf = cdf0.drop(columns = ['Date','Ticker','DE Ratio',
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
    'Diluted EPS',
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
    active_col = cdf.columns.to_list()

    # Split our preprocessed data into our features and target arrays
    y = cdf["Price"].values
    X = cdf.drop(["Price"],1).values

    # Split the preprocessed data into a training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    # Create a StandardScaler instances
    scaler = StandardScaler()

    # Fit the StandardScaler
    X_scaler = scaler.fit(X_train)

    # Scale the data
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)
    X_train_scaled

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

    train_size = len(X_train_scaled)

    batch_size = 32
    steps_per_epoch = train_size / batch_size
    save_period = 5
    save_freq = int(save_period * steps_per_epoch)

    # Compile the model
    nn.compile(loss="mean_absolute_error", optimizer="adam")

    # Train the model
    fit_model = nn.fit(X_train_scaled, y_train, epochs=50, batch_size = batch_size)

    #pred = X_test[:100]
    #pred = pred.reshape(-1,cdf.shape[1]-1)
    #pred.shape

    # Export the model to HDF5 file
    nn.save("Model/AlphabetSoupCharity_Optimization.h5")
    #return None

def data_predict(ticker):

    #Connect to PostgreSQL
    import pandas as pd
    import psycopg2
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler,OneHotEncoder, MinMaxScaler
    from sqlalchemy import create_engine
    from config import postgres_pass
    conn = psycopg2.connect(host="localhost", port = 5432, database="final_project", user="postgres", password=postgres_pass)

    db_string = f"postgresql://postgres:{postgres_pass}@127.0.0.1:5432/final_project"
    engine = create_engine(db_string) 

    # Move the Cleaned data to Postgres
    #cdf.to_sql(name='clean_data', con=engine, if_exists='replace') #### Check where to move ######

    # Select Ticker
    #ticker = 'adbe'

    #Call Yahoo Finance API
    #endpoint = "/v8/finance/quoteSummary/"
    #url = "https://yfapi.net" + endpoint + ticker

    # pull Ticker details from Yahoo Finance
    ticker_dict = yf.Ticker(ticker)
    t_info = ticker_dict.info

    # Get Ticker Features
    tick_df = pd.DataFrame(list(t_info.items()))
    tick_df = tick_df.transpose()
    new_header = tick_df.iloc[0] 
    for i in range(len(new_header)):
        new_header[i] = new_header[i].lower()
    new_header
    tick_df = tick_df[1:] 
    tick_df.columns = new_header

    #Selected Ticker to Postgres
    tick_df.to_sql(name='ticker', con=engine, if_exists='replace')

    # Create a cursor object
    cur = conn.cursor()

    #Query the Feature data for DL Model
    cur.execute("""SELECT 
    trailingPE, returnOnAssets, returnOnEquity, forwardPE, pegRatio, enterpriseToEbitda, earningsGrowth
    FROM ticker;""")      

    query_results = list(np.asarray(cur.fetchall()[0]))

    #Remove None values received from Yahoo Finance 
    for i in range(len(query_results)):
        if query_results[i] == None:
            query_results[i] =0
    

    # Load the Saved Model
    from tensorflow import keras
    nn = keras.models.load_model("Model/AlphabetSoupCharity_Optimization.h5")

    # Run the model to predict Stock Price
    query_results = np.array(query_results).reshape(-1,1)
    query_results = query_results.transpose()
    #query_results.shape

    # Create a StandardScaler instances
    #scaler = StandardScaler()

    # Fit the StandardScaler
    #X_scaler = scaler.fit(query_results)

    # Scale the data
    #Q_R_scaled = X_scaler.transform(query_results)
    
    predicted_price = pd.DataFrame(data=nn.predict(query_results), columns=['predictedprice'], index=None)

    #Save Predicted Price to Postgres
    #Selected Ticker to Postgres
    predicted_price.to_sql(name='predicted_price', con=engine, if_exists='replace', index = None)

    # Close the cursor and connection to so the server can allocate
    # bandwidth to other requests
    cur.close()
    conn.close()
    return None

def recommendation(cp, pp):
    if pp/cp>1.5:
        rec = "Strong Buy"
    elif pp/cp>1:
        rec = "Buy"
    else:
        rec = "Sell"
    return rec

def run_all(ticker):
    #data_etl()
    #data_analysis()
    data_predict(ticker)
    return None

if __name__ == "__main__":
    run()

