import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import time
import logging

logging.basicConfig(level=logging.DEBUG)

def fetch_stock_data(ticker, period="1y", retries=3, delay=5):
    for attempt in range(retries):
        try:
            logging.debug(f"Fetching data for {ticker}, attempt {attempt + 1}")
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                raise ValueError(f"No data found for {ticker}, possibly delisted.")
            logging.debug(f"Successfully fetched {len(df)} data points for {ticker}")
            return df['Close'].values
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            raise e

def preprocess_data(data, sequence_length=60):
    if len(data) < sequence_length:
        raise ValueError(f"Data length {len(data)} is less than required sequence length {sequence_length}")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    try:
        data = fetch_stock_data("AAPL")
        print(f"Fetched {len(data)} data points for AAPL")
        X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
        print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    except Exception as e:
        print(f"Error in test: {e}")