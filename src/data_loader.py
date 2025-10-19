import os
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import ta

# ---Configuration---
TICKERS = ["RELIANCE.NS", "NIFTY50.NS", "INFY.NS", "HDFCBANK.NS"]
START_DATE = "2020-01-01"
END_DATE = "2025-09-30"
SEQ_LEN = 30
DATA_DIR = "data"
RAW_DATA = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA = os.path.join(DATA_DIR, "processed")
os.makedirs(RAW_DATA, exist_ok=True)
os.makedirs(PROCESSED_DATA, exist_ok=True)

# ---Fetch Data---

def fetch_stock_data(tickers=TICKERS):
    all_data = {}
    for ticker in tqdm(tickers, desc="Fetching stock data"):
        df = yf.download(ticker, start=START_DATE, end=END_DATE)
        df.to_csv(os.path.join(RAW_DATA, f"{ticker}.csv"))
        all_data[ticker] = df
    return all_data

# ---Feature Engineering---
def add_indicators(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'].squeeze()).rsi()
    df['macd'] = ta.trend.MACD(df['Close'].squeeze()).macd()
    df['ema_20'] = ta.trend.EMAIndicator(df['Close'].squeeze(), window=20).ema_indicator()
    df['ema_50'] = ta.trend.EMAIndicator(df['Close'].squeeze(), window=50).ema_indicator()
    df.dropna(inplace=True)
    return df

# ---Sequence Creation---
def create_sequences(data, seq_len=SEQ_LEN):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 3])  # Assuming the target is the 'Close' price
    return np.array(X), np.array(y)

# ---Main Pipeline---
def prepare_data():
    print("Fetching stock data...")
    all_data = fetch_stock_data()

    combined_data = []
    for ticker, df in all_data.items():
        print(f"Processing data for {ticker}...")
        df = add_indicators(df)
        df['TICKER'] = ticker
        combined_data.append(df)

    data = pd.concat(combined_data)
    data.reset_index(inplace=True)
    data.to_csv(os.path.join(PROCESSED_DATA, "combined_data.csv"), index=False)
    print(f"Saved processed data to {os.path.join(PROCESSED_DATA, 'combined_data.csv')}")

    #Scale and sequence for one ticker 
    rel = combined_data[0].drop(columns=['TICKER'])
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(rel)

    X, y = create_sequences(scaled)
    print(f"Created sequences with shape: X={X.shape}, y={y.shape}")
    np.save(os.path.join(PROCESSED_DATA, "X.npy"), X)
    np.save(os.path.join(PROCESSED_DATA, "y.npy"), y)

    return X, y

if __name__ == "__main__":
    X, y = prepare_data()