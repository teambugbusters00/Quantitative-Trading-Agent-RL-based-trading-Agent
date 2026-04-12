import yfinance as yf
import pandas as pd
import numpy as np

def fetch_data(ticker="BTC-USD", period="1mo", interval="1h"):
    """
    Fetch historical market data using yfinance.
    """
    try:
        data = yf.download(ticker, period=period, interval=interval)
        if data.empty:
            raise ValueError(f"No data found for ticker {ticker}")
        
        # Ensure multi-index columns are handled if yfinance returns them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def add_indicators(df):
    """
    Add RSI and MACD indicators to the dataframe manually.
    """
    if df.empty:
        return df
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Drop NaN values created by indicators
    df.dropna(inplace=True)
    return df

def get_processed_data(ticker="BTC-USD"):
    """
    Fetch and process data in one go.
    """
    df = fetch_data(ticker)
    df = add_indicators(df)
    return df
