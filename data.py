"""
data.py

This script uses yfinance to Get Live Stock Data

Author: Michelle Phan
Date: Jan 27, 2023

Usage:
    python3 data.py
"""
import yfinance as yf
import pandas as pd

def get_stock_data(ticker, start="2022-01-01", end="2025-01-27"):
    """
    Fetches historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple).
        start (str): Start date for historical data.
        end (str): End date for historical data.

    Returns:
        pd.DataFrame: Stock data with Date, Open, High, Low, Close, Volume.
    """
    # Debugging print statement
    print(f"Fetching stock data for ticker: {ticker}")  
    assert isinstance(ticker, str), "Ticker should be a string!"

    stock = yf.Ticker(ticker)
    df = stock.history(start=start, end=end)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.reset_index(inplace=True)
    return df

# Example usage
df = get_stock_data("AAPL", start="2022-01-01", end="2025-01-27")
print(df.head())

# Save to CSV for RL training
df.to_csv("datasets/aapl_stock_data.csv", index=False)
