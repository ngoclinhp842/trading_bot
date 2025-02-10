"""
envirironment.py

This script sets up custom environments using OpenAI's Gym-AnyTrading environment.

Author: Michelle Phan
Date: Jan 27, 2023
"""
from gym_anytrading.envs import StocksEnv
import pandas as pd
from data import get_stock_data

def load_stock_data_live(ticker, start, end):
    """Fetch and format latest stock data for the RL agent."""
    if not isinstance(ticker, str):
        raise TypeError("Ticker must be a string, received:", type(ticker))

    df = get_stock_data(ticker, start=start, end=end)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df

# Custom Gym Trading Environment
class CustomStockEnv(StocksEnv):
    def __init__(self, ticker):
        df = load_stock_data_live(ticker, start="2022-01-01", end="2025-01-27")
        window_size = 10
        super().__init__(df=df, window_size=window_size, frame_bound=(window_size, len(df)))

# Example usage
env = CustomStockEnv("AAPL")
