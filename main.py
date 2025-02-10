"""
main.py


Analysis using quantstats(https://github.com/ranaroussi/quantstats) after training and testing the model

Author: Michelle Phan
Date: Jan 27, 2023

Usage:
    python3 main.py
"""

import gymnasium as gym

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs

from stable_baselines3 import PPO
from environment import CustomStockEnv, load_stock_data_live
from stable_baselines3.common.env_checker import check_env

def main():

    # Load trained model
    model = PPO.load("models/ppo_stock_trader")

    # Initialize environment
    env = CustomStockEnv("AAPL")
    check_env(env)

    # Ensure `env.reset()` returns correctly shaped observation
    vec_env = model.get_env()
    obs = env.reset()

    if isinstance(obs, tuple):  
        obs = obs[0]  # Extract observation if a tuple is returned
    print(obs)

    done = False

    while not done:
        action = env.action_space.sample()
        action, _states = model.predict(obs)  

        step_return = env.step(action)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # env.render()
        if done:
            break

    env.close()
    print("info:", info)

    # Example Usage
    ticker = "AAPL"
    df = load_stock_data_live(ticker, start="2022-01-01", end="2025-01-27")

    # extend pandas functionality with metrics, etc.
    qs.extend_pandas()
    net_worth = pd.Series(env.unwrapped.history['total_profit'], index=df.index[10+1:len(df)])
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.metrics(returns, title='PPO Returns', show=True)

if __name__ == '__main__':
    main()