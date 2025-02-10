"""
train.py

Training Script using PPO agent for stock trading Gymtrading StockEnv

Author: Michelle Phan
Date: Jan 27, 2023

Usage:
    python3 train.py
"""
from stable_baselines3 import PPO, A2C
from environment import CustomStockEnv

# Load stock data
env = CustomStockEnv("AAPL") 
print("Environment initialized successfully!")

# Train PPO Agent
# RL Algorithms: https://stable-baselines3.readthedocs.io/en/master/guide/algos.html
# MlpPolicy: https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
# Trading Attemp 1
# model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.99)

# Trading Attempt 2
model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.0003, gamma=0.95)
model.learn(total_timesteps=100000)  # Adjust based on dataset size

# Save the trained model
model.save("models/a2c_stock_trader")
print("Model trained and saved!")
