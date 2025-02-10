"""
test.py

Testing the model

Author: Michelle Phan
Date: Jan 27, 2023

Usage:
    python3 test.py
"""
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from environment import CustomStockEnv
from stable_baselines3.common.env_checker import check_env

# Load trained model
model = PPO.load("models/a2c_stock_trader")

# Initialize environment
env = CustomStockEnv("AAPL")
check_env(env)

# Ensure `env.reset()` returns correctly shaped observation
vec_env = model.get_env()
obs = env.reset()

if isinstance(obs, tuple):  
    obs = obs[0]  # Extract observation if a tuple is returned

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

# Plot Results
plt.figure(figsize=(16, 6))
env.unwrapped.render_all()
plt.show()