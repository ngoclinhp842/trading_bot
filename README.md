# Re-enforcement learning-based trading bot
## 🎯 Project Overview:
This project focuses on developing a reinforcement learning (RL)-based stock trading bot using `Gym-AnyTrading`, `Stable-Baselines3`, and `Yahoo Finance` data. The model is trained using `Proximal Policy Optimization (PPO)` to maximize long-term profits by making dynamic `buy` and `sell` decisions.

## ✏️ Define the Reinforcement Learning Problem:
### State (Observation Space)
- Current stock price.
- Portfolio state (cash, shares held, profit/loss).
- Technical indicators (Moving Averages, RSI, MACD).

### Action Space
- 0: `Sell`
- 1: `Buy`
  
### Reward Function 
Profit/loss per trade with penalties for excessive trades.

## ⚙️ Technology:
### Data Collection
- Data Source: Yahoo Finance (`yfinance` API).
- Ticker Used: AAPL (Apple Inc.).
- Timeframe: 1 Year (January 2024 - January 2025).
- Features Used: `Open`, `High`, `Low`, `Close`, `Volume`.

### Environment Setup
`GymAnyTrading`

### Trading Actions 
Even though trading algorithms use numerous actions such as `Buy`, `Sell`, `Hold`, `Enter`, etc, deciding whether to hold, enter, or existing stock is a statistical decision depending on many parameters such as your budget, the stock you trade, your money distribution policy in multiple markets, etc. Thus, it's a massive burden for an RL agent to consider all these parameters, and it may take years to develop such an agent! 

🌟 **Only `Sell`= 0 and `Buy`= 1 actions are adequate to train an agent just as well.**

### Trading Positions:
_refers to the commitment a trader or investor has in a particular asset, such as stocks, forex, commodities, or cryptocurrencies._
-  Long Position: Buying an asset with the expectation that its price will rise
- Short Position: Selling an asset (often borrowed) with the expectation that its price will fall

🌟 **Only use `Short`= 0 and `Long`= 1 positions**

### Trading Environments
- `TradingEnv` is an abstract class that inherits `gym.Env`. This class aims to provide a general-purpose environment for all kinds of trading markets.
- `StocksEnv`: a concrete class that inherits `TradingEnv` and implements its abstract methods.

### Model Training
- Algorithm: `Proximal Policy Optimization (PPO)`
- Hyperparameters:
```
gamma = 0.99 (reward discount factor)
learning_rate = 0.0003
batch_size = 64
policy = MlpPolicy
Training Steps: 100,000 timesteps
```

## 🚀 Running TradingBot:
To run TradingBot on the file main.py 

```sh
python3 main.py
```

## 👀 Output:
- Training
<p align="center">
  <img align="center" alt="Original Image" width="400" src="">
</p>

## 📝 Analysis:

## ⚖️ License:
Apache License 2.0
