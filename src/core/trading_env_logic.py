import numpy as np
import pandas as pd
from src.utils.helpers import fetch_market_data, apply_indicators

class TradingEnvLogic:
    """
    Core logical representation of the trading environment.
    """
    def __init__(self, ticker="BTC-USD", initial_balance=10000, max_steps=20):
        df = fetch_market_data(ticker)
        df = apply_indicators(df)
        self.df = df.tail(max_steps + 1).reset_index()
        
        self.initial_balance = initial_balance
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0  # 0: None, 1: Holding
        self.entry_price = 0
        self.total_reward = 0
        self.is_done = False
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return {
            "price": float(row['Close']),
            "rsi": float(row['rsi']),
            "macd": float(row['macd']),
            "position": "HOLDING" if self.position == 1 else "NONE",
            "info": {}
        }

    def step(self, action_str):
        action_mapping = {"buy()": 1, "sell()": 2, "hold()": 0}
        action = action_mapping.get(action_str.lower().strip(), 0)
        
        current_price = float(self.df.iloc[self.current_step]['Close'])
        reward = 0
        info = {"error": None}
        
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            else:
                info["error"] = "Already holding a position"
        
        elif action == 2:  # Sell
            if self.position == 1:
                profit = current_price - self.entry_price
                reward = profit
                self.balance += profit
                self.position = 0
                self.entry_price = 0
            else:
                info["error"] = "No position to sell"
                
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.is_done = True
            if self.position == 1:
                profit = current_price - self.entry_price
                reward += profit
                self.balance += profit
                self.position = 0
        
        self.total_reward += reward
        return self._get_observation(), reward, self.is_done, info
