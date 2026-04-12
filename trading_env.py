import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    A custom trading environment for OpenEnv compatibility.
    """
    def __init__(self, df, initial_balance=10000):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index()
        self.initial_balance = initial_balance
        self.max_steps = len(self.df) - 1
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: Price, RSI, MACD, Position (0: None, 1: Holding)
        # We use a Dict for easy LLM parsing or Box for RL training
        self.observation_space = spaces.Dict({
            "price": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "rsi": spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            "macd": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
            "position": spaces.Discrete(2)
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.current_step = 0
        self.position = 0  # 0: None, 1: Holding
        self.entry_price = 0
        self.total_reward = 0
        
        return self._get_observation(), {}

    def _get_observation(self):
        row = self.df.iloc[self.current_step]
        return {
            "price": float(row['Close']),
            "rsi": float(row['rsi']),
            "macd": float(row['macd']),
            "position": "HOLDING" if self.position == 1 else "NONE"
        }

    def step(self, action_str):
        # Convert string action to numeric if needed
        action_mapping = {"buy()": 1, "sell()": 2, "hold()": 0}
        action = action_mapping.get(action_str.lower().strip(), 0)
        
        current_price = float(self.df.iloc[self.current_step]['Close'])
        reward = 0
        done = False
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
                
        elif action == 0:  # Hold
            reward = 0
            
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            # Force sell if holding at the end
            if self.position == 1:
                profit = current_price - self.entry_price
                reward += profit
                self.balance += profit
                self.position = 0
        
        self.total_reward += reward
        
        return self._get_observation(), reward, done, info

    def render(self):
        print(f"Step: {self.current_step}, Balance: {self.balance:.2f}, Position: {self.position}")
