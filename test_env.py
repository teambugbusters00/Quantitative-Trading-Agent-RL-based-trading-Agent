import pandas as pd
from trading_env import TradingEnv
from data_utils import get_processed_data

def test_workflow():
    print("Testing data fetching and processing...")
    ticker = "BTC-USD"
    df = get_processed_data(ticker)
    
    if df.empty:
        print("Error: DataFrame is empty")
        return
    
    print(f"Fetched {len(df)} rows for {ticker}")
    print(f"Columns: {df.columns.tolist()}")
    
    print("\nTesting TradingEnv...")
    env = TradingEnv(df.tail(10))
    state, _ = env.reset()
    print(f"Initial State: {state}")
    
    actions = ["buy()", "hold()", "sell()"]
    for action in actions:
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward:.2f}, Done: {done}, Info: {info}")

if __name__ == "__main__":
    test_workflow()
