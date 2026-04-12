import os
import sys
from openai import OpenAI
from trading_env import TradingEnv
from data_utils import get_processed_data

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

# For local testing, you might want to uncomment this or set it in your environment
# if os.path.exists(".env"):
#     from dotenv import load_dotenv
#     load_dotenv()
#     HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    # During hackathon evaluation, HF_TOKEN will be provided.
    # We raise an error if it's missing to ensure compliance.
    print("[ERROR] HF_TOKEN environment variable is required")
    sys.exit(1)

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

def get_action(state):
    """
    Sends the market state to the LLM and gets the next action.
    """
    prompt = f"""
    You are a quantitative trading agent.

    Market State:
    Price: {state['price']:.2f}
    RSI: {state['rsi']:.2f}
    MACD: {state['macd']:.2f}
    Current Position: {state['position']}

    Rules:
    - Avoid overbought zones (RSI > 70)
    - Follow momentum when MACD positive
    - If Position is NONE, you can buy().
    - If Position is HOLDING, you can sell() or hold().

    Decide action: buy(), sell(), hold()
    Return ONLY the action name followed by parentheses.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0  # Zero temperature for consistent reasoning
        )
        action = response.choices[0].message.content.strip().lower()
        
        # Simple parsing to ensure it's one of the three actions
        for a in ["buy()", "sell()", "hold()"]:
            if a in action:
                return a
        return "hold()"
    except Exception as e:
        return f"error: {str(e)}"

def run_episode(ticker="BTC-USD", max_steps=20):
    """
    Runs a single trading episode.
    """
    # Fetch and process data
    df = get_processed_data(ticker)
    if df.empty:
        print(f"[ERROR] Could not fetch data for {ticker}")
        return

    # Initialize environment
    # We take the last 'max_steps' from the data for simulation
    env = TradingEnv(df.tail(max_steps + 1))
    
    rewards = []
    step = 0
    success = False

    # [START] line format
    print(f"[START] task=trading env=custom-crypto model={MODEL_NAME}")

    state, _ = env.reset()

    try:
        while True:
            action = get_action(state)
            
            if action.startswith("error:"):
                raise Exception(action)

            next_state, reward, done, info = env.step(action)

            rewards.append(round(reward, 2))
            step += 1

            error = info.get("error", None)
            error_str = error if error else "null"

            # [STEP] line format
            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}")

            state = next_state

            if done:
                # Success is defined as ending with a positive net profit
                total_profit = sum(rewards)
                success = total_profit > 0
                break

    except Exception as e:
        # Emit step with error if something fails
        print(f"[STEP] step={step+1} action=none reward=0.00 done=true error=\"{str(e)}\"")

    finally:
        # [END] line format
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
        env.close()

if __name__ == "__main__":
    # In a real OpenEnv challenge, this script would be called by the evaluator.
    # We default to BTC-USD for 20 steps.
    run_episode()
