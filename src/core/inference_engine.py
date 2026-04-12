import os
import sys
from openai import OpenAI
from src.api.client import TradingEnvClient

# Read environment variables with defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    # During hackathon evaluation, HF_TOKEN will be provided.
    # On HF Spaces, this should be set in Secrets.
    print("[ERROR] HF_TOKEN environment variable is required")
    # In a real environment, we might sys.exit(1), but for flexibility:
    api_key = "MOCK_TOKEN" 
else:
    api_key = HF_TOKEN

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=api_key
)

def get_llm_action(state):
    """
    Get action from LLM.
    """
    prompt = f"""
    You are a quantitative trading agent.

    Market State:
    Price: {state['price']:.2f}
    RSI: {state['rsi']:.2f}
    MACD: {state['macd']:.2f}
    Position: {state['position']}

    Decide action: buy(), sell(), hold()
    Return ONLY action name.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        action = response.choices[0].message.content.strip().lower()
        for a in ["buy()", "sell()", "hold()"]:
            if a in action: return a
        return "hold()"
    except Exception as e:
        return f"error: {str(e)}"

def run():
    """
    Main entry point for the agent execution.
    """
    # For HF Spaces deployment, the server runs in the same container.
    # While it's building/starting, we might need to wait or just assume it's up
    # because OpenEnv typically runs the server then the inference script.
    env = TradingEnvClient(base_url="http://localhost:7860")
    
    print(f"[START] task=trading env=openenv model={MODEL_NAME}")

    rewards = []
    step = 0
    success = False

    try:
        state = env.reset()

        while True:
            action = get_llm_action(state)
            
            if action.startswith("error:"):
                raise Exception(action)

            next_state, reward, done, info = env.step(action)

            rewards.append(round(reward, 2))
            step += 1

            error_str = info.get("error") if info.get("error") else "null"

            print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_str}")

            state = next_state

            if done:
                total_reward = sum(rewards)
                success = total_reward > 0
                break

    except Exception as e:
        print(f"[STEP] step={step+1} action=none reward=0.00 done=true error=\"{str(e)}\"")

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")
