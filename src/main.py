from fastapi import FastAPI
from src.core.trading_env_logic import TradingEnvLogic
from src.core.models import TradingAction, TradingObservation, TradingState
import os

app = FastAPI(title="Trading Environment Server")

# Global environment instance (simplification for single-user space)
env = TradingEnvLogic()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/reset")
def reset():
    obs = env.reset()
    return obs

@app.post("/step")
def step(action: TradingAction):
    obs, reward, done, info = env.step(action.action)
    return {
        "observation": obs,
        "reward": float(reward),
        "done": done,
        "info": info
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
