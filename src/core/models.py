from pydantic import BaseModel
from typing import List, Optional

class TradingAction(BaseModel):
    action: str  # buy(), sell(), hold()

class TradingObservation(BaseModel):
    price: float
    rsi: float
    macd: float
    position: str  # NONE, HOLDING
    info: dict

class TradingState(BaseModel):
    balance: float
    current_step: int
    total_reward: float
    is_done: bool
