import requests

class TradingEnvClient:
    """
    Client for interacting with the Trading Environment Server.
    """
    def __init__(self, base_url="http://localhost:7860"):
        self.base_url = base_url

    def reset(self):
        response = requests.post(f"{self.base_url}/reset")
        response.raise_for_status()
        return response.json()

    def step(self, action_str):
        payload = {"action": action_str}
        response = requests.post(f"{self.base_url}/step", json=payload)
        response.raise_for_status()
        data = response.json()
        return data["observation"], data["reward"], data["done"], data["info"]
