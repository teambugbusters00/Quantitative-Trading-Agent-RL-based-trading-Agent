---
title: META-ENV Trading Agent
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# RL-Based Trading Agent (OpenEnv Compatible)

This Space runs a quantitative trading agent simulations that follow the **OpenEnv** benchmarking format.

## Features
- **Simulation**: Uses real-world market data (BTC-USD).
- **LLM Reasoning**: Decisions are made by an LLM-based agent.
- **Gymnasium Interface**: Standardized trading environment.

## API Endpoints
- `/run`: Trigger a fresh trading episode and see the OpenEnv logs in real-time.
- `/health`: Health check endpoint.

## Local Execution
To run locally, clone the repo and use:
```bash
$env:HF_TOKEN = "your_token"
python inference.py
```
