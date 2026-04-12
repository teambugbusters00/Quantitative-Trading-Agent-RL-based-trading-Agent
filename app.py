from fastapi import FastAPI
import subprocess
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {
        "status": "Running",
        "project": "RL-Based Trading Agent",
        "version": "1.0.0",
        "endpoints": {
            "/run": "Execute a single trading episode and see logs",
            "/health": "Check space health"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/run")
def run_agent():
    """
    Triggers the inference.py script and returns the captured stdout.
    Note: For a production app, you might want to stream logs.
    """
    try:
        # We use the current environment (including HF_TOKEN, MODEL_NAME, etc.)
        result = subprocess.run(
            ["python", "inference.py"],
            capture_output=True,
            text=True,
            timeout=180 # 3 minute timeout
        )
        
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
    except subprocess.TimeoutExpired:
        return {"error": "Episode timed out"}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces for Docker listen on port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
