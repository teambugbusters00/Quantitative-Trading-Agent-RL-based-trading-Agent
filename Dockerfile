# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variables for non-interactive installs
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Hugging Face Spaces for Docker listen on port 7860
EXPOSE 7860

# Define environment variables (these should also be set in HF Spaces secrets)
ENV API_BASE_URL="https://router.huggingface.co/v1"
ENV MODEL_NAME="gpt-4.1-mini"

# Run app.py when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
