# ML Surrogate Agent Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy agent code
COPY . .
COPY ../utils /app/utils
COPY ../config /app/config

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run agent
CMD ["python", "-m", "agents.ml_surrogate.agent"]
