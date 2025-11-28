# Dockerfile for synthetic aerodynamic dataset generation
# Includes GPU support for ML augmentation (Tier 4)

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    build-essential \
    libhdf5-dev \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY . .

# Create output directory
RUN mkdir -p /data/synthetic_dataset

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python3", "batch_orchestrator.py", "--tier1", "1000", "--workers", "4", "--output", "/data/synthetic_dataset"]
