# BentoML Extraction Service Dockerfile
# Uses vLLM with CUDA 12.1

FROM nvidia/cuda:12.1-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    BENTOML_HOME=/home/bentoml

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    ffmpeg \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash bentoml

WORKDIR /app

# Copy pyproject.toml and install dependencies
COPY pyproject.toml ./
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && \
    pip install --no-cache-dir "bentoml>=1.3.0" && \
    pip install --no-cache-dir "vllm==0.6.0" --extra-index-url https://download.pytorch.org/whl/cu121

# Copy service code
COPY services/ ./services/

# Create model cache directory
RUN mkdir -p /home/bentoml/models && chown -R bentoml:bentoml /home/bentoml

USER bentoml

# Pre-download model (optional - can also be done at runtime)
# RUN python -c "from vllm import LLM; LLM(model='Qwen/Qwen2.5-7B-Instruct-AWQ', quantization='awq')"

EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Run BentoML service
CMD ["bentoml", "serve", "services.extraction:ExtractionService", "--host", "0.0.0.0", "--port", "3000"]