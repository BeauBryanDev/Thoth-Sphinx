# SphinxEyes Backend — Dockerfile
# First stage: Build
FROM python:3.12-slim AS builder
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
 
# Build deps for compiled wheels (numpy, pillow, asyncpg, cryptography)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*
 
# Create venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
 
# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt
 
# Stage 2: Runtime
 
FROM python:3.12-slim AS runtime
 
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"
 
# Runtime libs only (no build-essential)
# libpq5    : asyncpg runtime
# libgomp1  : OpenMP runtime for onnxruntime
# libgl1, libglib2.0-0 : opencv-python-headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*
 
# Non-root user
RUN groupadd --system --gid 1000 sphinx && \
    useradd  --system --uid 1000 --gid sphinx --shell /bin/bash --create-home sphinx
 
# Copy venv from builder
COPY --from=builder --chown=sphinx:sphinx /opt/venv /opt/venv
 
# App directory
WORKDIR /app
 
# Copy source  
COPY --chown=sphinx:sphinx vision/                 /app/vision/
COPY --chown=sphinx:sphinx app/                    /app/app/
 
# Artifacts are mounted at runtime  
 
USER sphinx
 
EXPOSE 8000
 
# Healthcheck asks the pipeline to load the ONNX model and verify it is
# healthy even if the pipeline is still loading. Use /health/ready in
# orchestrators that need to gate traffic on full readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health/live || exit 1
 
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
 