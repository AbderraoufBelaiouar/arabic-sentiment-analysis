# ================================================================
# STAGE 1: Builder — install all dependencies
# ================================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install uv for fast, reliable dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only the files needed to install dependencies first
# (layer caching: this layer only re-runs when these files change)
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install CPU-only PyTorch first (saves ~2GB vs CUDA version on PyPI)
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the project dependencies
RUN uv pip install --system --no-cache .

# ================================================================
# STAGE 2: Runtime — minimal image with only what's needed
# ================================================================
FROM python:3.12-slim AS runtime

# Security: don't run as root
RUN useradd --create-home --uid 1000 appuser

# Copy installed Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy model files (owned by appuser so the app can read them)
COPY --chown=appuser:appuser models/ /app/models/

WORKDIR /app
USER appuser

# --- Environment Variables ---
# Tell the app where the ONNX model is stored
ENV MODEL_PATH=/app/models/arabert-twitter-onnx

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "arabic_sentiment.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
