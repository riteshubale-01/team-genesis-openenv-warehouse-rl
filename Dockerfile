# Dockerfile — Warehouse OpenEnv
# Hugging Face Spaces compatible (2 vCPU / 8GB RAM safe)

FROM python:3.11-slim

# Non-root user for HF Spaces
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HF Spaces: port 7860
EXPOSE 7860

# Environment defaults
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

USER appuser

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
