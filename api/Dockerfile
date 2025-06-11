FROM python:3.10-slim

# Install system dependencies for scikit-learn
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model.joblib .
COPY main.py .

# Cloud Run required settings
ENV PORT=8080
ENV PYTHONUNBUFFERED=True
EXPOSE $PORT

# Production-ready server command
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:${PORT}", "main:app"]
