# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    MODEL_PATH=/app/models/model_5000.pth \
    DATA_PATH=/app/data/processed \
    VOCAB_ROOT_FOLDER=/app/data/Cora/CoRA_Raw \
    FORCE_CUDA=0 \
    TORCH_CUDA_ARCH_LIST="None"

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Create non-root user first
RUN useradd -m appuser

# Install Python dependencies and download NLTK data
RUN pip3 install --no-cache-dir -r requirements.txt && \
    mkdir -p /home/appuser/nltk_data && \
    chown -R appuser:appuser /home/appuser/nltk_data && \
    su - appuser -c "python3 -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('punkt_tab')\""

# Create necessary directories with proper permissions
RUN mkdir -p /app/models /app/data/processed /app/data/Cora/CoRA_Raw /tmp && \
    chmod -R 777 /tmp

# Copy model file
COPY models/model_5000.pth /app/models/

COPY models/model_5000_state_dict.pth /app/models/

# Copy the rest of the application
COPY . .

# Set permissions for app directory
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8080

# Set default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
