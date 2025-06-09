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
    DATA_PATH=/app/data/processed

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/models /app/data/processed

# Copy model file first
COPY models/model_5000.pth /app/models/

# Copy the rest of the application
COPY . .

RUN useradd -m appuser

USER appuser

EXPOSE 8001

# # Create necessary directories and ensure they exist
# RUN mkdir -p data/raw data/processed data/interim data/external \
#     models \
#     reports/figures && \
#     touch data/raw/.gitkeep \
#     data/processed/.gitkeep \
#     data/interim/.gitkeep \
#     data/external/.gitkeep \
#     models/.gitkeep \
#     reports/figures/.gitkeep

# Set default command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
