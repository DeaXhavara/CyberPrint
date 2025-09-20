FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p cyberprint/data/output

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["python", "start_backend.py"]
