FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Node.js for React frontend
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-server.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application code
COPY . .

# Build React frontend
WORKDIR /app/frontend
RUN npm install
RUN npm run build

# Back to main directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p cyberprint/data/output
RUN mkdir -p static

# Copy built frontend to static directory for serving
RUN cp -r frontend/build/* static/

# Expose port (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Run the application
CMD ["python", "server.py"]
