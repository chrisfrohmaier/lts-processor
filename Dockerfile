FROM python:3.9-slim

# Install system dependencies needed for compiling python packages (like healpy)
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (strategies, scripts, etc.)
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "appOverlay.py", "--server.port=8501", "--server.address=0.0.0.0"]
