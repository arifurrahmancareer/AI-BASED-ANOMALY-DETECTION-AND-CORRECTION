FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY app.py .
COPY yolov8n.pt .

# Create .streamlit config directory
RUN mkdir -p /app/.streamlit

# Create streamlit config
RUN echo "\
[server]\n\
port = 8501\n\
headless = true\n\
runOnSave = true\n\
" > /app/.streamlit/config.toml

CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
