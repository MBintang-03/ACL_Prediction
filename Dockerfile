FROM python:3.12-slim

# 1) System libs needed by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) Copy your code
COPY api.py .

# 4) Railway exposes $PORT; gunicorn will listen on it
ENV PORT=8080
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8080", "api:app"]
