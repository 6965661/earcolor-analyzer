FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    git \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone --depth 1 https://github.com/Jeremyszs/chord-engine.git /app/chord-engine
WORKDIR /app/chord-engine
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# EarColor override: keep trained BTC inference while avoiding the duplicate
# full-song chroma/beat pass that can exceed a 1 GB Railway instance.
COPY railway_pipeline.py /app/chord-engine/api/services/pipeline.py

ENV PYTHONUNBUFFERED=1
ENV XDG_CACHE_HOME=/tmp/cache
ENV TMPDIR=/tmp
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
