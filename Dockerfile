FROM python:3.11-slim

# Metadata
LABEL maintainer="Deepfake Audio Detection System"
LABEL description="Production-ready Streamlit app for deepfake audio detection"

# System dependencies for audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libsndfile1-dev \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# coqui-tts model indirirken TOS onayı ister; headless container'da EOFError verir.
# Bu değişken prompt'u tamamen atlar.
ENV COQUI_TOS_AGREED=1

# Streamlit'e telemetri sorma
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory
WORKDIR /app

# Copy requirements first (for layer caching)
COPY requirements.txt .

# ── Adım 1: pip araçlarını güncelle (setuptools eksikliği openai-whisper build'ini kırıyor)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Adım 2: torch / torchaudio CPU-only — sürüm sabitli, GPU olmayan ortam için
#    torch>=2.6 ile tts_models/xtts_v2 yüklemesi weights_only=True değişikliği nedeniyle bozuluyor.
RUN pip install --no-cache-dir \
    "torch>=2.1.0,<2.6.0" \
    "torchaudio>=2.1.0,<2.6.0" \
    --index-url https://download.pytorch.org/whl/cpu

# ── Adım 3: Kalan bağımlılıkları kur (torch artık mevcut, coqui-tts onu algılar)
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/reference_voices data/test_audio data/training_data models

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--server.fileWatcherType=none"]
