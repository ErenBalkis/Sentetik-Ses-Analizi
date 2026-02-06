"""
Proje doğrulama scripti
Tüm modüllerin düzgün çalıştığını kontrol eder
"""

import sys
import os

print("=" * 60)
print("SES TESPİT SİSTEMİ - MODÜL DOĞRULAMA")
print("=" * 60)

# Python versiyonu kontrolü
print(f"\n✓ Python versiyonu: {sys.version}")

# Modül importları
print("\n📦 Modüller kontrol ediliyor...")

try:
    import numpy
    print("✓ numpy")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import librosa
    print("✓ librosa")
except ImportError as e:
    print(f"✗ librosa: {e}")

try:
    import sklearn
    print("✓ scikit-learn")
except ImportError as e:
    print(f"✗ scikit-learn: {e}")

try:
    import lightgbm
    print("✓ lightgbm")
except ImportError as e:
    print(f"✗ lightgbm: {e}")

try:
    import streamlit
    print("✓ streamlit")
except ImportError as e:
    print(f"✗ streamlit: {e}")

try:
    import plotly
    print("✓ plotly")
except ImportError as e:
    print(f"✗ plotly: {e}")

try:
    import soundfile
    print("✓ soundfile")
except ImportError as e:
    print(f"✗ soundfile: {e}")

try:
    import pydub
    print("✓ pydub")
except ImportError as e:
    print(f"✗ pydub: {e}")

try:
    import torch
    print(f"✓ torch (CUDA: {torch.cuda.is_available()})")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import whisper
    print("✓ openai-whisper")
except ImportError as e:
    print(f"✗ openai-whisper: {e}")

try:
    import TTS
    print("✓ TTS (Coqui)")
except ImportError as e:
    print(f"✗ TTS: {e}")

# Proje modülleri
print("\n🔧 Proje modülleri kontrol ediliyor...")

try:
    from src.utils import AudioUtils, PathManager
    print("✓ src.utils")
except ImportError as e:
    print(f"✗ src.utils: {e}")

try:
    from src.stt_module import WhisperSTT
    print("✓ src.stt_module")
except ImportError as e:
    print(f"✗ src.stt_module: {e}")

try:
    from src.tts_module import CoquiTTS
    print("✓ src.tts_module")
except ImportError as e:
    print(f"✗ src.tts_module: {e}")

try:
    from src.ml_detector import AudioFeatureExtractor, DeepfakeDetector
    print("✓ src.ml_detector")
except ImportError as e:
    print(f"✗ src.ml_detector: {e}")

# Dizin yapısı
print("\n📁 Dizin yapısı kontrol ediliyor...")

required_dirs = [
    "data",
    "data/reference_voices",
    "data/test_audio",
    "data/training_data",
    "models",
    "src"
]

for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f"✓ {dir_path}/")
    else:
        print(f"✗ {dir_path}/ (eksik)")
        os.makedirs(dir_path, exist_ok=True)
        print(f"  → Oluşturuldu")

# Dosya yapısı
print("\n📄 Ana dosyalar kontrol ediliyor...")

required_files = [
    "app.py",
    "train_model.py",
    "requirements.txt",
    "README.md",
    "src/__init__.py",
    "src/utils.py",
    "src/stt_module.py",
    "src/tts_module.py",
    "src/ml_detector.py"
]

for file_path in required_files:
    if os.path.exists(file_path):
        print(f"✓ {file_path}")
    else:
        print(f"✗ {file_path} (eksik)")

print("\n" + "=" * 60)
print("DOĞRULAMA TAMAMLANDI")
print("=" * 60)

print("\n📋 Sonraki Adımlar:")
print("1. Bağımlılıkları yükleyin: pip install -r requirements.txt")
print("2. ML modelini eğitin: python train_model.py")
print("3. Uygulamayı çalıştırın: streamlit run app.py")

