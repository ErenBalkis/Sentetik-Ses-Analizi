# Ses Sentezleme ve Sahte Ses Tespiti Sistemi

Bu proje, kullanıcıdan alınan ses kaydını OpenAI Whisper ile metne dönüştüren, Coqui TTS (XTTS-v2) ile klonlanmış bir sese çeviren ve bir Makine Öğrenmesi modeliyle sesin gerçekliğini test eden Türkçe arayüzlü bir uygulamadır.

## Özellikler

- 🎤 **Speech-to-Text**: OpenAI Whisper ile Türkçe ses tanıma
- 🗣️ **Voice Cloning**: Coqui TTS XTTS-v2 ile ses klonlama
- 🤖 **Deepfake Detection**: LightGBM tabanlı yapay ses tespiti
- 🎮 **İnteraktif Test**: Yapay vs Gerçek ses tahmin oyunu
- 📊 **İnsan vs Makine**: Karşılaştırmalı doğruluk analizi
- 🇹🇷 **Türkçe Arayüz**: Tamamen Türkçe kullanıcı arayüzü

## Sistem Gereksinimleri

- Python 3.8 - 3.11 (3.12 desteklenmez)
- En az 8GB RAM
- GPU (opsiyonel ama önerilir - CUDA desteği)
- En az 5GB disk alanı (modeller için)

## Kurulum

1. **Depoyu klonlayın veya indirin**

2. **Sanal ortam oluşturun (önerilir)**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

3. **Bağımlılıkları yükleyin**
```bash
pip install -r requirements.txt
```

4. **Dizin yapısını oluşturun**
```bash
mkdir -p data/reference_voices data/test_audio data/training_data models
```

## Kullanım

### 1. ML Modelini Eğitin

İlk kullanımda makine öğrenmesi modelini eğitmeniz gerekir:

```bash
python train_model.py
```

Bu script otomatik olarak:
- Coqui TTS ile yapay sesler üretir
- Ses özelliklerini çıkarır
- LightGBM modelini eğitir
- Modeli `models/deepfake_detector.pkl` olarak kaydeder

### 2. Uygulamayı Çalıştırın

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

### 3. Uygulamayı Kullanın

**Adım 1: Ses Yükleme**
- Bir Türkçe ses dosyası yükleyin (WAV, MP3, OGG)
- Whisper otomatik olarak metne dönüştürecektir

**Adım 2: Referans Ses Seçimi**
- Voice cloning için bir referans ses yükleyin (3-10 saniye önerilir)

**Adım 3: Yapay Ses Üretimi**
- "Yapay Ses Üret" butonuna tıklayın
- Coqui TTS klonlanmış sesi oluşturacaktır

**Adım 4: İnteraktif Test**
- Dinlediğiniz sesin yapay mı gerçek mi olduğunu tahmin edin
- ML modeli ile karşılaştırın

**Adım 5: Sonuçları İnceleyin**
- İnsan vs Makine doğruluk oranlarını görüntüleyin

## Proje Yapısı

**Portfolio-Ready Clean Architecture**

```
ses-tespit-sistemi/
├── src/
│   ├── ui/                        # Modular UI Components
│   │   ├── __init__.py
│   │   ├── tab_stt.py            # STT tab with recording
│   │   ├── tab_tts.py            # TTS tab with transient output
│   │   ├── tab_test.py           # Interactive test tab
│   │   └── tab_comparison.py     # Comparison + XAI visualization
│   ├── stt_module.py             # Whisper STT sınıfı
│   ├── tts_module.py             # Coqui TTS sınıfı
│   ├── ml_detector.py            # ML model + XAI feature importance
│   └── utils.py                  # Utilities + TempFileManager
├── models/                        # Eğitilmiş ML modelleri
├── data/
│   ├── reference_voices/         # Referans ses dosyaları
│   ├── test_audio/               # Test ses dosyaları
│   └── training_data/            # ML eğitim verisi
├── app.py                        # Entry point only (~230 lines)
├── train_model.py                # ML model eğitim scripti
├── requirements.txt              # Version constraints
├── Dockerfile                    # Production Docker image
├── .dockerignore                 # Docker optimization
└── README.md                     
```

## Yeni Özellikler

### 🎙️ Tarayıcı Ses Kaydı
- STT ve Test sekmelerinde doğrudan mikrofondan kayıt
- **Disk'e otomatik kayıt YOK** - Clean Code prensibi
- Manuel indirme seçeneği (download button)

### 🗄️ Geçici Dosya Yönetimi
- `TempFileManager` ile otomatik temizlik
- Proje dizininde temp dosya birikmesi önlendi
- Context manager kullanımı

### 🔍 Açıklanabilir Yapay Zeka (XAI)
- Model feature importance görselleştirmesi
- Hangi özelliklerin tespitte önemli olduğunu gösterir
- Plotly interaktif grafik

### 🐳 Docker Desteği
- Production-ready Dockerfile
- Sistem bağımlılıkları dahil (ffmpeg, libsndfile1)
- Kolay deployment

### 🏗️ Temiz Mimari
- Modüler UI bileşenleri
- Separation of Concerns
- Type hints ve docstrings
- SOLID prensipleri

## Docker ile Çalıştırma

```bash
# Image oluştur
docker build -t Sentetik-Ses-Analizi .

# Çalıştır
docker run -p 8501:8501 Sentetik-Ses-Analizi

# Tarayıcıda aç: http://localhost:8501
```


## Teknik Detaylar

### Whisper STT
- Model: `whisper-base` (varsayılan)
- Dil: Türkçe (`tr`)
- Çıktı: Metin transkripti

### Coqui TTS
- Model: XTTS-v2
- Özellik: Voice cloning
- Dil Desteği: Türkçe

### ML Deepfake Detector
- Model: LightGBM Classifier
- Özellikler: MFCC, Mel Spectrogram, Chroma, Spectral Contrast, ZCR, RMS
- Eğitim: Gerçek ve yapay ses örnekleri

## Sorun Giderme

**Problem: TTS modeli yüklenmiyor**
- İlk kullanımda model otomatik indirilir (~2GB)
- İnternet bağlantınızı kontrol edin
- `~/.local/share/tts/` dizininde modelin olduğundan emin olun

**Problem: Whisper çok yavaş**
- GPU kullanımını etkinleştirin (CUDA)
- Daha küçük bir model kullanın: `whisper-tiny` veya `whisper-base`

**Problem: OutOfMemory hatası**
- Daha küçük batch size kullanın
- Model boyutunu küçültün
- RAM'inizi artırın

