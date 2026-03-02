<div align="center">

# 🎤 Sentetik Ses Analizi

**Türkçe konuşma tanıma · Ses klonlama · Deepfake ses tespiti**

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📌 Proje Hakkında

**Sentetik Ses Analizi**, gerçek sesler ile yapay zeka tarafından üretilmiş (deepfake) sesler arasındaki farkı hem insanların hem de makine öğrenmesi modellerinin ne kadar iyi ayırt edebildiğini araştıran bir uygulama platformudur.

Uygulama üç ana bileşenden oluşur:

| Bileşen | Teknoloji | Açıklama |
|---|---|---|
| 🎙️ Konuşma Tanıma (STT) | OpenAI Whisper | Türkçe sesi metne dönüştürür |
| 🗣️ Ses Sentezleme (TTS) | Coqui XTTS-v2 | Referans sesten voice cloning ile yeni ses üretir |
| 🤖 Deepfake Tespiti | LightGBM + XAI | Sesin gerçek mi yapay mı olduğunu sınıflandırır |

---

## ✨ Özellikler

- 🎙️ **Tarayıcı üzerinden mikrofon kaydı** — disk'e otomatik kayıt yok
- 🗣️ **Zero-shot voice cloning** — 3-10 saniyelik referans ses yeterli
- 🤖 **Açıklanabilir yapay zeka (XAI)** — hangi ses özelliği tespiti etkiliyor?
- 🎮 **İnteraktif tahmin oyunu** — insan vs makine doğruluk karşılaştırması
- 🐳 **Tek komutla Docker ile çalışma**
- 🇹🇷 **Tamamen Türkçe arayüz**

---

## 🚀 Hızlı Başlangıç

### Docker ile (Önerilen)

```bash
# 1. Repoyu klonla
git clone https://github.com/ErenBalkis/Sentetik-Ses-Analizi.git
cd Sentetik-Ses-Analizi

# 2. Image'ı build et (~5-10 dakika, ilk seferinde)
docker build -t sentetik-ses-analizi .

# 3. Çalıştır
docker run -p 8501:8501 sentetik-ses-analizi
```

Tarayıcıda [http://localhost:8501](http://localhost:8501) adresini aç.

> **Not:** XTTS-v2 modeli (~1.8 GB) ilk çalıştırmada otomatik indirilir. İnternet bağlantısı gerekli.

---

### Manuel Kurulum (Python)

<details>
<summary>Adımları göster</summary>

**Gereksinimler**
- Python 3.10 veya 3.11 (3.12+ desteklenmez)
- ffmpeg (sistem bağımlılığı)
- En az 8 GB RAM, 5 GB disk

```bash
# 1. Repoyu klonla
git clone https://github.com/ErenBalkis/Sentetik-Ses-Analizi.git
cd Sentetik-Ses-Analizi

# 2. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate   # Linux / macOS
# venv\Scripts\activate    # Windows

# 3. PyTorch'u önce kur (sürüm kritik)
pip install "torch>=2.1.0,<2.6.0" "torchaudio>=2.1.0,<2.6.0" --index-url https://download.pytorch.org/whl/cpu

# 4. Geri kalan bağımlılıkları kur
pip install -r requirements.txt

# 5. Klasör yapısını oluştur
mkdir -p data/reference_voices data/test_audio data/training_data models
```

**Uygulamayı başlat**

```bash
# (İlk kullanımda) ML modelini eğit
python train_model.py

# Streamlit uygulamasını başlat
streamlit run app.py
```

</details>

---

## 🗂️ Proje Yapısı

```
Sentetik-Ses-Analizi/
├── src/
│   ├── ui/
│   │   ├── tab_stt.py          # Konuşma tanıma sekmesi
│   │   ├── tab_tts.py          # Ses sentezleme sekmesi
│   │   ├── tab_test.py         # İnteraktif test sekmesi
│   │   └── tab_comparison.py   # Karşılaştırma & XAI sekmesi
│   ├── stt_module.py           # Whisper STT sarmalayıcı sınıf
│   ├── tts_module.py           # Coqui TTS sarmalayıcı sınıf
│   ├── ml_detector.py          # LightGBM dedektör + XAI
│   └── utils.py                # Yardımcı araçlar, geçici dosya yönetimi
├── data/
│   ├── reference_voices/       # Voice cloning için referans sesler
│   ├── test_audio/             # Test ses dosyaları
│   └── training_data/          # ML eğitim verisi
├── models/                     # Eğitilmiş model dosyaları (.pkl)
├── app.py                      # Streamlit giriş noktası
├── train_model.py              # ML model eğitim scripti
├── requirements.txt            # Sürüm sabitli bağımlılıklar
├── Dockerfile                  # Production Docker image
└── .dockerignore
```

---

## 🧠 Teknik Mimari

```
Kullanıcı Sesi
      │
      ▼
┌─────────────┐     ┌──────────────────┐
│  Whisper    │────▶│  Transkript      │
│  (STT)      │     │  (Türkçe metin)  │
└─────────────┘     └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐     ┌──────────────────┐
                    │  Coqui XTTS-v2   │────▶│  Sentetik Ses    │
                    │  (Voice Cloning) │     │  (WAV çıktı)     │
                    └──────────────────┘     └──────────────────┘
                                                      │
                              ┌───────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  LightGBM        │
                    │  Dedektör        │──▶  Gerçek / Yapay
                    │  (MFCC, Mel,     │
                    │   Chroma, ZCR…)  │
                    └──────────────────┘
```

### ML Özellik Seti

| Özellik Grubu | Açıklama |
|---|---|
| MFCC (40 katsayı) | Konuşma karakteristiği |
| Mel Spektrogram | Frekans enerji dağılımı |
| Chroma | Perde (pitch) özellikleri |
| Spectral Contrast | Harmonik yapı |
| Zero Crossing Rate | Gürültü karakteri |
| RMS Energy | Ses yoğunluğu |

---

## ⚠️ Bilinen Sürüm Kısıtlamaları

| Paket | Sürüm Aralığı | Neden |
|---|---|---|
| `torch` / `torchaudio` | `>=2.1.0, <2.6.0` | PyTorch 2.6+ XTTS-v2 model yüklemesini bozuyor |
| `numpy` | `>=1.24.0, <2.0.0` | `openai-whisper` NumPy 2.x ile uyumsuz |
| `transformers` | `>=4.40.0, <4.45.0` | `isin_mps_friendly` 4.45'te kaldırıldı; coqui-tts bunu kullanıyor |
| Python | `3.10` veya `3.11` | coqui-tts 3.12+ desteklemiyor |

---

## 🐞 Sorun Giderme

<details>
<summary><b>TTS modeli indirilmiyor / EOFError</b></summary>

Docker ortamında `COQUI_TOS_AGREED=1` env değişkeni eksik olabilir.

```bash
<<<<<<< HEAD
docker run -e COQUI_TOS_AGREED=1 -p 8501:8501 sentetik-ses-analizi
=======
# Image oluştur
docker build -t Sentetik-Ses-Analizi .

# Çalıştır
docker run -p 8501:8501 Sentetik-Ses-Analizi

# Tarayıcıda aç: http://localhost:8501
>>>>>>> c9b636d7129c6436bb62eb98456c620114335968
```

</details>

<details>
<summary><b>Whisper çok yavaş</b></summary>

CUDA destekli GPU yoksa `base` model yerine `tiny` kullanın. `src/ui/tab_stt.py` içinde `model_size` parametresini değiştirin.

</details>

<details>
<summary><b>OutOfMemory hatası</b></summary>

XTTS-v2 yaklaşık **4 GB VRAM** ister. CPU modunda bellek kullanımı daha yüksektir. Minimum 8 GB RAM önerilir.

</details>

<details>
<summary><b>pkg_resources ModuleNotFoundError (pip install sırasında)</b></summary>

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

</details>

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.

---

<div align="center">

**[Eren Balkış](https://github.com/ErenBalkis)** tarafından geliştirilmiştir.

</div>
