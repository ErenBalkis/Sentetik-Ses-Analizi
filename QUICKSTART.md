# Ses Sentezleme ve Sahte Ses Tespiti Sistemi - Hızlı Başlangıç Kılavuzu

## 🚀 Hızlı Başlangıç

### 1. Proje Dizinine Git
```bash
cd /home/eren/.gemini/antigravity/scratch/ses-tespit-sistemi
```

### 2. Otomatik Kurulum (Önerilen)
```bash
bash start.sh
```

Bu script:
- ✓ Sanal ortam oluşturur
- ✓ Tüm bağımlılıkları yükler
- ✓ Dizin yapısını oluşturur
- ✓ Modülleri doğrular

### 3. ML Modelini Eğit

İlk kullanımda ML modelini eğitmeniz gerekir:

```bash
python train_model.py
```

**Not:** Script size demo veri seti oluşturma seçeneği sunacaktır.

### 4. Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcınızda otomatik olarak `http://localhost:8501` adresi açılacaktır.

---

## 📖 Kullanım Rehberi

### Adım 1: Modelleri Yükle

Sidebar'dan sırayla tıklayın:
1. 🎤 Whisper STT Yükle
2. 🗣️ Coqui TTS Yükle  
3. 🤖 ML Model Yükle

**İlk kullanımda:** Whisper (~150MB) ve TTS (~2GB) modelleri otomatik indirilecektir. Bu işlem 5-10 dakika sürebilir.

### Adım 2: Ses Tanıma (Tab 1)

1. Bir Türkçe ses dosyası yükleyin (WAV, MP3, OGG)
2. "Metne Dönüştür" butonuna tıklayın
3. Whisper metni otomatik tanıyacaktır

### Adım 3: Yapay Ses Üretimi (Tab 2)

1. Referans ses yükleyin (3-10 saniye önerilir)
2. Sentezlenecek metni girin veya STT metnini kullanın
3. "Yapay Ses Üret" butonuna tıklayın
4. Üretilen yapay sesi dinleyin

### Adım 4: İnteraktif Test (Tab 3)

1. "Rastgele Test Sesi Oluştur" butonuna tıklayın
2. Test sesini dinleyin
3. "YAPAY SES" veya "GERÇEK SES" seçin
4. Sonucu görün ve ML modelinin tahminini karşılaştırın

### Adım 5: Sonuçları İnceleyin (Tab 4)

- İnsan ve makine doğruluk oranları
- İnteraktif grafik
- Hangisi daha başarılı analizi

---

## 📊 Özellikler

✅ **Speech-to-Text:** OpenAI Whisper ile Türkçe ses tanıma  
✅ **Text-to-Speech:** Coqui TTS XTTS-v2 ile voice cloning  
✅ **Deepfake Detection:** LightGBM ile 420 özellikli sahte ses tespiti  
✅ **İnteraktif Test:** Yapay/Gerçek tahmin oyunu  
✅ **Karşılaştırma:** İnsan vs Makine analizi  
✅ **Türkçe Arayüz:** Tamamen Türkçe GUI  

---

## 🛠️ Sorun Giderme

### Problem: Bağımlılık yükleme hatası
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Problem: TTS modeli yüklenmiyor
- İnternet bağlantınızı kontrol edin
- İlk kullanımda ~2GB indirme gerekir
- Sabırlı olun, bu işlem 5-10 dakika sürebilir

### Problem: ML modeli bulunamadı
```bash
python train_model.py
```
komutunu çalıştırarak modeli eğitin.

### Problem: CUDA hatası
GPU kullanamıyorsanız CPU ile çalışacaktır (daha yavaş).

---

## 📁 Proje Yapısı

```
ses-tespit-sistemi/
├── app.py                  # Ana uygulama
├── train_model.py          # Model eğitimi
├── verify.py               # Doğrulama
├── start.sh                # Kurulum scripti
├── requirements.txt        # Bağımlılıklar
├── README.md              # Dokümantasyon
├── src/                   # Kaynak kodlar
│   ├── stt_module.py      # Whisper STT
│   ├── tts_module.py      # Coqui TTS
│   ├── ml_detector.py     # ML tespiti
│   └── utils.py           # Yardımcılar
├── data/                  # Veri dosyaları
└── models/                # Eğitilmiş modeller
```

---

## 💡 İpuçları

- Referans ses kalitesi ne kadar iyi olursa, yapay ses o kadar gerçekçi olur
- En az 3 saniye, maksimum 30 saniye referans ses kullanın
- Daha fazla test yaparak doğruluğu artırabilirsiniz
- Demo veri seti ile başlayıp sonra kendi ses dosyalarınızı ekleyebilirsiniz

---

## 📞 Destek

Proje dosyaları: `/home/eren/.gemini/antigravity/scratch/ses-tespit-sistemi`

Detaylı dokümantasyon: `README.md`

---

**İyi Kullanımlar! 🎉**
