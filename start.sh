#!/bin/bash

# Ses Tespiti Sistemi - Hızlı Başlangıç Scripti

echo "================================================="
echo "  SES SENTEZlEME VE SAHTE SES TESPİT SİSTEMİ   "
echo "================================================="
echo ""

# Renk kodları
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Python kontrolü
if ! command -v python3 &> /dev/null
then
    echo -e "${RED}✗ Python3 bulunamadı!${NC}"
    echo "Lütfen Python 3.8-3.11 yükleyin."
    exit 1
fi

echo -e "${GREEN}✓ Python bulundu: $(python3 --version)${NC}"
echo ""

# Sanal ortam kontrolü
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}→ Sanal ortam oluşturuluyor...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Sanal ortam oluşturuldu${NC}"
else
    echo -e "${GREEN}✓ Sanal ortam mevcut${NC}"
fi
echo ""

# Sanal ortamı aktifleştir
echo -e "${YELLOW}→ Sanal ortam aktifleştiriliyor...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Sanal ortam aktif${NC}"
echo ""

# Bağımlılıkları kur
echo -e "${YELLOW}→ Bağımlılıklar yükleniyor... (bu biraz zaman alabilir)${NC}"
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Bağımlılıklar yüklendi${NC}"
else
    echo -e "${RED}✗ Bağımlılık yükleme hatası!${NC}"
    echo "Manuel yükleme için: pip install -r requirements.txt"
    exit 1
fi
echo ""

# Dizin yapısını oluştur
echo -e "${YELLOW}→ Dizin yapısı oluşturuluyor...${NC}"
mkdir -p data/reference_voices
mkdir -p data/test_audio
mkdir -p data/training_data/real
mkdir -p data/training_data/synthetic
mkdir -p models
echo -e "${GREEN}✓ Dizinler oluşturuldu${NC}"
echo ""

# Verifikasyon
echo -e "${YELLOW}→ Modüller kontrol ediliyor...${NC}"
python3 verify.py
echo ""

# ML modeli kontrol
if [ -f "models/deepfake_detector.pkl" ]; then
    echo -e "${GREEN}✓ ML modeli mevcut${NC}"
else
    echo -e "${YELLOW}! ML modeli bulunamadı${NC}"
    echo "  ML modelini eğitmek için: python train_model.py"
fi
echo ""

echo "================================================="
echo "               KURULUM TAMAMLANDI"
echo "================================================="
echo ""
echo -e "${GREEN}📋 Şimdi yapabilecekleriniz:${NC}"
echo ""
echo "1. ML Modelini Eğitin (ilk kullanımda gerekli):"
echo "   python train_model.py"
echo ""
echo "2. Uygulamayı Başlatın:"
echo "   streamlit run app.py"
echo ""
echo "3. Tarayıcınızda açılacak adresi kullanın:"
echo "   http://localhost:8501"
echo ""
echo -e "${YELLOW}Not: İlk çalıştırmada Whisper ve TTS modelleri otomatik indirilecektir.${NC}"
echo "     Bu işlem internet bağlantınıza bağlı olarak 5-10 dakika sürebilir."
echo ""
