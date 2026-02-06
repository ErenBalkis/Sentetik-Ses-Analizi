"""
ML Model Eğitim Scripti - Requirements.txt Uyumlu Sürüm
Bu script, özellik sayısındaki değişiklikleri (382 -> 573) modele tanıtmak için
yeniden eğitim yapar.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import soundfile as sf

# Modülleri import et
sys.path.insert(0, os.path.dirname(__file__))
from src.ml_detector import AudioFeatureExtractor, DeepfakeDetector
from src.tts_module import CoquiTTS
from src.utils import PathManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Eğitim için veri seti oluşturur"""
    
    def __init__(self, base_dir: str = "."):
        self.path_manager = PathManager(base_dir)
        self.path_manager.create_directories()
    
    def generate_synthetic_data(
        self,
        num_samples: int = 50,
        reference_voice_path: str = None
    ):
        """TTS ile yapay ses örnekleri üretir"""
        logger.info(f"{num_samples} yapay ses örneği üretiliyor...")
        
        tts = CoquiTTS()
        
        sample_texts = [
            "Merhaba, nasılsınız bugün?",
            "Hava durumu oldukça güzel.",
            "Yapay zeka çok hızlı gelişiyor.",
            "Bugün çok güzel bir gün.",
            "Sinyal işleme dersi çok ilginç.",
            "Makine öğrenmesi geleceğin teknolojisi.",
            "Ses tanıma sistemleri çok gelişti.",
            "Derin öğrenme modelleri başarılı.",
            "Proje çalışması devam ediyor.",
            "Sonuçlar oldukça tatmin edici.",
            "Teknoloji hayatımızı kolaylaştırıyor.",
            "Bilgisayar bilimleri çok geniş bir alan.",
            "Veri bilimi önemli bir dalı.",
            "Algoritmalar problem çözümünde önemli.",
            "Yazılım geliştirme sürekli evrim geçiriyor.",
            "Programlama dilleri çok çeşitli.",
            "Web teknolojileri hızla ilerliyor.",
            "Mobil uygulamalar hayatın parçası oldu.",
            "Bulut bilişim altyapıları gelişiyor.",
            "Siber güvenlik çok kritik bir konu.",
        ]
        
        if reference_voice_path is None:
            logger.warning("Referans ses belirtilmedi.")
            return
        
        synthetic_dir = self.path_manager.training_data_dir / "synthetic"
        synthetic_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(num_samples), desc="Yapay ses üretiliyor"):
            text = sample_texts[i % len(sample_texts)]
            output_path = synthetic_dir / f"synthetic_{i:03d}.wav"
            
            try:
                tts.synthesize(
                    text=text,
                    speaker_wav=reference_voice_path,
                    language="tr",
                    output_path=str(output_path)
                )
            except Exception as e:
                logger.error(f"Ses üretim hatası (örnek {i}): {e}")
                continue
        
        logger.info(f"Yapay sesler oluşturuldu: {synthetic_dir}")
    
    def prepare_dataset(self):
        """Eğitim ve test verilerini hazırlar"""
        logger.info("Veri seti özellik çıkarımı başlıyor...")
        
        extractor = AudioFeatureExtractor()
        real_dir = self.path_manager.training_data_dir / "real"
        synthetic_dir = self.path_manager.training_data_dir / "synthetic"
        
        X = []
        y = []
        
        # Gerçek sesler (Etiket: 0)
        if real_dir.exists():
            real_files = list(real_dir.glob("*.wav"))
            for audio_file in tqdm(real_files, desc="Gerçek sesler işleniyor"):
                try:
                    features = extractor.extract_features(str(audio_file))
                    X.append(features)
                    y.append(0)
                except Exception as e:
                    logger.error(f"Hata ({audio_file.name}): {e}")
        
        # Yapay sesler (Etiket: 1)
        if synthetic_dir.exists():
            synthetic_files = list(synthetic_dir.glob("*.wav"))
            for audio_file in tqdm(synthetic_files, desc="Yapay sesler işleniyor"):
                try:
                    features = extractor.extract_features(str(audio_file))
                    X.append(features)
                    y.append(1)
                except Exception as e:
                    logger.error(f"Hata ({audio_file.name}): {e}")
        
        if len(X) == 0:
            raise ValueError("Hiç eğitim verisi bulunamadı!")
        
        # DataFrame'e çevirmeden numpy array olarak döndür
        return np.array(X), np.array(y)


def main():
    logger.info("=" * 60)
    logger.info("MODEL EGITIM VE GUNCELLEME MODULU")
    logger.info("=" * 60)
    
    # 1. Veri Hazırlığı
    dataset_gen = DatasetGenerator()
    
    # Referans ses kontrolü ve veri üretimi (gerekirse)
    ref_voice_dir = Path("data/reference_voices")
    ref_voices = list(ref_voice_dir.glob("*.wav")) if ref_voice_dir.exists() else []
    
    if len(ref_voices) == 0:
        logger.warning("Referans ses yok! Demo veri seti kullanilsin mi?")
        user_input = input("Demo veri seti olustur (e/h): ").lower()
        if user_input == 'e':
            create_demo_data()
    else:
        logger.info(f"{len(ref_voices)} referans ses bulundu.")
        user_input = input("Yeni yapay sesler uretilsin mi? (e/h): ").lower()
        if user_input == 'e':
            num = int(input("Kac adet? (orn: 50): ") or 50)
            dataset_gen.generate_synthetic_data(num, str(ref_voices[0]))
    
    # Özellikleri çıkar
    X, y = dataset_gen.prepare_dataset()
    logger.info(f"Egitim icin hazir veri: {len(X)} ornek")
    logger.info(f"Ozellik vektoru boyutu: {X.shape[1]} (Beklenen: 573)")

    # 2. Modelleri Karşılaştır
    models_to_try = ["svm", "random_forest", "lightgbm"]
    results_summary = []
    
    best_accuracy = 0.0
    best_model_name = ""
    best_detector = None
    
    logger.info("\n" + "-" * 40)
    logger.info("MODELLER EGITILIYOR")
    logger.info("-" * 40)

    for model_name in models_to_try:
        logger.info(f"\n>> Model Egitiliyor: {model_name.upper()}...")
        
        detector = DeepfakeDetector(model_type=model_name)
        # CV ile eğitim yap
        res = detector.train(X, y, test_size=0.2, perform_cv=True)
        
        # Sonuçları kaydet
        acc = res['accuracy']
        cv_mean = res['cv_mean']
        
        results_summary.append({
            "Model": model_name.upper(),
            "Test Acc": f"%{acc*100:.2f}",
            "CV Acc": f"%{cv_mean*100:.2f}",
            "Std Dev": f"{res['cv_std']:.4f}"
        })
        
        logger.info(f"{model_name.upper()} Basarisi: %{acc*100:.2f}")
        
        # En iyiyi seç
        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = model_name
            best_detector = detector
            
    # 3. Raporlama ve Kayıt
    logger.info("\n" + "=" * 60)
    logger.info("KARSILASTIRMA SONUCU")
    logger.info("=" * 60)
    
    # Pandas DataFrame ile tabloyu bas (Tabulate gerektirmez)
    df_results = pd.DataFrame(results_summary)
    print("\n")
    print(df_results.to_string(index=False))
    print("\n")
    
    logger.info(f"KAZANAN MODEL: {best_model_name.upper()} (Dogruluk: %{best_accuracy*100:.2f})")
    
    # En iyi modeli kaydet
    if best_detector:
        model_path = Path("models/deepfake_detector.pkl")
        model_path.parent.mkdir(exist_ok=True)
        best_detector.save_model(str(model_path))
        logger.info(f"En iyi model kaydedildi: {model_path}")
        logger.info("Uygulama artik guncel modeli (573 ozellikli) kullanacak.")

def create_demo_data():
    """Demo amaçlı veri oluşturur"""
    real_dir = Path("data/training_data/real")
    synthetic_dir = Path("data/training_data/synthetic")
    real_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 16000
    duration = 3
    logger.info("Demo ses dosyalari olusturuluyor...")
    
    # Basit sinyal üretimi
    for i in range(30):
        t = np.linspace(0, duration, sr * duration)
        # Gerçek: Daha karmaşık ve gürültülü
        sig_real = np.sin(2 * np.pi * 200 * t) + np.random.normal(0, 0.1, len(t))
        # Yapay: Daha temiz ve düzenli
        sig_fake = np.sin(2 * np.pi * 220 * t) + np.random.normal(0, 0.01, len(t))
        
        sf.write(real_dir / f"real_{i:03d}.wav", sig_real, sr)
        sf.write(synthetic_dir / f"synthetic_{i:03d}.wav", sig_fake, sr)

if __name__ == "__main__":
    main()
