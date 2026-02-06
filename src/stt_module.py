"""
Speech-to-Text (STT) modülü
OpenAI Whisper kullanarak ses tanıma
"""

import whisper
import logging
from typing import Optional, Dict
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperSTT:
    """
    OpenAI Whisper tabanlı Speech-to-Text sınıfı
    Türkçe ses tanıma için optimize edilmiştir
    """
    
    def __init__(self, model_size: str = "base", device: Optional[str] = None):
        """
        Whisper modelini başlatır
        
        Args:
            model_size: Model boyutu ('tiny', 'base', 'small', 'medium', 'large')
            device: Hesaplama cihazı ('cuda', 'cpu' veya None - otomatik)
        """
        self.model_size = model_size
        
        # Cihazı belirle
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Whisper modeli yükleniyor: {model_size} (cihaz: {self.device})")
        
        try:
            self.model = whisper.load_model(model_size, device=self.device)
            logger.info(f"Whisper {model_size} modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Whisper model yükleme hatası: {e}")
            raise
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "tr",
        task: str = "transcribe",
        verbose: bool = False
    ) -> Dict:
        """
        Ses dosyasını metne dönüştürür
        
        Args:
            audio_path: Ses dosyası yolu
            language: Dil kodu (varsayılan: 'tr' - Türkçe)
            task: 'transcribe' veya 'translate'
            verbose: Detaylı çıktı
            
        Returns:
            Dict: {
                'text': str,           # Tam metin
                'segments': List,      # Zaman damgalı segmentler
                'language': str        # Algılanan dil
            }
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Ses dosyası bulunamadı: {audio_path}")
        
        try:
            logger.info(f"Ses tanıma başlatılıyor: {audio_path}")
            
            result = self.model.transcribe(
                audio_path,
                language=language,
                task=task,
                verbose=verbose,
                fp16=False  # CPU uyumluluğu için
            )
            
            text = result["text"].strip()
            logger.info(f"Ses tanıma tamamlandı. Metin uzunluğu: {len(text)} karakter")
            
            return {
                'text': text,
                'segments': result.get('segments', []),
                'language': result.get('language', language)
            }
            
        except Exception as e:
            logger.error(f"Ses tanıma hatası: {e}")
            raise
    
    def transcribe_from_bytes(
        self,
        audio_bytes: bytes,
        language: str = "tr"
    ) -> str:
        """
        Bellek içi ses verisini metne dönüştürür
        
        Args:
            audio_bytes: Ses verisi (bytes)
            language: Dil kodu
            
        Returns:
            str: Tanınan metin
        """
        import tempfile
        import os
        
        # Geçici dosya oluştur
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        try:
            result = self.transcribe(tmp_path, language=language)
            return result['text']
        finally:
            # Geçici dosyayı temizle
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def get_model_info(self) -> Dict:
        """
        Model bilgilerini döndürür
        
        Returns:
            Dict: Model bilgileri
        """
        return {
            'model_size': self.model_size,
            'device': self.device,
            'cuda_available': torch.cuda.is_available()
        }


# Model boyutu bilgileri
MODEL_INFO = {
    'tiny': {
        'size': '~75 MB',
        'speed': 'Çok Hızlı',
        'accuracy': 'Düşük',
        'description': 'En hızlı ama en az doğru'
    },
    'base': {
        'size': '~150 MB',
        'speed': 'Hızlı',
        'accuracy': 'Orta',
        'description': 'Hız ve doğruluk dengesi (önerilen)'
    },
    'small': {
        'size': '~500 MB',
        'speed': 'Orta',
        'accuracy': 'İyi',
        'description': 'Daha yüksek doğruluk'
    },
    'medium': {
        'size': '~1.5 GB',
        'speed': 'Yavaş',
        'accuracy': 'Çok İyi',
        'description': 'Yüksek doğruluk'
    },
    'large': {
        'size': '~3 GB',
        'speed': 'Çok Yavaş',
        'accuracy': 'En İyi',
        'description': 'En yüksek doğruluk ama çok yavaş'
    }
}


if __name__ == "__main__":
    # Test kodu
    stt = WhisperSTT(model_size="base")
    print("Whisper STT modülü başarıyla yüklendi!")
    print(f"Model bilgileri: {stt.get_model_info()}")

