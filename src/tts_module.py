"""
Text-to-Speech (TTS) modülü
Coqui TTS XTTS-v2 kullanarak voice cloning
"""

import torch
from TTS.api import TTS
import logging
from typing import Optional
from pathlib import Path
import numpy as np
import os
import tempfile
from src.utils import AudioUtils, TTSPreprocessor

logger = logging.getLogger(__name__)

class CoquiTTS:
    """
    Coqui TTS XTTS-v2 tabanlı ses sentezleme sınıfı
    Voice cloning özelliği ile gerçekçi yapay ses üretir
    """
    
    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Coqui TTS XTTS-v2 modeli yükleniyor (cihaz: {self.device})")
        
        try:
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.device)
            logger.info("Coqui TTS modeli başarıyla yüklendi")
        except Exception as e:
            logger.error(f"TTS model yükleme hatası: {e}")
            raise
    
    def synthesize(
        self,
        text: str,
        speaker_wav: str,
        language: str = "tr",
        output_path: Optional[str] = None,
        speed: float = 1.0,
        temperature: float = 0.65, 
        top_p: float = 0.85,
        top_k: int = 50,
        repetition_penalty: float = 2.0,
        noise_level: float = 0.0,
        noise_type: str = "artificial",
        noise_file_path: Optional[str] = None
    ) -> str:
        """
    	Metni klonlanmış sesle sentezler
    	Args:
        text: Sentezlenecek metin
        speaker_wav: Referans ses dosyası (voice cloning için)
        language: Dil kodu (varsayılan: 'tr' - Türkçe)
        output_path: Çıkış dosyası yolu (opsiyonel)       
    	Returns:
        str: Oluşturulan ses dosyasının yolu
    """
        if not Path(speaker_wav).exists():
            raise FileNotFoundError(f"Referans ses dosyası bulunamadı: {speaker_wav}")
        
        if not text or len(text.strip()) == 0:
            raise ValueError("Sentezlenecek metin boş olamaz")
        
        if output_path is None:
            output_path = "output_tts.wav"

        # 1. Metin Ön İşleme
        clean_text = TTSPreprocessor.clean_text(text)
        
        if len(clean_text) > 250:
            segments = TTSPreprocessor.split_long_text(clean_text, max_length=250)
            logger.info(f"Uzun metin {len(segments)} parçaya bölündü.")
        else:
            segments = [clean_text]

        combined_audio = []
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                for idx, segment in enumerate(segments):
                    temp_file = os.path.join(temp_dir, f"part_{idx}.wav")
                    logger.info(f"Parça {idx+1}/{len(segments)} sentezleniyor...")
                    
                    self.tts.tts_to_file(
                        text=segment,
                        speaker_wav=speaker_wav,
                        language=language,
                        file_path=temp_file,
                        split_sentences=False,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty
                    )
                    
                    y, sr = AudioUtils.load_audio(temp_file)
                    combined_audio.append(y)
                    # Cümleler arası hafif sessizlik (0.3sn)
                    silence = np.zeros(int(sr * 0.3))
                    combined_audio.append(silence)

            # 2. Birleştir
            final_audio = np.concatenate(combined_audio)

            # 3. Hız Ayarı
            if speed != 1.0:
                logger.info(f"Hız ayarlanıyor: x{speed}")
                final_audio = AudioUtils.change_audio_speed(final_audio, speed)

            # 4. Gürültü Ekleme
            if noise_level > 0:
                if noise_type == "real" and noise_file_path and os.path.exists(noise_file_path):
                    logger.info(f"Gerçek oda gürültüsü ekleniyor: {noise_file_path}")
                    final_audio = AudioUtils.add_real_background_noise(final_audio, noise_file_path, noise_level)
                else:
                    logger.info("Yapay (White) gürültü ekleniyor.")
                    final_audio = AudioUtils.add_artificial_noise(final_audio, noise_level)

            # 5. Kaydet
            AudioUtils.save_audio(final_audio, output_path, sr)
            return output_path
            
        except Exception as e:
            logger.error(f"Ses sentezleme hatası: {e}")
            raise

