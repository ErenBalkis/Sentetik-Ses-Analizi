"""
Yardımcı fonksiyonlar modülü
Ses dosyası işleme, validasyon ve metin ön işleme
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple
import soundfile as sf
from pydub import AudioSegment
import numpy as np
import librosa
import tempfile
import io
from contextlib import contextmanager

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AudioUtils:
    """Ses dosyası işleme yardımcı sınıfı"""
    
    SUPPORTED_FORMATS = ['.wav', '.mp3', '.ogg', '.flac', '.m4a']
    SAMPLE_RATE = 22050
    
    @staticmethod
    def validate_audio_file(file_path: str) -> bool:
        """Ses dosyasının geçerliliğini kontrol eder"""
        if not os.path.exists(file_path):
            logger.error(f"Dosya bulunamadı: {file_path}")
            return False
        
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in AudioUtils.SUPPORTED_FORMATS:
            logger.error(f"Desteklenmeyen format: {file_ext}")
            return False
        
        return True
    
    @staticmethod
    def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
        """Ses dosyasını WAV formatına dönüştürür"""
        if not AudioUtils.validate_audio_file(input_path):
            raise ValueError(f"Geçersiz ses dosyası: {input_path}")
        
        if output_path is None:
            output_path = Path(input_path).with_suffix('.wav')
        
        try:
            audio = AudioSegment.from_file(input_path)
            audio = audio.set_frame_rate(AudioUtils.SAMPLE_RATE)
            audio = audio.set_channels(1)  # Mono
            audio.export(output_path, format='wav')
            return str(output_path)
        except Exception as e:
            logger.error(f"Dönüştürme hatası: {e}")
            raise
    
    @staticmethod
    def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
        """Ses dosyasını yükler"""
        try:
            # Mono olarak yüklemesi önemli
            audio_data, sample_rate = librosa.load(file_path, sr=sr, mono=True)
            return audio_data, sample_rate
        except Exception as e:
            logger.error(f"Ses yükleme hatası: {e}")
            raise
    
    @staticmethod
    def save_audio(audio_data: np.ndarray, output_path: str, sr: int = SAMPLE_RATE):
        """Ses verisini dosyaya kaydeder"""
        try:
            sf.write(output_path, audio_data, sr)
            logger.info(f"Ses kaydedildi: {output_path}")
        except Exception as e:
            logger.error(f"Ses kaydetme hatası: {e}")
            raise

    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """Ses dosyasının süresini saniye cinsinden döndürür"""
        try:
            duration = librosa.get_duration(path=file_path)
            return duration
        except Exception as e:
            logger.error(f"Süre hesaplama hatası: {e}")
            return 0.0

    @staticmethod
    def change_audio_speed(audio_data: np.ndarray, speed_rate: float) -> np.ndarray:
        """Ses hızını değiştirir (Time Stretching)"""
        if speed_rate == 1.0:
            return audio_data
        
        try:
            y_stretched = librosa.effects.time_stretch(audio_data, rate=speed_rate)
            return y_stretched
        except Exception as e:
            logger.error(f"Hız değiştirme hatası: {e}")
            return audio_data

    @staticmethod
    def bytes_to_wav_bytes(audio_bytes: bytes, target_sr: int = SAMPLE_RATE) -> bytes:
        """
        Bellekteki ses verisini (herhangi bir format) standart WAV formatına ve 
        projenin sample rate değerine (22050Hz) dönüştürür.
        """
        try:
            # BytesIO ile veriyi okuyoruz
            input_io = io.BytesIO(audio_bytes)
            
            # Pydub ile yüklüyoruz (Pydub formatı otomatik tanır - WebM, MP3, WAV vs.)
            audio = AudioSegment.from_file(input_io)
            
            # Sample rate'i 22050Hz'e sabitliyoruz ve Mono yapıyoruz (STT ve TTS için en iyisi)
            audio = audio.set_frame_rate(target_sr).set_channels(1)
            
            # Tekrar bytes olarak dışarı aktarıyoruz
            output_io = io.BytesIO()
            audio.export(output_io, format="wav")
            output_io.seek(0)
            
            return output_io.read()
            
        except Exception as e:
            logger.error(f"Ses dönüştürme hatası: {e}")
            # Dönüştüremezse orijinali döndür (hata patlamasın diye)
            return audio_bytes

    @staticmethod
    def add_artificial_noise(audio_data: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Sese yapay Gaussian (Beyaz) gürültü ekler."""
        if noise_level <= 0:
            return audio_data
            
        try:
            noise = np.random.normal(0, noise_level, audio_data.shape)
            noisy_audio = audio_data + noise
            
            # Clipping önleme
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val
                
            return noisy_audio.astype(np.float32)
        except Exception as e:
            logger.error(f"Yapay gürültü ekleme hatası: {e}")
            return audio_data

    @staticmethod
    def add_real_background_noise(audio_data: np.ndarray, noise_path: str, noise_level: float = 0.01) -> np.ndarray:
        """
        Sese gerçek bir ortam kaydı ekler.
        Eğer gürültü kısa ise döngüye sokar (loop), uzun ise keser (crop).
        """
        if noise_level <= 0 or not os.path.exists(noise_path):
            return audio_data

        try:
            # Gürültü dosyasını aynı sample rate ile yükle
            noise_data, _ = AudioUtils.load_audio(noise_path, sr=AudioUtils.SAMPLE_RATE)
            
            target_len = len(audio_data)
            noise_len = len(noise_data)

            # --- Sinyal İşleme: Süre Eşitleme ---
            if noise_len < target_len:
                # Gürültü kısa ise: Döngü (Tile) yap
                tile_count = int(np.ceil(target_len / noise_len))
                noise_data = np.tile(noise_data, tile_count)
                noise_data = noise_data[:target_len]
            else:
                # Gürültü uzun ise: Kes (Crop)
                noise_data = noise_data[:target_len]

            # --- Normalizasyon ve Karıştırma ---
            # Gürültünün kendi ses seviyesini normalize et
            noise_max = np.max(np.abs(noise_data))
            if noise_max > 0:
                noise_data = noise_data / noise_max
            
            # Ana sese ekle
            noisy_audio = audio_data + (noise_data * noise_level)

            # Çıkış clipping kontrolü
            max_val = np.max(np.abs(noisy_audio))
            if max_val > 1.0:
                noisy_audio = noisy_audio / max_val

            return noisy_audio.astype(np.float32)

        except Exception as e:
            logger.error(f"Gerçek gürültü ekleme hatası: {e}")
            return audio_data


class TTSPreprocessor:
    """TTS için metin ön işleme sınıfı"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        text = ' '.join(text.split())
        return text.strip()
    
    @staticmethod
    def split_long_text(text: str, max_length: int = 250) -> list:
        text = text.replace('!', '!|').replace('?', '?|').replace('.', '.|')
        raw_sentences = text.split('|')
        
        segments = []
        current_segment = ""
        
        for sentence in raw_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_segment) + len(sentence) + 1 <= max_length:
                current_segment += sentence + " "
            else:
                if current_segment:
                    segments.append(current_segment.strip())
                current_segment = sentence + " "
        
        if current_segment:
            segments.append(current_segment.strip())
        return segments


class PathManager:
    """Proje dosya yollarını yöneten sınıf"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.reference_voices_dir = self.data_dir / "reference_voices"
        self.test_audio_dir = self.data_dir / "test_audio"
        self.training_data_dir = self.data_dir / "training_data"
    
    def create_directories(self):
        dirs = [
            self.data_dir, 
            self.models_dir, 
            self.reference_voices_dir, 
            self.test_audio_dir, 
            self.training_data_dir
        ]
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        return self.models_dir / model_name


class TempFileManager:
    """
    Geçici dosya yönetimi için context manager
    Disk temizliğini garanti eder - Clean Code prensibi
    """
    
    @staticmethod
    @contextmanager
    def create_temp_audio_file(suffix: str = '.wav'):
        """
        Geçici ses dosyası oluşturur ve otomatik temizler
        
        Args:
            suffix: Dosya uzantısı
            
        Yields:
            str: Geçici dosya yolu
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_path = temp_file.name
        temp_file.close()
        
        try:
            yield temp_path
        finally:
            # Dosyayı temizle
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Geçici dosya silinemedi: {temp_path}, {e}")
    
    @staticmethod
    def bytes_to_temp_file(audio_bytes: bytes, suffix: str = '.wav') -> str:
        """
        Bytes verisini geçici dosyaya yazar
        
        Args:
            audio_bytes: Ses verisi
            suffix: Dosya uzantısı
            
        Returns:
            str: Geçici dosya yolu (manuel temizlik gerektirir)
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(audio_bytes)
        temp_path = temp_file.name
        temp_file.close()
        return temp_path
    
    @staticmethod
    def audio_to_bytes(audio_data: np.ndarray, sr: int = 22050) -> bytes:
        """
        Numpy array'i bytes'a dönüştürür
        
        Args:
            audio_data: Ses verisi
            sr: Sample rate
            
        Returns:
            bytes: WAV formatında bytes
        """
        byte_io = io.BytesIO()
        sf.write(byte_io, audio_data, sr, format='WAV')
        byte_io.seek(0)
        return byte_io.read()


def format_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

