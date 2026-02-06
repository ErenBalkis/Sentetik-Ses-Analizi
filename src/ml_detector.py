"""
Makine Öğrenmesi Deepfake Detector modülü
Ses özellik çıkarımı ve yapay ses tespiti
"""

import numpy as np
import librosa
import logging
from typing import Dict, Tuple, Optional
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

logger = logging.getLogger(__name__)


class AudioFeatureExtractor:
    """
    Ses dosyalarından özellik çıkarımı yapan sınıf
    MFCC, Mel Spectrogram, Chroma, Spectral Contrast, ZCR, RMS özellikleri
    """
    
    def __init__(self, sr: int = 16000, n_mfcc: int = 40):
        """
        Özellik çıkarıcıyı başlatır
        
        Args:
            sr: Örnekleme hızı
            n_mfcc: MFCC katsayı sayısı
        """
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def extract_features(self, audio_path: str) -> np.ndarray:
        """
        Ses dosyasından özellikleri çıkarır
        
        Args:
            audio_path: Ses dosyası yolu
            
        Returns:
            np.ndarray: Özellik vektörü
        """
        try:
            # Ses dosyasını yükle
            y, sr = librosa.load(audio_path, sr=self.sr, duration=20)  # Maksimum 20 saniye
            
            # Özellik listesi
            features = []
            
            # 1. MFCC (Mel-Frequency Cepstral Coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_max = np.max(mfcc, axis=1)
            features.extend(mfcc_mean)
            features.extend(mfcc_std)
            features.extend(mfcc_max)
            
            # 2. Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_mean = np.mean(mel_spec_db, axis=1)
            mel_std = np.std(mel_spec_db, axis=1)
            mel_max = np.max(mel_spec_db, axis=1)
            features.extend(mel_mean)
            features.extend(mel_std)
            features.extend(mel_max)
            
            # 3. Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            chroma_max = np.max(chroma, axis=1)
            features.extend(chroma_mean)
            features.extend(chroma_std)
            features.extend(chroma_max)
            
            # 4. Spectral Contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            contrast_std = np.std(contrast, axis=1)
            contrast_max = np.max(contrast, axis=1)
            features.extend(contrast_mean)
            features.extend(contrast_std)
            features.extend(contrast_max)
            
            # 5. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zcr)
            zcr_std = np.std(zcr)
            zcr_max = np.max(zcr)
            features.extend([zcr_mean, zcr_std, zcr_max])
            
            # 6. RMS Energy
            rms = librosa.feature.rms(y=y)
            rms_mean = np.mean(rms)
            rms_std = np.std(rms)
            rms_max = np.max(rms)
            features.extend([rms_mean, rms_std, rms_max])
            
            # 7. Spectral Centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            sc_mean = np.mean(spectral_centroid)
            sc_std = np.std(spectral_centroid)
            sc_max = np.max(spectral_centroid)
            features.extend([sc_mean, sc_std, sc_max])
            
            # 8. Spectral Rolloff
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            sr_mean = np.mean(spectral_rolloff)
            sr_std = np.std(spectral_rolloff)
            sr_max = np.max(spectral_rolloff)
            features.extend([sr_mean, sr_std, sr_max])
            
            feature_vector = np.array(features)
            logger.debug(f"Özellik vektörü boyutu: {len(feature_vector)}")
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Özellik çıkarma hatası ({audio_path}): {e}")
            raise
    
    def get_feature_count(self) -> int:
        """Toplam özellik sayısını döndürür"""
        # MFCC: n_mfcc * 2 (mean + std)
        # Mel: 128 * 2 (varsayılan)
        # Chroma: 12 * 2
        # Contrast: 7 * 2
        # ZCR: 2
        # RMS: 2
        # Spectral Centroid: 2
        # Spectral Rolloff: 2
        return (self.n_mfcc * 2) + (128 * 2) + (12 * 2) + (7 * 2) + 2 + 2 + 2 + 2


class DeepfakeDetector:
    """
    Makine öğrenmesi tabanlı sahte ses tespit sınıfı
    """
    
    def __init__(self, model_type: str = "lightgbm"):
        """
        Detector'ı başlatır
        
        Args:
            model_type: Model tipi ('lightgbm', 'random_forest', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_extractor = AudioFeatureExtractor()
        
        # Model seçimi
        if model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=31,
                random_state=42,
                verbose=-1
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == "svm":
            self.model = SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
        
        logger.info(f"DeepfakeDetector oluşturuldu: {model_type}")
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        perform_cv: bool = True
    ) -> Dict:
        """
        Modeli eğitir
        
        Args:
            X: Özellik matrisi (n_samples, n_features)
            y: Etiketler (0: Gerçek, 1: Yapay)
            test_size: Test seti oranı
            perform_cv: Cross-validation yap
            
        Returns:
            Dict: Eğitim metrikleri
        """
        logger.info(f"Model eğitimi başlatılıyor: {len(X)} örnek")
        
        # Veriyi normalize et
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Model eğitimi
        logger.info("Model eğitiliyor...")
        self.model.fit(X_train, y_train)
        
        # Tahminler
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['Gerçek', 'Yapay'])
        
        logger.info(f"Test Doğruluğu: {accuracy:.4f}")
        logger.info(f"\n{class_report}")
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'test_size': len(X_test),
            'train_size': len(X_train)
        }
        
        # Cross-validation
        if perform_cv:
            logger.info("Cross-validation yapılıyor...")
            cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
            results['cv_scores'] = cv_scores
            results['cv_mean'] = np.mean(cv_scores)
            results['cv_std'] = np.std(cv_scores)
            logger.info(f"CV Doğruluğu: {results['cv_mean']:.4f} (+/- {results['cv_std']:.4f})")
        
        return results
    
    def predict(self, audio_path: str) -> int:
        """
        Sesin yapay mı gerçek mi olduğunu tahmin eder
        
        Args:
            audio_path: Ses dosyası yolu
            
        Returns:
            int: 0 (Gerçek) veya 1 (Yapay)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # Özellikleri çıkar
        features = self.feature_extractor.extract_features(audio_path)
        features = features.reshape(1, -1)
        
        # Normalize et
        features_scaled = self.scaler.transform(features)
        
        # Tahmin
        prediction = self.model.predict(features_scaled)[0]
        return int(prediction)
    
    def predict_proba(self, audio_path: str) -> Tuple[float, float]:
        """
        Tahmin olasılıklarını döndürür
        
        Args:
            audio_path: Ses dosyası yolu
            
        Returns:
            Tuple[float, float]: (gerçek_olasılığı, yapay_olasılığı)
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        # Özellikleri çıkar
        features = self.feature_extractor.extract_features(audio_path)
        features = features.reshape(1, -1)
        
        # Normalize et
        features_scaled = self.scaler.transform(features)
        
        # Olasılıklar
        probabilities = self.model.predict_proba(features_scaled)[0]
        return float(probabilities[0]), float(probabilities[1])
    
    def save_model(self, model_path: str):
        """
        Modeli ve scaler'ı kaydeder
        
        Args:
            model_path: Model dosyası yolu
        """
        if self.model is None:
            raise ValueError("Kaydedilecek model yok!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_extractor_params': {
                'sr': self.feature_extractor.sr,
                'n_mfcc': self.feature_extractor.n_mfcc
            }
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model kaydedildi: {model_path}")
    
    def load_model(self, model_path: str):
        """
        Modeli ve scaler'ı yükler
        
        Args:
            model_path: Model dosyası yolu
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        
        # Feature extractor parametrelerini güncelle
        params = model_data['feature_extractor_params']
        self.feature_extractor = AudioFeatureExtractor(
            sr=params['sr'],
            n_mfcc=params['n_mfcc']
        )
        
        logger.info(f"Model yüklendi: {model_path}")
    
    def get_feature_names(self) -> list:
        """
        Özellik isimlerini döndürür (XAI için)
        
        Returns:
            list: Özellik isimlerinin listesi
        """
        feature_names = []
        n_mfcc = self.feature_extractor.n_mfcc
        
        # MFCC features (mean, std, max)
        for i in range(n_mfcc):
            feature_names.append(f'mfcc_{i}_mean')
        for i in range(n_mfcc):
            feature_names.append(f'mfcc_{i}_std')
        for i in range(n_mfcc):
            feature_names.append(f'mfcc_{i}_max')
        
        # Mel Spectrogram (128 bands default)
        for i in range(128):
            feature_names.append(f'mel_{i}_mean')
        for i in range(128):
            feature_names.append(f'mel_{i}_std')
        for i in range(128):
            feature_names.append(f'mel_{i}_max')
        
        # Chroma (12 pitch classes)
        for i in range(12):
            feature_names.append(f'chroma_{i}_mean')
        for i in range(12):
            feature_names.append(f'chroma_{i}_std')
        for i in range(12):
            feature_names.append(f'chroma_{i}_max')
        
        # Spectral Contrast (7 bands)
        for i in range(7):
            feature_names.append(f'contrast_{i}_mean')
        for i in range(7):
            feature_names.append(f'contrast_{i}_std')
        for i in range(7):
            feature_names.append(f'contrast_{i}_max')
        
        # Other features
        feature_names.extend(['zcr_mean', 'zcr_std', 'zcr_max'])
        feature_names.extend(['rms_mean', 'rms_std', 'rms_max'])
        feature_names.extend(['spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_max'])
        feature_names.extend(['spectral_rolloff_mean', 'spectral_rolloff_std', 'spectral_rolloff_max'])
        
        return feature_names
    
    def get_feature_importance(self, top_n: int = 15) -> Dict[str, float]:
        """
        Model özellik önemlerini döndürür (XAI - Explainable AI)
        
        Args:
            top_n: En önemli kaç özellik döndürülecek
            
        Returns:
            Dict[str, float]: Özellik adı -> önem skoru mapping
        """
        if self.model is None:
            raise ValueError("Model henüz eğitilmedi veya yüklenmedi!")
        
        # LightGBM ve RandomForest için feature_importances_ var
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = self.get_feature_names()
            
            # Top N özelliği seç
            indices = np.argsort(importances)[::-1][:top_n]
            
            result = {}
            for idx in indices:
                result[feature_names[idx]] = float(importances[idx])
            
            return result
        else:
            logger.warning(f"{self.model_type} modeli feature importance desteklemiyor")
            return {}


if __name__ == "__main__":
    # Test kodu
    extractor = AudioFeatureExtractor()
    detector = DeepfakeDetector(model_type="lightgbm")
    print(f"ML Detector modülü başarıyla yüklendi!")
    print(f"Özellik sayısı: {extractor.get_feature_count()}")

