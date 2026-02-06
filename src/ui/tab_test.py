"""
Interactive Test Tab UI Module
Handles interactive deepfake detection testing with audio recording
"""

import streamlit as st
import numpy as np
from pathlib import Path
import os
from typing import Optional
from src.ml_detector import DeepfakeDetector
from src.utils import TempFileManager


def render_test_tab():
    """
    Renders the Interactive Test tab
    Users can test their ability to detect deepfakes vs ML model
    """
    st.markdown('<div class="sub-header">🎮 İnteraktif Test: Yapay mı? Gerçek mi?</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">🎧 Bir ses dinleyin ve yapay mı gerçek mi tahmin edin!</div>', unsafe_allow_html=True)
    
    # Test sesi hazırlama seçenekleri
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎲 Rastgele Test Sesi Oluştur", key="generate_test"):
            _generate_random_test_audio()
    
    with col2:
        # Kendi sesini kaydedip test et
        if st.button("🎙️ Kendi Sesimi Test Et", key="test_own_voice"):
            st.session_state.test_mode = "record"
            st.rerun()
    
    # Kendi ses kaydı modu
    if st.session_state.get('test_mode') == 'record':
        _render_own_voice_test()
    
    # Test sesi oynat
    if st.session_state.get('test_audio_bytes') or st.session_state.get('test_audio_path'):
        st.markdown("#### 🎧 Test Sesi")
        
        # Ses göster
        if st.session_state.get('test_audio_bytes'):
            st.audio(st.session_state.test_audio_bytes, format="audio/wav")
        elif st.session_state.test_audio_path:
            st.audio(st.session_state.test_audio_path)
        
        st.markdown("### 🤔 Bu ses yapay mı, gerçek mi?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🤖 YAPAY SES", key="predict_fake", use_container_width=True):
                _make_prediction(user_prediction=1)
        
        with col2:
            if st.button("👤 GERÇEK SES", key="predict_real", use_container_width=True):
                _make_prediction(user_prediction=0)
    else:
        st.info("👆 Önce bir test sesi oluşturun veya kendi sesinizi kaydedin.")


def _render_own_voice_test() -> None:
    """
    Renders interface for testing own recorded voice
    """
    st.markdown("#### 🎙️ Kendi Sesinizi Kaydedin")
    st.info("Sesinizi kaydettikten sonra yapay mı gerçek mi tahmin edin!")
    
    recorded_audio = st.audio_input("Kayıt Başlat", key="test_audio_recorder")
    
    if recorded_audio:
        audio_bytes = recorded_audio.getvalue()
        st.session_state.test_audio_bytes = audio_bytes
        st.session_state.test_audio_label = 0  # Gerçek ses (kullanıcı kaydetti)
        st.session_state.test_mode = None
        
        st.success("✅ Ses kaydedildi! Şimdi tahmin yapın.")
        st.rerun()


def _generate_random_test_audio() -> None:
    """
    Generates a random test audio (real or synthetic)
    """
    is_real = np.random.choice([True, False])
    
    if is_real:
        # Gerçek ses kullan
        real_dir = Path("data/training_data/real")
        if real_dir.exists():
            real_files = list(real_dir.glob("*.wav"))
            if real_files:
                selected_file = str(np.random.choice(real_files))
                st.session_state.test_audio_path = selected_file
                st.session_state.test_audio_bytes = None
                st.session_state.test_audio_label = 0  # Gerçek
                st.rerun()
                return
        st.warning("Gerçek ses dosyası bulunamadı!")
    else:
        # Yapay ses kullan - önce synthesized olanı dene
        if st.session_state.get('synthesized_audio_bytes'):
            st.session_state.test_audio_bytes = st.session_state.synthesized_audio_bytes
            st.session_state.test_audio_path = None
            st.session_state.test_audio_label = 1  # Yapay
            st.rerun()
            return
        
        # Yoksa training data'dan yapay ses al
        synthetic_dir = Path("data/training_data/synthetic")
        if synthetic_dir.exists():
            synthetic_files = list(synthetic_dir.glob("*.wav"))
            if synthetic_files:
                selected_file = str(np.random.choice(synthetic_files))
                st.session_state.test_audio_path = selected_file
                st.session_state.test_audio_bytes = None
                st.session_state.test_audio_label = 1  # Yapay
                st.rerun()
                return
        st.warning("Yapay ses dosyası bulunamadı!")


def _make_prediction(user_prediction: int) -> None:
    """
    Processes user prediction and compares with ML model
    
    Args:
        user_prediction: User's prediction (0=Real, 1=Fake)
    """
    if not _has_test_audio():
        st.error("❌ Test sesi hazır değil!")
        return
    
    # ML modelini yükle
    if st.session_state.ml_model is None:
        with st.spinner('🤖 ML modeli yükleniyor...'):
            st.session_state.ml_model = load_ml_model()
    
    if not st.session_state.ml_model:
        st.error("❌ ML modeli yüklenemedi!")
        return
    
    # Test sesini geçici dosyaya kaydet (eğer bytes ise)
    temp_path_to_clean = None
    audio_path_for_ml = None
    
    try:
        if st.session_state.get('test_audio_bytes'):
            temp_path_to_clean = TempFileManager.bytes_to_temp_file(
                st.session_state.test_audio_bytes,
                suffix='.wav'
            )
            audio_path_for_ml = temp_path_to_clean
        else:
            audio_path_for_ml = st.session_state.test_audio_path
        
        # ML tahmini al
        ml_prediction = st.session_state.ml_model.predict(audio_path_for_ml)
        ml_proba_real, ml_proba_fake = st.session_state.ml_model.predict_proba(audio_path_for_ml)
        
        # Doğru cevap
        correct_label = st.session_state.test_audio_label
        correct_text = "GERÇEK" if correct_label == 0 else "YAPAY"
        
        # Doğruluk kontrolü
        user_correct = (user_prediction == correct_label)
        ml_correct = (ml_prediction == correct_label)
        
        # Sonuçları kaydet
        st.session_state.user_predictions.append(user_prediction)
        st.session_state.ml_predictions.append(ml_prediction)
        st.session_state.correct_labels.append(correct_label)
        
        # Sonuçları göster
        _display_prediction_results(
            user_prediction, ml_prediction,
            user_correct, ml_correct,
            correct_text,
            ml_proba_real, ml_proba_fake
        )
        
        # Test sesini sıfırla
        st.session_state.test_audio_path = None
        st.session_state.test_audio_bytes = None
        st.session_state.test_audio_label = None
        
    finally:
        # Geçici dosyayı temizle
        if temp_path_to_clean and os.path.exists(temp_path_to_clean):
            try:
                os.remove(temp_path_to_clean)
            except:
                pass


def _has_test_audio() -> bool:
    """Checks if test audio is available"""
    return (st.session_state.get('test_audio_path') is not None or 
            st.session_state.get('test_audio_bytes') is not None) and \
           st.session_state.get('test_audio_label') is not None


def _display_prediction_results(
    user_prediction: int,
    ml_prediction: int,
    user_correct: bool,
    ml_correct: bool,
    correct_text: str,
    ml_proba_real: float,
    ml_proba_fake: float
) -> None:
    """
    Displays prediction results comparison
    
    Args:
        user_prediction: User's prediction
        ml_prediction: ML model's prediction
        user_correct: Whether user was correct
        ml_correct: Whether ML was correct
        correct_text: Correct answer text
        ml_proba_real: ML probability for real
        ml_proba_fake: ML probability for fake
    """
    st.markdown("---")
    st.markdown("### 📊 Tahmin Sonuçları")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 👤 Sizin Tahminiz")
        user_text = "YAPAY" if user_prediction == 1 else "GERÇEK"
        if user_correct:
            st.markdown(f'<div class="success-box">✅ <strong>{user_text}</strong><br>Doğru tahmin!</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ <strong>{user_text}</strong><br>Yanlış tahmin!</div>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🤖 ML Tahmini")
        ml_text = "YAPAY" if ml_prediction == 1 else "GERÇEK"
        if ml_correct:
            st.markdown(f'<div class="success-box">✅ <strong>{ml_text}</strong><br>Doğru tahmin!</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="error-box">❌ <strong>{ml_text}</strong><br>Yanlış tahmin!</div>', 
                       unsafe_allow_html=True)
    
    st.markdown(f'<div class="info-box">🎯 <strong>Doğru Cevap: {correct_text}</strong></div>', 
               unsafe_allow_html=True)
    
    # ML güven skorları
    st.markdown("#### 📊 ML Model Güven Skorları")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Gerçek Olasılığı", f"{ml_proba_real*100:.1f}%")
    with col2:
        st.metric("Yapay Olasılığı", f"{ml_proba_fake*100:.1f}%")
    
    st.success("✅ Tahmin kaydedildi! Yeni bir test için butonlara tıklayın.")


@st.cache_resource
def load_ml_model() -> Optional[DeepfakeDetector]:
    """
    Loads and caches the ML detector model
    
    Returns:
        DeepfakeDetector: Loaded model or None
    """
    detector = DeepfakeDetector(model_type="lightgbm")
    model_path = "models/deepfake_detector.pkl"
    
    if os.path.exists(model_path):
        detector.load_model(model_path)
        return detector
    else:
        st.warning("⚠️ ML model dosyası bulunamadı!")
        return None
