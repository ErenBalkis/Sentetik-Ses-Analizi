"""
Speech-to-Text Tab UI Module
Handles STT functionality with audio recording support
"""

import streamlit as st
import os
import io
from typing import Optional
from src.stt_module import WhisperSTT
from src.utils import AudioUtils, TempFileManager, format_duration


def render_stt_tab():
    """
    Renders the Speech-to-Text tab
    Supports both file upload and browser-based audio recording
    """
    st.markdown('<div class="sub-header">🎤 Ses Tanıma (Speech-to-Text)</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">📝 Ses dosyası yükleyin veya tarayıcıdan kaydedin ve Whisper ile metne dönüştürün.</div>', unsafe_allow_html=True)
    
    # İki seçenek: Dosya yükleme veya Kayıt
    tab_upload, tab_record = st.tabs(["📁 Dosya Yükle", "🎙️ Ses Kaydet"])
    
    # Tab 1: File Upload
    with tab_upload:
        uploaded_audio = st.file_uploader(
            "Ses dosyası yükleyin (WAV, MP3, OGG)",
            type=['wav', 'mp3', 'ogg'],
            key="stt_audio_upload"
        )
        
        if uploaded_audio is not None:
            _process_audio_upload(uploaded_audio)
    
    # Tab 2: Audio Recording
    with tab_record:
        _render_audio_recording()
    
    # Sonuçları göster
    if st.session_state.get('transcribed_text', ''):
        st.markdown('<div class="success-box">✅ Ses başarıyla metne dönüştürüldü!</div>', unsafe_allow_html=True)
        st.text_area(
            "Tanınan Metin:",
            st.session_state.transcribed_text,
            height=150,
            key="transcribed_text_display"
        )


def _process_audio_upload(uploaded_audio) -> None:
    """
    Processes uploaded audio file
    
    Args:
        uploaded_audio: Streamlit UploadedFile object
    """
    # Dosyayı geçici olarak işle - disk'e kaydetme
    audio_bytes = uploaded_audio.getvalue()
    
    # Ses bilgilerini göster
    col1, col2 = st.columns(2)
    
    with col1:
        st.audio(audio_bytes, format=f"audio/{uploaded_audio.type.split('/')[-1]}")
    
    with col2:
        # Duration hesaplamak için geçici dosya gerekiyor
        with TempFileManager.create_temp_audio_file(suffix=f".{uploaded_audio.name.split('.')[-1]}") as temp_path:
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            duration = AudioUtils.get_audio_duration(temp_path)
            st.info(f"⏱️ Süre: {format_duration(duration)}")
    
    # Transkripsiyon
    if st.button("🎯 Metne Dönüştür", key="transcribe_upload_btn"):
        _transcribe_audio(audio_bytes, uploaded_audio.name)


def _render_audio_recording() -> None:
    """
    Renders audio recording interface using st.audio_input
    """
    st.markdown("#### 🎙️ Tarayıcıdan Ses Kaydedin")
    st.info("🎤 Mikrofonunuzdan doğrudan ses kaydedin. Dosya otomatik kaydedilmez.")
    
    # Audio input
    recorded_audio = st.audio_input("Kayıt Başlat", key="stt_audio_recorder")
    
    if recorded_audio is not None:
        # Ham veriyi al
        raw_audio_bytes = recorded_audio.getvalue()
        
        # --- DÜZELTME BURADA ---
        # Ham veriyi doğrudan kullanmak yerine önce standart WAV'a çeviriyoruz.
        # Bu işlem 'Tiz Ses' sorununu ve Whisper'ın format hatasını çözer.
        with st.spinner("Ses işleniyor..."):
            processed_audio_bytes = AudioUtils.bytes_to_wav_bytes(raw_audio_bytes)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # İşlenmiş sesi oynat
            st.audio(processed_audio_bytes, format="audio/wav")
        
        with col2:
            # İndirme butonuna işlenmiş sesi ver
            st.download_button(
                label="⬇️ Kaydı İndir",
                data=processed_audio_bytes,
                file_name="kayit_22khz.wav",
                mime="audio/wav",
                key="download_recorded_audio"
            )
        
        # Transkripsiyon butonuna da işlenmiş sesi gönder
        if st.button("🎯 Metne Dönüştür", key="transcribe_record_btn"):
            _transcribe_audio(processed_audio_bytes, "kayit.wav")


def _transcribe_audio(audio_bytes: bytes, filename: str) -> None:
    """
    Transcribes audio from bytes data
    
    Args:
        audio_bytes: Audio data in bytes
        filename: Original filename for context
    """
    # Model yükle
    if st.session_state.stt_model is None:
        with st.spinner('🎤 Whisper modeli yükleniyor...'):
            st.session_state.stt_model = load_stt_model()
    
    # Geçici dosya oluştur, transkribe et ve temizle
    with st.spinner("🔍 Ses analiz ediliyor..."):
        with TempFileManager.create_temp_audio_file(suffix=f".{filename.split('.')[-1]}") as temp_path:
            with open(temp_path, 'wb') as f:
                f.write(audio_bytes)
            
            result = st.session_state.stt_model.transcribe(temp_path, language="tr")
            st.session_state.transcribed_text = result['text']
    
    st.success("✅ Transkripsiyon tamamlandı!")
    st.rerun()


@st.cache_resource
def load_stt_model() -> WhisperSTT:
    """
    Loads and caches the Whisper STT model
    
    Returns:
        WhisperSTT: Loaded model instance
    """
    return WhisperSTT(model_size="base")
