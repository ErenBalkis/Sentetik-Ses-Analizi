"""
Text-to-Speech Tab UI Module
Handles TTS functionality with transient output (no disk clutter)
"""

import streamlit as st
import os
from typing import Optional
from src.tts_module import CoquiTTS
from src.utils import TempFileManager


def render_tts_tab():
    """
    Renders the Text-to-Speech tab
    Synthesis results stored in session state, not permanently on disk
    """
    st.markdown('<div class="sub-header">🗣️ Gelişmiş Ses Sentezleme (TTS)</div>', unsafe_allow_html=True)
    
    # 1. Referans Ses
    st.markdown("#### 1. Referans Ses Yükleyin (Voice Cloning)")
    
    # Tabs for file upload or recording
    ref_tab_upload, ref_tab_record = st.tabs(["📁 Dosya Yükle", "🎙️ Ses Kaydet"])
    
    ref_bytes = None
    
    with ref_tab_upload:
        reference_audio = st.file_uploader("Klonlanacak Ses", type=['wav', 'mp3'], key="ref_audio_upload")
        
        if reference_audio:
            ref_bytes = reference_audio.getvalue()
            st.audio(ref_bytes, format=f"audio/{reference_audio.type.split('/')[-1]}")
    
    with ref_tab_record:
        st.info("🎤 Mikrofonunuzdan referans ses kaydedin (3-10 saniye önerilir)")
        recorded_ref = st.audio_input("Kayıt Başlat", key="ref_audio_recorder")
        
        if recorded_ref:
            ref_bytes = recorded_ref.getvalue()
            st.audio(ref_bytes, format="audio/wav")
            
            st.download_button(
                label="⬇️ Kaydı İndir",
                data=ref_bytes,
                file_name="referans_kayit.wav",
                mime="audio/wav",
                key="download_ref_recording"
            )
    
    # 2. Metin ve Ayarlar
    st.markdown("#### 2. Metin ve Ayarlar")
    col_text, col_settings = st.columns([2, 1])
    
    with col_text:
        default_text = st.session_state.get('transcribed_text', '') or "Merhaba, bu bir test sesidir."
        synthesis_text = st.text_area("Metin:", default_text, height=350)
    
    with col_settings:
        st.markdown("**🎛️ Ses Ayarları**")
        speed = st.slider("Hız", 0.5, 1.5, 0.85, 0.05)
        
        st.markdown("---")
        st.markdown("**🔊 Gürültü Ayarları**")
        
        # Gürültü Tipi
        noise_type_label = st.radio(
            "Gürültü Tipi:",
            ("Yapay (White Noise)", "Oda Gürültüsü (WAV)")
        )
        
        noise_type = "artificial"
        custom_noise_bytes = None
        
        if noise_type_label == "Oda Gürültüsü (WAV)":
            noise_type = "real"
            uploaded_noise = st.file_uploader(
                "Gürültü Dosyası (20sn+ önerilir)",
                type=['wav', 'mp3'],
                key="noise_upload"
            )
            
            if uploaded_noise:
                custom_noise_bytes = uploaded_noise.getvalue()
                st.success("✅ Gürültü yüklendi")
                st.audio(custom_noise_bytes, format="audio/wav")
            else:
                st.info("⚠️ Lütfen gürültü dosyası yükleyin.")
        
        # Gürültü Seviyesi
        noise_level = st.slider("Gürültü Seviyesi", 0.0, 0.2, 0.02, 0.005)
    
    # 3. Sentezle
    if st.button("🎵 Sentezle", use_container_width=True):
        _synthesize_audio(
            ref_bytes=ref_bytes,
            text=synthesis_text,
            speed=speed,
            noise_level=noise_level,
            noise_type=noise_type,
            custom_noise_bytes=custom_noise_bytes
        )
    
    # 4. Sonuç - Session state'ten göster
    if st.session_state.get('synthesized_audio_bytes'):
        st.markdown("#### 3. Sonuç")
        st.audio(st.session_state.synthesized_audio_bytes, format="audio/wav")
        
        st.download_button(
            label="⬇️ İndir",
            data=st.session_state.synthesized_audio_bytes,
            file_name="sentezlenmis_ses.wav",
            mime="audio/wav",
            key="download_synthesized"
        )


def _synthesize_audio(
    ref_bytes: Optional[bytes],
    text: str,
    speed: float,
    noise_level: float,
    noise_type: str,
    custom_noise_bytes: Optional[bytes]
) -> None:
    """
    Synthesizes audio and stores in session state (no disk save)
    
    Args:
        ref_bytes: Reference audio bytes
        text: Text to synthesize
        speed: Speed multiplier
        noise_level: Noise intensity
        noise_type: Type of noise ('artificial' or 'real')
        custom_noise_bytes: Custom noise file bytes
    """
    # Validation
    if not ref_bytes:
        st.error("❌ Referans ses yok!")
        return
    
    if not text:
        st.error("❌ Metin yok!")
        return
    
    if noise_type == "real" and not custom_noise_bytes:
        st.error("❌ Gürültü dosyası seçilmedi!")
        return
    
    # Model yükle
    if st.session_state.tts_model is None:
        with st.spinner('🗣️ TTS modeli yükleniyor...'):
            st.session_state.tts_model = load_tts_model()
    
    # Geçici dosyalar kullanarak sentezle
    with st.spinner("İşleniyor..."):
        # Referans sesi geçici kaydet
        with TempFileManager.create_temp_audio_file(suffix='.wav') as ref_path:
            with open(ref_path, 'wb') as f:
                f.write(ref_bytes)
            
            # Gürültü dosyasını geçici kaydet (eğer varsa)
            noise_path = None
            if noise_type == "real" and custom_noise_bytes:
                noise_path = TempFileManager.bytes_to_temp_file(custom_noise_bytes, suffix='.wav')
            
            try:
                # Output için geçici dosya
                with TempFileManager.create_temp_audio_file(suffix='.wav') as output_path:
                    st.session_state.tts_model.synthesize(
                        text=text,
                        speaker_wav=ref_path,
                        language="tr",
                        output_path=output_path,
                        speed=speed,
                        noise_level=noise_level,
                        noise_type=noise_type,
                        noise_file_path=noise_path
                    )
                    
                    # Sonucu bytes olarak oku ve session state'e kaydet
                    with open(output_path, 'rb') as f:
                        st.session_state.synthesized_audio_bytes = f.read()
            
            finally:
                # Gürültü dosyasını temizle
                if noise_path and os.path.exists(noise_path):
                    try:
                        os.remove(noise_path)
                    except:
                        pass
    
    st.success("Tamamlandı!")
    st.rerun()


@st.cache_resource
def load_tts_model() -> CoquiTTS:
    """
    Loads and caches the TTS model
    
    Returns:
        CoquiTTS: Loaded model instance
    """
    return CoquiTTS()
