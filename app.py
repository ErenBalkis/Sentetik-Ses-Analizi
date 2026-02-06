"""
Ses Sentezleme ve Sahte Ses Oluşturma Sistemi
Ana Streamlit Uygulaması
"""

import streamlit as st
import os
import sys
from pathlib import Path
import numpy as np
import logging

# Modül yolu ayarı
sys.path.insert(0, os.path.dirname(__file__))

# UI modüllerini import et
from src.ui import (
    render_stt_tab,
    render_tts_tab,
    render_test_tab,
    render_comparison_tab
)

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sayfa yapılandırması
st.set_page_config(
    page_title="Ses Tespiti Sistemi",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #1f77b4;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid white;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #28a745;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: black;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state() -> None:
    """
    Initializes Streamlit session state variables
    Ensures all required state is available
    """
    # Model states
    if 'stt_model' not in st.session_state:
        st.session_state.stt_model = None
    if 'tts_model' not in st.session_state:
        st.session_state.tts_model = None
    if 'ml_model' not in st.session_state:
        st.session_state.ml_model = None
    
    # Data states
    if 'transcribed_text' not in st.session_state:
        st.session_state.transcribed_text = ""
    if 'synthesized_audio_bytes' not in st.session_state:
        st.session_state.synthesized_audio_bytes = None
    
    # Test states
    if 'test_audio_path' not in st.session_state:
        st.session_state.test_audio_path = None
    if 'test_audio_bytes' not in st.session_state:
        st.session_state.test_audio_bytes = None
    if 'test_audio_label' not in st.session_state:
        st.session_state.test_audio_label = None
    
    # Prediction tracking
    if 'user_predictions' not in st.session_state:
        st.session_state.user_predictions = []
    if 'ml_predictions' not in st.session_state:
        st.session_state.ml_predictions = []
    if 'correct_labels' not in st.session_state:
        st.session_state.correct_labels = []


def render_sidebar() -> None:
    """
    Renders the application sidebar with controls and statistics
    """
    with st.sidebar:
        st.markdown("### ⚙️ Ayarlar")
        
        st.markdown("#### Model Yükleme")
        st.info("💡 Modeller ihtiyaç duyulduğunda otomatik yüklenir")
        
        st.markdown("---")
        
        # İstatistikler
        st.markdown("### 📊 İstatistikler")
        total_tests = len(st.session_state.user_predictions)
        st.metric("Toplam Test", total_tests)
        
        if total_tests > 0:
            user_acc = np.mean([
                1 if u == c else 0 
                for u, c in zip(st.session_state.user_predictions, st.session_state.correct_labels)
            ]) * 100
            ml_acc = np.mean([
                1 if m == c else 0 
                for m, c in zip(st.session_state.ml_predictions, st.session_state.correct_labels)
            ]) * 100
            
            st.metric("İnsan Doğruluğu", f"{user_acc:.1f}%")
            st.metric("ML Doğruluğu", f"{ml_acc:.1f}%")
        
        
        # Reset button
        if st.button("🔄 Uygulamayı Sıfırla"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def main():
    """
    Main application entry point
    Orchestrates the entire UI flow
    """
    # Initialize
    init_session_state()
    
    # Header
    st.markdown(
        '<div class="main-header">🎤 Ses Sentezleme ve Sahte Ses Oluşturma Sistemi</div>', 
        unsafe_allow_html=True
    )
    st.markdown("---")
    
    # Sidebar
    render_sidebar()
    
    # Main content - Tabs
    tabs = st.tabs([
        "1️⃣ Ses Tanıma (STT)",
        "2️⃣ Ses Sentezleme (TTS)", 
        "3️⃣ İnteraktif Test",
        "4️⃣ Karşılaştırma"
    ])
    
    # Render each tab
    with tabs[0]:
        render_stt_tab()
    
    with tabs[1]:
        render_tts_tab()
    
    with tabs[2]:
        render_test_tab()
    
    with tabs[3]:
        render_comparison_tab()


if __name__ == "__main__":
    main()
