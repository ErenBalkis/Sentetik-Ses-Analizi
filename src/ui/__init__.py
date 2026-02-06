"""
UI modules package
Clean separation of Streamlit UI components
"""

from .tab_stt import render_stt_tab
from .tab_tts import render_tts_tab
from .tab_test import render_test_tab
from .tab_comparison import render_comparison_tab

__all__ = [
    'render_stt_tab',
    'render_tts_tab', 
    'render_test_tab',
    'render_comparison_tab'
]
