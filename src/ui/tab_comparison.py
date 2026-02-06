"""
Comparison Tab UI Module
Displays human vs ML performance comparison with XAI feature importance
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Optional


def render_comparison_tab():
    """
    Renders the comparison tab showing human vs ML accuracy
    Includes Explainable AI (XAI) feature importance visualization
    """
    st.markdown('<div class="sub-header">📊 İnsan vs Makine Karşılaştırması</div>', unsafe_allow_html=True)
    
    if len(st.session_state.get('user_predictions', [])) == 0:
        st.info("📝 Henüz test yapılmadı. Lütfen önce 'İnteraktif Test' sekmesinden tahminlerde bulunun.")
        return
    
    # Doğruluk metrikleri
    _display_accuracy_metrics()
    
    # Karşılaştırma grafiği
    _display_comparison_chart()
    
    # Sonuç açıklaması
    _display_result_summary()
    
    # XAI: Feature Importance
    st.markdown("---")
    _display_feature_importance()
    
    # Detaylı istatistikler
    _display_detailed_statistics()


def _display_accuracy_metrics() -> None:
    """Displays accuracy metrics for human and ML"""
    user_preds = st.session_state.user_predictions
    ml_preds = st.session_state.ml_predictions
    correct_labels = st.session_state.correct_labels
    
    total = len(user_preds)
    user_correct = sum([1 if u == c else 0 for u, c in zip(user_preds, correct_labels)])
    ml_correct = sum([1 if m == c else 0 for m, c in zip(ml_preds, correct_labels)])
    
    user_accuracy = (user_correct / total) * 100
    ml_accuracy = (ml_correct / total) * 100
    
    # Store for later use
    st.session_state.current_user_accuracy = user_accuracy
    st.session_state.current_ml_accuracy = ml_accuracy
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Toplam Test", total)
    with col2:
        st.metric("👤 İnsan Doğruluğu", f"{user_accuracy:.1f}%")
    with col3:
        st.metric("🤖 ML Doğruluğu", f"{ml_accuracy:.1f}%")


def _display_comparison_chart() -> None:
    """Displays accuracy comparison bar chart"""
    user_accuracy = st.session_state.current_user_accuracy
    ml_accuracy = st.session_state.current_ml_accuracy
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='İnsan',
        x=['Doğruluk'],
        y=[user_accuracy],
        marker_color='#1f77b4',
        text=[f'{user_accuracy:.1f}%'],
        textposition='auto',
    ))
    
    fig.add_trace(go.Bar(
        name='Makine',
        x=['Doğruluk'],
        y=[ml_accuracy],
        marker_color='#ff7f0e',
        text=[f'{ml_accuracy:.1f}%'],
        textposition='auto',
    ))
    
    fig.update_layout(
        title='İnsan vs Makine Doğruluk Karşılaştırması',
        yaxis_title='Doğruluk (%)',
        yaxis_range=[0, 100],
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def _display_result_summary() -> None:
    """Displays result summary with winner announcement"""
    user_accuracy = st.session_state.current_user_accuracy
    ml_accuracy = st.session_state.current_ml_accuracy
    
    st.markdown("---")
    st.markdown("### 🏆 Sonuç")
    
    if user_accuracy > ml_accuracy:
        difference = user_accuracy - ml_accuracy
        st.markdown(
            f'<div class="success-box">👤 <strong>İnsan daha başarılı!</strong><br>'
            f'İnsan %{user_accuracy:.1f} doğrulukla tahmin ederken, ML modeli %{ml_accuracy:.1f} doğrulukla tahmin etti.<br>'
            f'İnsan, makineyi %{difference:.1f} farkla geçti! 🎉</div>',
            unsafe_allow_html=True
        )
    elif ml_accuracy > user_accuracy:
        difference = ml_accuracy - user_accuracy
        st.markdown(
            f'<div class="warning-box">🤖 <strong>Makine daha başarılı!</strong><br>'
            f'ML modeli %{ml_accuracy:.1f} doğrulukla tahmin ederken, insan %{user_accuracy:.1f} doğrulukla tahmin etti.<br>'
            f'Yapay zeka, insanı %{difference:.1f} farkla geçti! 🚀</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="info-box">🤝 <strong>Berabere!</strong><br>'
            f'Hem insan hem de makine %{user_accuracy:.1f} doğrulukla tahmin etti. Mükemmel bir denge! ⚖️</div>',
            unsafe_allow_html=True
        )


def _display_feature_importance() -> None:
    """
    Displays XAI feature importance visualization
    Shows which features the ML model relies on most
    """
    st.markdown("### 🔍 Açıklanabilir Yapay Zeka (XAI)")
    st.markdown("**Model hangi özellikleri kullanarak karar veriyor?**")
    
    if st.session_state.ml_model is None:
        st.info("ML modeli yüklü değil. XAI gösterimi için model gereklidir.")
        return
    
    try:
        # Feature importance al
        feature_importance = st.session_state.ml_model.get_feature_importance(top_n=15)
        
        if not feature_importance:
            st.warning("Bu model feature importance desteklemiyor.")
            return
        
        # Plotly grafiği oluştur
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Türkçe açıklamalar ekle
        feature_labels = _translate_feature_names(features)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=feature_labels,
            x=importances,
            orientation='h',
            marker=dict(
                color=importances,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Önem")
            ),
            text=[f'{imp:.4f}' for imp in importances],
            textposition='auto',
        ))
        
        fig.update_layout(
            title='En Önemli 15 Özellik',
            xaxis_title='Önem Skoru',
            yaxis_title='Özellik',
            height=500,
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Açıklama
        st.info(
            "📌 **Açıklama:** Bu özellikler modelin sahte ses tespitinde en çok güvendiği faktörlerdir. "
            "Örneğin, MFCC (Mel-Frequency Cepstral Coefficients) özellikleri sesin spektral karakteristiğini, "
            "RMS (Root Mean Square) ses enerjisini, ZCR (Zero Crossing Rate) ise frekans değişimini temsil eder."
        )
        
    except Exception as e:
        st.error(f"XAI görselleştirme hatası: {e}")


def _translate_feature_names(features: list) -> list:
    """
    Translates technical feature names to Turkish descriptions
    
    Args:
        features: List of feature names
        
    Returns:
        list: Translated feature names
    """
    translations = []
    for feat in features:
        if 'mfcc' in feat:
            idx = feat.split('_')[1]
            stat = feat.split('_')[2] if len(feat.split('_')) > 2 else ''
            stat_tr = {'mean': 'Ort', 'std': 'StdSapma', 'max': 'Maks'}.get(stat, stat)
            translations.append(f'MFCC-{idx} ({stat_tr})')
        elif 'mel' in feat:
            translations.append(f'Mel Spektrum {feat}')
        elif 'chroma' in feat:
            translations.append(f'Chroma {feat}')
        elif 'contrast' in feat:
            translations.append(f'Spektral Kontrast {feat}')
        elif 'zcr' in feat:
            translations.append(f'Zero Crossing Rate ({feat.split("_")[1]})')
        elif 'rms' in feat:
            translations.append(f'RMS Enerji ({feat.split("_")[1]})')
        elif 'spectral_centroid' in feat:
            translations.append(f'Spektral Centroid ({feat.split("_")[-1]})')
        elif 'spectral_rolloff' in feat:
            translations.append(f'Spektral Rolloff ({feat.split("_")[-1]})')
        else:
            translations.append(feat)
    
    return translations


def _display_detailed_statistics() -> None:
    """Displays detailed prediction statistics"""
    with st.expander("📈 Detaylı İstatistikler"):
        st.write("#### Tahmin Dağılımı")
        
        user_preds = st.session_state.user_predictions
        ml_preds = st.session_state.ml_predictions
        correct_labels = st.session_state.correct_labels
        
        total = len(user_preds)
        user_correct = sum([1 if u == c else 0 for u, c in zip(user_preds, correct_labels)])
        ml_correct = sum([1 if m == c else 0 for m, c in zip(ml_preds, correct_labels)])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**İnsan Tahminleri:**")
            st.write(f"- Doğru: {user_correct}/{total}")
            st.write(f"- Yanlış: {total - user_correct}/{total}")
        
        with col2:
            st.write("**ML Tahminleri:**")
            st.write(f"- Doğru: {ml_correct}/{total}")
            st.write(f"- Yanlış: {total - ml_correct}/{total}")
