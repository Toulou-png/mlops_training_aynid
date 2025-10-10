import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from datetime import datetime

def load_latest_monitoring_report():
    """Charge le dernier rapport de monitoring"""
    reports_dir = "monitoring_reports"
    if not os.path.exists(reports_dir):
        return None
    
    # Trouver le rapport le plus récent
    report_dirs = [d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))]
    if not report_dirs:
        return None
    
    latest_dir = sorted(report_dirs)[-1]
    report_path = os.path.join(reports_dir, latest_dir, "monitoring_report.json")
    
    with open(report_path, 'r') as f:
        return json.load(f)

def create_monitoring_dashboard():
    """Crée un dashboard Streamlit pour le monitoring"""
    st.set_page_config(page_title="Aynid - Monitoring ML", layout="wide")
    
    st.title("🔍 Aynid - Dashboard de Monitoring ML")
    st.markdown("---")
    
    # Chargement du rapport
    report = load_latest_monitoring_report()
    if not report:
        st.error("Aucun rapport de monitoring trouvé")
        return
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        data_drift = report['alerts']['data_drift_alert']
        st.metric(
            "Drift des Données", 
            "DÉTECTÉ" if data_drift else "OK",
            delta=None,
            delta_color="inverse" if data_drift else "normal"
        )
    
    with col2:
        perf_drift = report['alerts']['performance_drift_alert']
        st.metric(
            "Drift de Performance", 
            "DÉTECTÉ" if perf_drift else "OK",
            delta=None,
            delta_color="inverse" if perf_drift else "normal"
        )
    
    with col3:
        critical_alert = report['alerts']['critical_alert']
        st.metric(
            "Alerte Critique", 
            "ACTIVE" if critical_alert else "INACTIVE",
            delta=None,
            delta_color="inverse" if critical_alert else "normal"
        )
    
    with col4:
        drifted_features = len(report['data_drift']['summary']['drifted_features'])
        total_features = report['data_drift']['summary']['total_features_monitored']
        st.metric(
            "Features avec Drift", 
            f"{drifted_features}/{total_features}",
            delta=None
        )
    
    st.markdown("---")
    
    # Sections détaillées
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Drift des Données")
        
        if report['data_drift']['summary']['drift_detected']:
            st.warning(f"**{len(report['data_drift']['summary']['drifted_features'])}** features avec drift détecté")
            
            # Top features avec drift
            drift_features = []
            for feature, result in report['data_drift'].items():
                if feature != 'summary' and result.get('drift_detected'):
                    drift_features.append({
                        'feature': feature,
                        'p_value': result.get('p_value', 0),
                        'type': 'Numérique' if 'mean_drift_percentage' in result else 'Catégoriel'
                    })
            
            if drift_features:
                drift_df = pd.DataFrame(drift_features).sort_values('p_value')
                st.dataframe(drift_df, use_container_width=True)
        else:
            st.success("Aucun drift détecté dans les données")
    
    with col2:
        st.subheader("🤖 Performance du Modèle")
        
        current_perf = report['performance_drift']['current_performance']
        ref_metrics = report['performance_drift']['detailed_metrics']
        
        # Métriques de performance
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric("AUC", f"{current_perf.get('auc', 0):.3f}")
        
        with metrics_col2:
            st.metric("Accuracy", f"{current_perf.get('accuracy', 0):.3f}")
        
        with metrics_col3:
            st.metric("F1-Score", f"{current_perf.get('f1_score', 0):.3f}")
        
        # Détection de drift de performance
        if report['performance_drift']['summary']['performance_drift_detected']:
            st.error("Dégradation de performance détectée!")
            
            for metric in report['performance_drift']['summary']['drifted_metrics']:
                detail = ref_metrics.get(metric, {})
                st.write(f"**{metric}**: {detail.get('degradation', 0):.1%} de dégradation")
        else:
            st.success("Performance stable")
    
    # Graphiques
    st.markdown("---")
    st.subheader("📈 Visualisations")
    
    # Charger l'image du drift analysis
    reports_dir = "monitoring_reports"
    report_dirs = sorted([d for d in os.listdir(reports_dir) if os.path.isdir(os.path.join(reports_dir, d))])
    if report_dirs:
        latest_dir = report_dirs[-1]
        drift_plot_path = os.path.join(reports_dir, latest_dir, "drift_analysis.png")
        perf_plot_path = os.path.join(reports_dir, latest_dir, "performance_history.png")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(drift_plot_path):
                st.image(drift_plot_path, caption="Analyse du Drift des Features")
        
        with col2:
            if os.path.exists(perf_plot_path):
                st.image(perf_plot_path, caption="Historique des Performances")
    
    # Dernière mise à jour
    st.markdown("---")
    timestamp = report.get('timestamp', '')
    st.caption(f"Dernière mise à jour: {timestamp}")

if __name__ == "__main__":
    create_monitoring_dashboard()