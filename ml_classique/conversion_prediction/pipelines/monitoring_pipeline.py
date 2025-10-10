import argparse
import sys
import os
import pickle
import pandas as pd
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring.drift_detector import MonitoringPipeline
from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.utils.config_manager import ConfigManager

def run_monitoring():
    """Pipeline de monitoring quotidien"""
    print("🔍 Lancement du pipeline de monitoring...")
    
    # Chargement du modèle et des données de référence
    model_path = "models/conversion_model.pkl"
    reference_data_path = "data/processed/train.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(reference_data_path):
        print("❌ Modèle ou données de référence non trouvés")
        return
    
    # Chargement du modèle
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Chargement des données de référence
    reference_data = pd.read_csv(reference_data_path)
    
    # Chargement des métriques de référence
    try:
        with open('models/metrics.json', 'r') as f:
            import json
            metrics_data = json.load(f)
            reference_metrics = metrics_data.get('models', {}).get('best_model', {})
    except:
        print("⚠️  Métriques de référence non trouvées, utilisation de valeurs par défaut")
        reference_metrics = {'auc': 0.8, 'accuracy': 0.75, 'f1_score': 0.7}
    
    # Chargement des données courantes (dernières 24h par exemple)
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    
    # Filtrer les données récentes pour le monitoring
    if 'sessions' in raw_data and 'session_start' in raw_data['sessions'].columns:
        raw_data['sessions']['session_start'] = pd.to_datetime(raw_data['sessions']['session_start'])
        recent_cutoff = datetime.now() - timedelta(days=1)
        raw_data['sessions'] = raw_data['sessions'][
            raw_data['sessions']['session_start'] >= recent_cutoff
        ]
    
    # Feature engineering pour les données courantes
    engineer = FeatureEngineer()
    current_features = engineer.create_features(raw_data)
    
    # Préparation des données de test
    target = ConfigManager().get('model.target')
    if target in current_features.columns:
        X_current = current_features.drop(columns=[target, 'session_id', 'user_id'], errors='ignore')
        y_current = current_features[target]
    else:
        print("❌ Variable cible non trouvée dans les données courantes")
        return
    
    # Initialisation du pipeline de monitoring
    monitor = MonitoringPipeline(reference_data, model, reference_metrics)
    
    # Exécution du monitoring
    report = monitor.generate_monitoring_dashboard(
        current_features, 
        X_current, 
        y_current,
        save_dir=f"monitoring_reports/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Alertes critiques
    if report['alerts']['critical_alert']:
        print("🚨 ALERTE CRITIQUE: Drift détecté à la fois dans les données et la performance!")
        # Ici vous pourriez ajouter l'envoi d'email/notification
    
    elif report['alerts']['data_drift_alert']:
        print("⚠️  Alerte: Drift détecté dans les données")
    
    elif report['alerts']['performance_drift_alert']:
        print("⚠️  Alerte: Drift détecté dans la performance du modèle")
    
    else:
        print("✅ Aucun drift critique détecté")
    
    print("✅ Pipeline de monitoring terminé")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', default='daily', choices=['daily', 'weekly', 'monthly'])
    args = parser.parse_args()
    
    run_monitoring()