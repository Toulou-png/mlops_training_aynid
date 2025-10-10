import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import warnings
import mlflow
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, Any
from ..utils.config_manager import config

class DataDriftDetector:
    def __init__(self, reference_data: pd.DataFrame, current_data: pd.DataFrame):
        self.reference_data = reference_data
        self.current_data = current_data
        self.drift_results = {}
        
    def detect_numerical_drift(self, column: str, threshold: float = 0.05) -> Dict[str, Any]:
        """Détecte le drift pour une variable numérique avec test de Kolmogorov-Smirnov"""
        ref_data = self.reference_data[column].dropna()
        curr_data = self.current_data[column].dropna()
        
        # Test statistique
        statistic, p_value = ks_2samp(ref_data, curr_data)
        
        # Métriques de distribution
        ref_mean = ref_data.mean()
        curr_mean = curr_data.mean()
        ref_std = ref_data.std()
        curr_std = curr_data.std()
        
        # Calcul du drift en pourcentage
        mean_drift_pct = abs((curr_mean - ref_mean) / ref_mean) * 100 if ref_mean != 0 else 0
        std_drift_pct = abs((curr_std - ref_std) / ref_std) * 100 if ref_std != 0 else 0
        
        drift_detected = p_value < threshold
        
        result = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'ks_statistic': statistic,
            'reference_mean': ref_mean,
            'current_mean': curr_mean,
            'reference_std': ref_std,
            'current_std': curr_std,
            'mean_drift_percentage': mean_drift_pct,
            'std_drift_percentage': std_drift_pct,
            'threshold': threshold
        }
        
        return result
    
    def detect_categorical_drift(self, column: str, threshold: float = 0.05) -> Dict[str, Any]:
        """Détecte le drift pour une variable catégorielle avec test du chi-carré"""
        # Préparation des données
        ref_counts = self.reference_data[column].value_counts()
        curr_counts = self.current_data[column].value_counts()
        
        # Création d'un tableau de contingence
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        
        ref_contingency = [ref_counts.get(cat, 0) for cat in all_categories]
        curr_contingency = [curr_counts.get(cat, 0) for cat in all_categories]
        
        contingency_table = [ref_contingency, curr_contingency]
        
        # Test du chi-carré
        try:
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
            drift_detected = p_value < threshold
        except:
            # En cas d'erreur (données insuffisantes)
            chi2, p_value, dof, expected = 0, 1, 0, None
            drift_detected = False
        
        # Calcul de la divergence KL (simplifiée)
        ref_probs = ref_counts / ref_counts.sum()
        curr_probs = curr_counts / curr_counts.sum()
        
        kl_divergence = 0
        for cat in all_categories:
            p = ref_probs.get(cat, 1e-10)
            q = curr_probs.get(cat, 1e-10)
            kl_divergence += p * np.log(p / q) if p > 0 else 0
        
        result = {
            'drift_detected': drift_detected,
            'p_value': p_value,
            'chi2_statistic': chi2,
            'kl_divergence': kl_divergence,
            'reference_distribution': ref_probs.to_dict(),
            'current_distribution': curr_probs.to_dict(),
            'threshold': threshold
        }
        
        return result
    
    def detect_drift_all_features(self, threshold: float = 0.05) -> Dict[str, Any]:
        """Détecte le drift pour toutes les features"""
        numerical_features = config.get('features.numerical', [])
        categorical_features = config.get('features.categorical', [])
        
        all_results = {}
        drifted_features = []
        
        # Drift numérique
        for feature in numerical_features:
            if feature in self.reference_data.columns and feature in self.current_data.columns:
                result = self.detect_numerical_drift(feature, threshold)
                all_results[feature] = result
                if result['drift_detected']:
                    drifted_features.append(feature)
        
        # Drift catégoriel
        for feature in categorical_features:
            if feature in self.reference_data.columns and feature in self.current_data.columns:
                result = self.detect_categorical_drift(feature, threshold)
                all_results[feature] = result
                if result['drift_detected']:
                    drifted_features.append(feature)
        
        # Métriques globales
        total_features = len(numerical_features) + len(categorical_features)
        drift_ratio = len(drifted_features) / total_features if total_features > 0 else 0
        
        summary = {
            'total_features_monitored': total_features,
            'drifted_features_count': len(drifted_features),
            'drift_ratio': drift_ratio,
            'drift_detected': len(drifted_features) > 0,
            'drifted_features': drifted_features,
            'timestamp': datetime.now().isoformat(),
            'threshold_used': threshold
        }
        
        all_results['summary'] = summary
        self.drift_results = all_results
        
        return all_results
    
    def generate_drift_report(self, save_path: str = None):
        """Génère un rapport complet de drift"""
        if not self.drift_results:
            self.detect_drift_all_features()
        
        report = {
            'drift_analysis': self.drift_results,
            'monitoring_config': {
                'numerical_features': config.get('features.numerical', []),
                'categorical_features': config.get('features.categorical', [])
            },
            'data_statistics': {
                'reference_data_shape': self.reference_data.shape,
                'current_data_shape': self.current_data.shape,
                'reference_data_period': {
                    'min_date': self.reference_data.get('session_start', pd.Series()).min(),
                    'max_date': self.reference_data.get('session_start', pd.Series()).max()
                } if 'session_start' in self.reference_data.columns else {},
                'current_data_period': {
                    'min_date': self.current_data.get('session_start', pd.Series()).min(),
                    'max_date': self.current_data.get('session_start', pd.Series()).max()
                } if 'session_start' in self.current_data.columns else {}
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report
    
    def plot_drift_analysis(self, save_path: str = None):
        """Crée des visualisations pour l'analyse de drift"""
        if not self.drift_results:
            self.detect_drift_all_features()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Features avec drift
        drifted_features = self.drift_results.get('summary', {}).get('drifted_features', [])
        non_drifted_features = [f for f in self.drift_results.keys() 
                              if f != 'summary' and f not in drifted_features]
        
        axes[0, 0].pie([len(drifted_features), len(non_drifted_features)], 
                      labels=['Drift Détecté', 'Pas de Drift'], 
                      autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        axes[0, 0].set_title('Distribution du Drift par Feature')
        
        # 2. Top features avec le plus de drift (p-values)
        p_values = []
        feature_names = []
        for feature, result in self.drift_results.items():
            if feature != 'summary' and 'p_value' in result:
                p_values.append(result['p_value'])
                feature_names.append(feature)
        
        if p_values:
            top_drift_idx = np.argsort(p_values)[:10]
            top_features = [feature_names[i] for i in top_drift_idx]
            top_p_values = [p_values[i] for i in top_drift_idx]
            
            axes[0, 1].barh(top_features, top_p_values, color='salmon')
            axes[0, 1].axvline(x=0.05, color='red', linestyle='--', label='Seuil (0.05)')
            axes[0, 1].set_xlabel('P-value')
            axes[0, 1].set_title('Top 10 Features avec le Plus de Drift')
            axes[0, 1].legend()
        
        # 3. Distribution des p-values
        axes[1, 0].hist(p_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(x=0.05, color='red', linestyle='--', label='Seuil de significativité')
        axes[1, 0].set_xlabel('P-value')
        axes[1, 0].set_ylabel('Nombre de Features')
        axes[1, 0].set_title('Distribution des P-values de Drift')
        axes[1, 0].legend()
        
        # 4. Drift temporel (si disponible)
        if 'session_start' in self.reference_data.columns and 'session_start' in self.current_data.columns:
            ref_dates = pd.to_datetime(self.reference_data['session_start']).dt.date
            curr_dates = pd.to_datetime(self.current_data['session_start']).dt.date
            
            ref_date_counts = ref_dates.value_counts().sort_index()
            curr_date_counts = curr_dates.value_counts().sort_index()
            
            axes[1, 1].plot(ref_date_counts.index, ref_date_counts.values, 
                           label='Données de Référence', alpha=0.7)
            axes[1, 1].plot(curr_date_counts.index, curr_date_counts.values, 
                           label='Données Courantes', alpha=0.7)
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Nombre de Sessions')
            axes[1, 1].set_title('Distribution Temporelle des Sessions')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig

class ModelPerformanceMonitor:
    def __init__(self, model, reference_metrics: Dict[str, float]):
        self.model = model
        self.reference_metrics = reference_metrics
        self.performance_history = []
    
    def calculate_model_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Calcule les métriques de performance du modèle"""
        from sklearn.metrics import (roc_auc_score, accuracy_score, 
                                   precision_score, recall_score, f1_score,
                                   confusion_matrix)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'auc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'log_loss': -np.log(y_pred_proba[y_test == 1]).mean() if (y_test == 1).any() else 0
        }
        
        # Matrice de confusion
        cm = confusion_matrix(y_test, y_pred)
        metrics.update({
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        })
        
        return metrics
    
    def detect_performance_drift(self, X_test: pd.DataFrame, y_test: pd.Series, 
                               threshold: float = 0.1) -> Dict[str, Any]:
        """Détecte la dérive de performance du modèle"""
        current_metrics = self.calculate_model_metrics(X_test, y_test)
        
        performance_drift = {}
        drift_detected = False
        drifted_metrics = []
        
        for metric_name, ref_value in self.reference_metrics.items():
            if metric_name in current_metrics:
                curr_value = current_metrics[metric_name]
                
                # Calcul de la dégradation relative
                if ref_value != 0:
                    degradation = (ref_value - curr_value) / ref_value
                else:
                    degradation = 0
                
                metric_drift = abs(degradation) > threshold
                
                performance_drift[metric_name] = {
                    'reference_value': ref_value,
                    'current_value': curr_value,
                    'degradation': degradation,
                    'degradation_absolute': ref_value - curr_value,
                    'drift_detected': metric_drift,
                    'threshold': threshold
                }
                
                if metric_drift:
                    drift_detected = True
                    drifted_metrics.append(metric_name)
        
        # Enregistrement dans l'historique
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'drift_detected': drift_detected,
            'drifted_metrics': drifted_metrics
        })
        
        summary = {
            'performance_drift_detected': drift_detected,
            'drifted_metrics': drifted_metrics,
            'total_metrics_monitored': len(self.reference_metrics),
            'threshold_used': threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'summary': summary,
            'detailed_metrics': performance_drift,
            'current_performance': current_metrics
        }
    
    def log_to_mlflow(self, drift_results: Dict[str, Any], run_name: str = "monitoring"):
        """Log les résultats de monitoring dans MLflow"""
        with mlflow.start_run(run_name=run_name):
            # Log des métriques de performance
            mlflow.log_metrics(drift_results['current_performance'])
            
            # Log des indicateurs de drift
            mlflow.log_metrics({
                'performance_drift_detected': int(drift_results['summary']['performance_drift_detected']),
                'drifted_metrics_count': len(drift_results['summary']['drifted_metrics'])
            })
            
            # Log des paramètres de monitoring
            mlflow.log_params({
                'monitoring_threshold': drift_results['summary']['threshold_used'],
                'metrics_monitored_count': drift_results['summary']['total_metrics_monitored']
            })
            
            # Log des résultats détaillés
            mlflow.log_dict(drift_results, "monitoring_results.json")
            
            print("✅ Résultats de monitoring loggés dans MLflow")

class MonitoringPipeline:
    def __init__(self, reference_data: pd.DataFrame, model, reference_metrics: Dict[str, float]):
        self.data_drift_detector = DataDriftDetector(
            reference_data, 
            pd.DataFrame()  # Données courantes à définir plus tard
        )
        self.performance_monitor = ModelPerformanceMonitor(model, reference_metrics)
    
    def run_monitoring(self, current_data: pd.DataFrame, X_test: pd.DataFrame, 
                      y_test: pd.Series, data_drift_threshold: float = 0.05,
                      performance_threshold: float = 0.1) -> Dict[str, Any]:
        """Exécute le pipeline complet de monitoring"""
        print("🔍 Lancement du monitoring...")
        
        # Mise à jour des données courantes
        self.data_drift_detector.current_data = current_data
        
        # 1. Détection de drift des données
        print("1. Analyse du drift des données...")
        data_drift_results = self.data_drift_detector.detect_drift_all_features(data_drift_threshold)
        
        # 2. Détection de drift de performance
        print("2. Analyse du drift de performance...")
        performance_drift_results = self.performance_monitor.detect_performance_drift(
            X_test, y_test, performance_threshold
        )
        
        # 3. Génération de rapports
        print("3. Génération des rapports...")
        data_drift_report = self.data_drift_detector.generate_drift_report()
        
        # 4. Log dans MLflow
        print("4. Logging dans MLflow...")
        self.performance_monitor.log_to_mlflow(performance_drift_results)
        
        # Rapport consolidé
        consolidated_report = {
            'timestamp': datetime.now().isoformat(),
            'data_drift': data_drift_report,
            'performance_drift': performance_drift_results,
            'alerts': {
                'data_drift_alert': data_drift_results['summary']['drift_detected'],
                'performance_drift_alert': performance_drift_results['summary']['performance_drift_detected'],
                'critical_alert': (data_drift_results['summary']['drift_detected'] and 
                                 performance_drift_results['summary']['performance_drift_detected'])
            }
        }
        
        print("✅ Monitoring terminé")
        return consolidated_report
    
    def generate_monitoring_dashboard(self, current_data: pd.DataFrame, 
                                    X_test: pd.DataFrame, y_test: pd.Series,
                                    save_dir: str = "monitoring_reports"):
        """Génère un dashboard complet de monitoring"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Exécution du monitoring
        report = self.run_monitoring(current_data, X_test, y_test)
        
        # Graphiques de drift
        drift_plot_path = os.path.join(save_dir, "drift_analysis.png")
        self.data_drift_detector.plot_drift_analysis(drift_plot_path)
        
        # Sauvegarde du rapport
        report_path = os.path.join(save_dir, "monitoring_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Graphique de performance historique
        self._plot_performance_history(save_dir)
        
        print(f"📊 Dashboard généré dans: {save_dir}")
        return report
    
    def _plot_performance_history(self, save_dir: str):
        """Plot l'historique des performances"""
        if len(self.performance_monitor.performance_history) < 2:
            return
        
        history = self.performance_monitor.performance_history
        timestamps = [pd.to_datetime(h['timestamp']) for h in history]
        
        metrics_to_plot = ['auc', 'accuracy', 'f1_score']
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 10))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [h['metrics'].get(metric, 0) for h in history]
            axes[i].plot(timestamps, values, marker='o', linewidth=2)
            axes[i].set_ylabel(metric.upper())
            axes[i].set_title(f'Évolution de {metric.upper()}')
            axes[i].grid(True, alpha=0.3)
            
            # Marquer les points avec drift
            for j, h in enumerate(history):
                if metric in h.get('drifted_metrics', []):
                    axes[i].plot(timestamps[j], values[j], 'ro', markersize=8)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/performance_history.png", dpi=300, bbox_inches='tight')
        plt.close()