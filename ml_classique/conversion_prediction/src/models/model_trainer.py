import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import pickle
from ..utils.config_manager import config


class ModelTrainer:
    def __init__(self):
        self.experiment_name = config.get('mlflow.experiment_name', 'default_experiment')
        mlflow.set_experiment(self.experiment_name)
        os.makedirs("models", exist_ok=True)  # Création du dossier pour sauvegarde

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                     X_val: pd.DataFrame, y_val: pd.Series) -> dict:
        """Entraîne plusieurs modèles et les compare"""

        # 🧹 Nettoyage et imputations
        print("🧹 Vérification et traitement des valeurs manquantes...")

        # 1️⃣ Nettoyage des cibles
        if y_train.isnull().any():
            print("⚠️ Valeurs manquantes détectées dans y_train : suppression des lignes correspondantes.")
            mask = ~y_train.isnull()
            X_train = X_train.loc[mask]
            y_train = y_train.loc[mask]

        if y_val.isnull().any():
            print("⚠️ Valeurs manquantes détectées dans y_val : suppression des lignes correspondantes.")
            mask = ~y_val.isnull()
            X_val = X_val.loc[mask]
            y_val = y_val.loc[mask]

        # 2️⃣ Imputation des features (médiane pour numériques, constante pour catégorielles)
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        cat_cols = X_train.select_dtypes(exclude=[np.number]).columns

        if len(num_cols) > 0:
            imputer_num = SimpleImputer(strategy="median")
            X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
            X_val[num_cols] = imputer_num.transform(X_val[num_cols])

        if len(cat_cols) > 0:
            imputer_cat = SimpleImputer(strategy="most_frequent")
            X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
            X_val[cat_cols] = imputer_cat.transform(X_val[cat_cols])

        # ✅ Vérification post-imputation
        if X_train.isnull().sum().sum() > 0 or X_val.isnull().sum().sum() > 0:
            raise ValueError("❌ Des NaN subsistent après imputation dans X_train ou X_val.")

        if X_train.empty or y_train.empty:
            raise ValueError("❌ Les données d'entraînement sont vides après nettoyage.")
        if len(np.unique(y_train)) < 2:
            raise ValueError("❌ La variable cible y_train ne contient pas au moins deux classes.")

        # 📦 Définition des modèles
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=config.get('train.n_estimators', 100),
                max_depth=config.get('train.max_depth', 6),
                random_state=config.get('model.random_state', 42)
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=config.get('train.n_estimators', 100),
                max_depth=config.get('train.max_depth', 6),
                learning_rate=config.get('train.learning_rate', 0.1),
                random_state=config.get('model.random_state', 42)
            ),
            'logistic_regression': LogisticRegression(
                random_state=config.get('model.random_state', 42),
                max_iter=1000
            )
        }

        best_model = None
        best_score = 0
        results = {}

        # 🚀 Entraînement et évaluation
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                print(f"🎯 Entraînement du modèle {model_name}...")

                model.fit(X_train, y_train)

                # Prédictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

                # Métriques
                auc_score = roc_auc_score(y_val, y_pred_proba) if y_pred_proba is not None else 0.0
                accuracy = (y_pred == y_val).mean()

                # Validation croisée
                try:
                    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
                except Exception as e:
                    print(f"⚠️ Erreur pendant la validation croisée pour {model_name}: {e}")
                    cv_scores = np.array([np.nan])

                # Log des paramètres et métriques dans MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    'auc': auc_score,
                    'accuracy': accuracy,
                    'cv_auc_mean': np.nanmean(cv_scores),
                    'cv_auc_std': np.nanstd(cv_scores)
                })

                # Sauvegarde du modèle dans MLflow
                mlflow.sklearn.log_model(model, model_name)

                # Feature importance
                if hasattr(model, 'feature_importances_'):
                    self._log_feature_importance(model, X_train.columns, model_name)

                # Sauvegarde des résultats
                results[model_name] = {
                    'model': model,
                    'auc': auc_score,
                    'accuracy': accuracy,
                    'cv_scores': cv_scores
                }

                # Sélection du meilleur modèle
                if auc_score > best_score:
                    best_score = auc_score
                    best_model = model_name

        # 💾 Sauvegarde du meilleur modèle
        best_model_obj = results[best_model]['model']
        model_path = 'models/conversion_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(best_model_obj, f)
        mlflow.log_artifact(model_path)

        # 📊 Sauvegarde des métriques pour DVC
        metrics = {
            'best_model': best_model,
            'best_auc': best_score,
            'models': {name: {'auc': result['auc'], 'accuracy': result['accuracy']}
                       for name, result in results.items()}
        }
        metrics_path = 'models/metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(metrics_path)

        print(f"✅ Entraînement terminé. Meilleur modèle : {best_model} (AUC = {best_score:.4f})")

        return results

    def _log_feature_importance(self, model, feature_names, model_name):
        """Log et visualise l'importance des features"""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Graphique d'importance
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.title(f'Feature Importance - {model_name}')
        plt.tight_layout()

        img_path = f'models/feature_importance_{model_name}.png'
        plt.savefig(img_path)
        mlflow.log_artifact(img_path)
        plt.close()

        # Log CSV
        csv_content = importance_df.to_csv(index=False)
        mlflow.log_text(csv_content, f"feature_importance_{model_name}.csv")
