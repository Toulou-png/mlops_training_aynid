import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from typing import Tuple
import os
from ..utils.config_manager import config


class FeatureEngineer:
    def __init__(self):
        self.features_path = config.get('data.features_path')
        self.preprocessor = None
        self.feature_names = []
        
    def create_features(self, data: dict) -> pd.DataFrame:
        """Crée le dataset de features pour la prédiction de conversion"""
        
        sessions = data['sessions'].copy()
        users = data['users'].copy()
        clicks = data['clicks'].copy()
        transactions = data['transactions'].copy()
    
        # Création des différentes features
        session_features = self._create_session_features(sessions, clicks)
        user_features = self._create_user_features(users, transactions)
        time_features = self._create_time_features(sessions)
        
        # Fusion
        features_df = session_features.merge(user_features, on='user_id', how='left')
        features_df = features_df.merge(time_features, on='session_id', how='left')
        
        # Variable cible
        features_df = self._create_target(features_df, transactions)
        
        return features_df
    

    def _create_session_features(self, sessions: pd.DataFrame, clicks: pd.DataFrame) -> pd.DataFrame:
        """Features liées à la session"""
        session_clicks = clicks.groupby('session_id').agg({
            'product_id': 'count',
            'time_spent': ['sum', 'mean'],
            'page_views': 'sum'
        }).reset_index()
        
        session_clicks.columns = ['session_id', 'click_count', 'total_time_spent', 
                                  'avg_time_per_click', 'total_page_views']
        
        features = sessions.merge(session_clicks, on='session_id', how='left')
        features = features.fillna({
            'click_count': 0,
            'total_time_spent': 0,
            'avg_time_per_click': 0,
            'total_page_views': 0
        })
        
        return features[['session_id', 'user_id', 'device_type', 'location', 
                         'session_duration', 'click_count', 'total_time_spent',
                         'avg_time_per_click', 'total_page_views']]
    

    def _create_user_features(self, users: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """Features liées à l'historique utilisateur"""

        expected_cols = {'user_id', 'transaction_id', 'product_id'}
        if not expected_cols.issubset(transactions.columns):
            raise ValueError(f"Colonnes manquantes dans transactions: {expected_cols - set(transactions.columns)}")

        # Détection automatique du montant
        amount_col = 'total_amount' if 'total_amount' in transactions.columns else \
                     'amount' if 'amount' in transactions.columns else None
        if amount_col is None:
            raise ValueError("La colonne du montant n'a pas été trouvée dans transactions (attendu: 'total_amount' ou 'amount').")

        user_history = transactions.groupby('user_id').agg({
            'transaction_id': 'count',
            amount_col: ['sum', 'mean'],
            'product_id': 'nunique'
        }).reset_index()

        user_history.columns = ['user_id', 'total_purchases', 'total_spent', 
                                'avg_order_value', 'unique_products']

        # Merge sans doublons
        features = users.merge(user_history, on='user_id', how='left', suffixes=('', '_history'))

        features = features.fillna({
            'total_purchases': 0,
            'total_spent': 0,
            'avg_order_value': 0,
            'unique_products': 0
        })

        expected_features = [
            'user_id', 'age', 'gender', 'preferred_category', 'total_orders',
            'total_purchases', 'total_spent', 'avg_order_value', 'unique_products'
        ]

        existing_features = [col for col in expected_features if col in features.columns]
        return features[existing_features]
    

    def _create_time_features(self, sessions: pd.DataFrame) -> pd.DataFrame:
        """Features temporelles"""
        sessions = sessions.copy()
        sessions['session_start'] = pd.to_datetime(sessions['session_start'])
        sessions['session_start_hour'] = sessions['session_start'].dt.hour
        sessions['session_start_dayofweek'] = sessions['session_start'].dt.dayofweek
        sessions['is_weekend'] = sessions['session_start_dayofweek'].isin([5, 6]).astype(int)
        
        return sessions[['session_id', 'session_start_hour', 'session_start_dayofweek', 'is_weekend']]
    

    def _create_target(self, features_df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
        """Crée la variable cible will_purchase"""
        completed_transactions = transactions[transactions['status'] == 'completed']
        purchasing_sessions = set(completed_transactions['session_id'])
        features_df['will_purchase'] = features_df['session_id'].isin(purchasing_sessions).astype(int)
        return features_df
    

    def fit_preprocessor(self, df: pd.DataFrame):
        """Entraîne le préprocesseur sur les données"""
        categorical_features = config.get('features.categorical') or []
        numerical_features = config.get('features.numerical') or []
        
        # Vérif colonnes
        missing = [c for c in categorical_features + numerical_features if c not in df.columns]
        if missing:
            raise ValueError(f"Colonnes manquantes dans le DataFrame d'entraînement : {missing}")
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        self.feature_names = numerical_features + categorical_features
        self.preprocessor.fit(df[self.feature_names])
        
        os.makedirs(self.features_path, exist_ok=True)
        with open(os.path.join(self.features_path, 'preprocessor.pkl'), 'wb') as f:
            pickle.dump(self.preprocessor, f)
        with open(os.path.join(self.features_path, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
    

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforme les features avec le préprocesseur"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_preprocessor first.")
        
        # Vérif colonnes présentes
        missing_cols = [c for c in self.feature_names if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Colonnes manquantes dans les features : {missing_cols}")

        # Transformation
        transformed = self.preprocessor.transform(df[self.feature_names])

        # Conversion si sortie sparse
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        # Récupération correcte des noms de features
        try:
            feature_names_out = self.preprocessor.get_feature_names_out(self.feature_names)
        except Exception:
            # Récupération manuelle
            feature_names_out = []
            for name, transformer, features in self.preprocessor.transformers_:
                if name == 'num':
                    feature_names_out.extend(features)
                elif name == 'cat' and hasattr(transformer, 'get_feature_names_out'):
                    cat_features = transformer.get_feature_names_out(features)
                    feature_names_out.extend(cat_features)

        # Vérification cohérence shape
        if transformed.shape[1] != len(feature_names_out):
            print(f"⚠️ Avertissement : Shape mismatch - transformed={transformed.shape}, feature_names_out={len(feature_names_out)}")
            # Ajustement fallback si nécessaire
            feature_names_out = [f"f_{i}" for i in range(transformed.shape[1])]

        print(f"✅ Shape transformé final : {transformed.shape}")

        return pd.DataFrame(transformed, columns=feature_names_out, index=df.index)
