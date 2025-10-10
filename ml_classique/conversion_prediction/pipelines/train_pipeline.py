import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_loader import DataLoader
from src.data.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.utils.config_manager import ConfigManager
from sklearn.model_selection import train_test_split
import pandas as pd

def prepare_data():
    """Pipeline de préparation des données"""
    print("📊 Préparation des données...")
    
    # Chargement des données
    loader = DataLoader()
    raw_data = loader.load_raw_data()
    
    # Feature engineering
    engineer = FeatureEngineer()
    features_df = engineer.create_features(raw_data)
    
    # Séparation train/test
    target = ConfigManager().get('model.target')
    X = features_df.drop(columns=[target, 'session_id', 'user_id'])
    y = features_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=ConfigManager().get('model.test_size'),
        random_state=ConfigManager().get('model.random_state'),
        stratify=y
    )
    
    # Préprocessing
    engineer.fit_preprocessor(X_train)
    X_train_processed = engineer.transform_features(X_train)
    X_test_processed = engineer.transform_features(X_test)
    
    # Sauvegarde
    train_df = pd.concat([X_train_processed, y_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([X_test_processed, y_test.reset_index(drop=True)], axis=1)
    
    loader.save_processed_data(train_df, test_df)
    print("✅ Données préparées et sauvegardées")

def train_model():
    """Pipeline d'entraînement du modèle"""
    print("🤖 Entraînement du modèle...")
    
    # Chargement des données processées
    processed_path = ConfigManager().get('data.processed_path')
    train_df = pd.read_csv(os.path.join(processed_path, 'train.csv'))
    test_df = pd.read_csv(os.path.join(processed_path, 'test.csv'))
    
    # Séparation features/target
    target = ConfigManager().get('model.target')
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]
    
    # Entraînement
    trainer = ModelTrainer()
    results = trainer.train_models(X_train, y_train, X_test, y_test)
    
    print("✅ Modèle entraîné et sauvegardé")

def evaluate_model():
    """Pipeline d'évaluation du modèle"""
    print("📈 Évaluation du modèle...")
    # Implémentation similaire...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=['prepare', 'train', 'evaluate'])
    args = parser.parse_args()
    
    if args.stage == 'prepare':
        prepare_data()
    elif args.stage == 'train':
        train_model()
    elif args.stage == 'evaluate':
        evaluate_model()