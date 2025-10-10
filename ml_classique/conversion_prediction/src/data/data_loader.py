import pandas as pd
import os
from typing import Tuple, Dict
from ..utils.config_manager import config

class DataLoader:
    def __init__(self):
        self.raw_path = config.get('data.raw_path')
        self.processed_path = config.get('data.processed_path')
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Charge toutes les données brutes"""
        data_files = {
            'sessions': 'sessions.csv',
            'users': 'users.csv', 
            'products': 'products.csv',
            'clicks': 'clicks.csv',
            'transactions': 'transactions.csv'
        }
        
        data = {}
        for name, filename in data_files.items():
            filepath = os.path.join(self.raw_path, filename)
            if os.path.exists(filepath):
                data[name] = pd.read_csv(filepath)
            else:
                print(f"⚠️  Fichier {filepath} non trouvé")
        
        return data
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Sauvegarde les données processées"""
        os.makedirs(self.processed_path, exist_ok=True)
        train_df.to_csv(os.path.join(self.processed_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(self.processed_path, 'test.csv'), index=False)