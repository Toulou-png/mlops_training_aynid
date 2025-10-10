import yaml
import os
from typing import Dict, Any

class ConfigManager:
    def __init__(self, config_path: str = "config/base.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def get(self, key: str, default=None) -> Any:
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default
    
    def update_from_dvc(self, dvc_params: Dict[str, Any]):
        """Met à jour la config avec les paramètres DVC"""
        for key, value in dvc_params.items():
            keys = key.split('.')
            config_level = self.config
            for k in keys[:-1]:
                config_level = config_level.setdefault(k, {})
            config_level[keys[-1]] = value

config = ConfigManager()