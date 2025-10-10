import os
import subprocess
import sys

def run_dvc_pipeline():
    """Exécute le pipeline DVC complet"""
    print("🚀 Lancement du pipeline MLOps...")
    
    # Étape 1: Préparation des données
    print("1. Préparation des données avec DVC...")
    subprocess.run(['dvc', 'repro'], check=True)
    
    # Étape 2: Vérification des métriques
    print("2. Affichage des métriques...")
    subprocess.run(['dvc', 'metrics', 'show'], check=True)
    
    # Étape 3: Push vers le storage DVC (si configuré)
    print("3. Sauvegarde des données versionnées...")
    subprocess.run(['dvc', 'push'], check=True)

if __name__ == "__main__":
    run_dvc_pipeline()