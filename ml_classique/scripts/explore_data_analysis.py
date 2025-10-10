# Code pour explorer les données générées
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_aynid_data(filepath="ml_classique/conversion_prediction/data/"):
    """Charge tous les datasets Aynid"""
    datasets = {}
    suffixes = ['users', 'products', 'sessions', 'searches', 'clicks', 
                'transactions', 'reviews', 'clustering_features', 'carts']
    
    for suffix in suffixes:
        try:
            datasets[suffix] = pd.read_csv(f"{filepath}_{suffix}.csv")
        except FileNotFoundError:
            print(f"⚠️  Fichier {filepath}_{suffix}.csv non trouvé")
    
    return datasets

def explore_data_analysis(datasets):
    """Analyse exploratoire des données"""
    
    # 1. Statistiques utilisateurs
    print("=== ANALYSE UTILISATEURS ===")
    users = datasets['users']
    print(f"Nombre d'utilisateurs: {len(users)}")
    print(f"Répartition genre: {users['gender'].value_counts().to_dict()}")
    print(f"Âge moyen: {users['age'].mean():.1f} ans")
    
    # 2. Analyse produits
    print("\n=== ANALYSE PRODUITS ===")
    products = datasets['products']
    print(f"Nombre de produits: {len(products)}")
    print("Répartition par catégorie:")
    print(products['category'].value_counts())
    
    # 3. Analyse transactions
    print("\n=== ANALYSE TRANSACTIONS ===")
    transactions = datasets['transactions']
    print(f"Nombre de transactions: {len(transactions)}")
    print(f"Taux de conversion: {(len(transactions) / len(datasets['sessions']) * 100):.1f}%")
    print(f"Panier moyen: {transactions['total_amount'].mean():.2f}€")
    
    # 4. Visualisations
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    users['age'].hist(bins=20)
    plt.title('Distribution des âges')
    
    plt.subplot(2, 3, 2)
    products['category'].value_counts().plot.pie(autopct='%1.1f%%')
    plt.title('Répartition des catégories')
    
    plt.subplot(2, 3, 3)
    transactions['total_amount'].hist(bins=30)
    plt.title('Distribution des montants d\'achat')
    
    plt.subplot(2, 3, 4)
    datasets['sessions']['will_purchase'].value_counts().plot.bar()
    plt.title('Sessions avec achat')
    
    plt.subplot(2, 3, 5)
    if 'carts' in datasets:
        datasets['carts']['cart_abandoned'].value_counts().plot.pie(autopct='%1.1f%%')
        plt.title('Paniers abandonnés')
    
    plt.tight_layout()
    plt.show()

# Charger et analyser les données
datasets = load_aynid_data()
explore_data_analysis(datasets)