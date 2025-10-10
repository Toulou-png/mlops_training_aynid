import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from sklearn.datasets import make_blobs

# Initialisation
fake = Faker('fr_FR')
np.random.seed(42)
random.seed(42)

def generate_aynid_data(n_users=5000, n_products=1000, n_sessions=20000):
    """
    Génère un jeu de données complet pour la plateforme Aynid
    """
    print("🔄 Génération des données Aynid...")
    
    # 1. DONNÉES UTILISATEURS
    print("👥 Génération des utilisateurs...")
    users = []
    for i in range(n_users):
        user = {
            'user_id': f"user_{i:05d}",
            'age': np.random.randint(18, 70),
            'gender': np.random.choice(['M', 'F'], p=[0.48, 0.52]),
            'location': fake.city(),
            'registration_date': fake.date_between(start_date='-2y', end_date='today'),
            'total_orders': np.random.poisson(5),
            'avg_order_value': np.random.normal(85, 30),
            'preferred_category': np.random.choice(['electronics', 'fashion', 'home', 'sports', 'beauty'])
        }
        user['avg_order_value'] = max(10, user['avg_order_value'])  # Éviter les valeurs négatives
        users.append(user)
    
    users_df = pd.DataFrame(users)
    
    # 2. DONNÉES PRODUITS
    print("📦 Génération des produits...")
    categories = ['electronics', 'fashion', 'home', 'sports', 'beauty', 'books', 'toys']
    subcategories = {
        'electronics': ['smartphone', 'laptop', 'headphones', 'tablet'],
        'fashion': ['clothing', 'shoes', 'accessories', 'watches'],
        'home': ['furniture', 'kitchen', 'decor', 'appliances'],
        'sports': ['fitness', 'outdoor', 'team_sports', 'yoga'],
        'beauty': ['skincare', 'makeup', 'haircare', 'fragrance'],
        'books': ['fiction', 'business', 'science', 'cookbooks'],
        'toys': ['educational', 'action_figures', 'board_games', 'dolls']
    }
    
    products = []
    for i in range(n_products):
        category = np.random.choice(categories)
        subcategory = np.random.choice(subcategories[category])
        
        product = {
            'product_id': f"prod_{i:05d}",
            'name': fake.catch_phrase(),
            'category': category,
            'subcategory': subcategory,
            'price': np.random.lognormal(4, 0.8),
            'brand': fake.company(),
            'rating': np.random.normal(4.2, 0.5),
            'review_count': np.random.poisson(45),
            'stock_quantity': np.random.randint(0, 1000),
            'is_featured': np.random.choice([True, False], p=[0.1, 0.9]),
            'creation_date': fake.date_between(start_date='-1y', end_date='today')
        }
        product['price'] = round(max(5, product['price']), 2)
        product['rating'] = max(1, min(5, round(product['rating'], 1)))
        products.append(product)
    
    products_df = pd.DataFrame(products)
    
    # 3. DONNÉES SESSIONS ET COMPORTEMENT
    print("🌐 Génération des sessions utilisateur...")
    sessions = []
    searches = []
    clicks = []
    
    for session_id in range(n_sessions):
        user_id = np.random.choice(users_df['user_id'])
        product_id = np.random.choice(products_df['product_id'])
        session_date = fake.date_time_between(start_date='-3months', end_date='now')
        
        # Données de session
        session = {
            'session_id': f"session_{session_id:06d}",
            'user_id': user_id,
            'session_start': session_date,
            'session_duration': np.random.exponential(300),  # en secondes
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], p=[0.6, 0.35, 0.05]),
            'location': np.random.choice(users_df[users_df['user_id'] == user_id]['location'].values)
        }
        sessions.append(session)
        
        # Données de recherche
        search_terms = [
            'smartphone pas cher', 'livre roman', 'chaussures sport', 'meuble salon',
            'cosmétique bio', 'jeux enfants', 'ordinateur portable', 'vêtement été',
            'équipement fitness', 'décoration maison', 'cadeau anniversaire'
        ]
        
        if np.random.random() > 0.3:  # 70% des sessions ont une recherche
            search = {
                'search_id': f"search_{session_id:06d}",
                'session_id': session['session_id'],
                'query': np.random.choice(search_terms),
                'results_count': np.random.randint(5, 100),
                'search_timestamp': session_date + timedelta(seconds=10)
            }
            searches.append(search)
        
        # Données de clics
        n_clicks = np.random.poisson(3)
        for click_num in range(n_clicks):
            click_product = np.random.choice(products_df['product_id'])
            click_time = session_date + timedelta(seconds=30 + click_num * 20)
            
            click = {
                'click_id': f"click_{session_id:06d}_{click_num:02d}",
                'session_id': session['session_id'],
                'product_id': click_product,
                'click_timestamp': click_time,
                'time_spent': np.random.exponential(45),
                'page_views': np.random.randint(1, 5)
            }
            clicks.append(click)
    
    sessions_df = pd.DataFrame(sessions)
    searches_df = pd.DataFrame(searches) if searches else pd.DataFrame()
    clicks_df = pd.DataFrame(clicks)
    
    # 4. DONNÉES TRANSACTIONS
    print("💰 Génération des transactions...")
    transactions = []
    n_transactions = int(n_sessions * 0.1)  # 10% des sessions donnent une transaction
    
    for i in range(n_transactions):
        session = sessions_df.iloc[np.random.randint(0, len(sessions_df))]
        user_id = session['user_id']
        
        # Sélectionner des produits cohérents avec la catégorie préférée de l'utilisateur
        user_pref = users_df[users_df['user_id'] == user_id]['preferred_category'].values[0]
        available_products = products_df[products_df['category'] == user_pref]
        
        if len(available_products) > 0:
            product_id = np.random.choice(available_products['product_id'])
        else:
            product_id = np.random.choice(products_df['product_id'])
        
        product_price = products_df[products_df['product_id'] == product_id]['price'].values[0]
        
        transaction = {
            'transaction_id': f"trans_{i:06d}",
            'user_id': user_id,
            'product_id': product_id,
            'session_id': session['session_id'],
            'purchase_date': session['session_start'] + timedelta(minutes=30),
            'quantity': np.random.randint(1, 4),
            'unit_price': product_price,
            'total_amount': 0,  # Calculé plus bas
            'status': np.random.choice(['completed', 'cancelled', 'refunded'], p=[0.85, 0.1, 0.05]),
            'payment_method': np.random.choice(['credit_card', 'paypal', 'bank_transfer'])
        }
        transaction['total_amount'] = transaction['quantity'] * transaction['unit_price']
        transactions.append(transaction)
    
    transactions_df = pd.DataFrame(transactions)
    
    # 5. DONNÉES AVIS
    print("⭐ Génération des avis produits...")
    reviews = []
    
    # Seulement pour les transactions complétées
    completed_transactions = transactions_df[transactions_df['status'] == 'completed']
    
    for _, transaction in completed_transactions.iterrows():
        if np.random.random() > 0.7:  # 30% des achats ont un avis
            review = {
                'review_id': f"review_{len(reviews):06d}",
                'user_id': transaction['user_id'],
                'product_id': transaction['product_id'],
                'rating': np.random.randint(1, 6),
                'review_text': fake.paragraph(nb_sentences=2),
                'review_date': transaction['purchase_date'] + timedelta(days=np.random.randint(1, 14)),
                'helpful_votes': np.random.poisson(2)
            }
            reviews.append(review)
    
    reviews_df = pd.DataFrame(reviews)
    
    # 6. DONNÉES POUR CLUSTERING (features utilisateur)
    print("🔍 Génération des features pour clustering...")
    
    # Créer des features comportementales synthétiques
    X, y = make_blobs(n_samples=n_users, centers=4, n_features=5, random_state=42)
    
    clustering_features = []
    for i, user_id in enumerate(users_df['user_id']):
        features = {
            'user_id': user_id,
            'feature_1': X[i][0],
            'feature_2': X[i][1],
            'feature_3': X[i][2],
            'feature_4': X[i][3],
            'feature_5': X[i][4],
            'cluster_label': y[i]
        }
        clustering_features.append(features)
    
    clustering_df = pd.DataFrame(clustering_features)
    
    return {
        'users': users_df,
        'products': products_df,
        'sessions': sessions_df,
        'searches': searches_df,
        'clicks': clicks_df,
        'transactions': transactions_df,
        'reviews': reviews_df,
        'clustering_features': clustering_df
    }

def create_prediction_targets(data_dict):
    """
    Crée des variables cibles pour les problèmes de prédiction
    """
    print("\n🎯 Création des variables cibles pour la prédiction...")
    
    # 1. Cible pour la prédiction d'achat
    transactions = data_dict['transactions']
    sessions = data_dict['sessions']
    
    # Marquer les sessions avec achat
    sessions_with_purchase = set(transactions[transactions['status'] == 'completed']['session_id'])
    data_dict['sessions']['will_purchase'] = data_dict['sessions']['session_id'].isin(sessions_with_purchase).astype(int)
    
    # 2. Cible pour la prédiction de panier abandonné
    # Simuler des données de panier
    carts = []
    for _, session in data_dict['sessions'].iterrows():
        if np.random.random() > 0.5:  # 50% des sessions ont un panier
            n_items = np.random.randint(1, 5)
            cart_value = np.random.normal(75, 25)
            
            cart = {
                'session_id': session['session_id'],
                'user_id': session['user_id'],
                'cart_items_count': n_items,
                'cart_value': max(10, cart_value),
                'cart_created': session['session_start'],
                'cart_abandoned': np.random.choice([True, False], p=[0.6, 0.4])
            }
            carts.append(cart)
    
    data_dict['carts'] = pd.DataFrame(carts)
    
    return data_dict

def save_datasets(data_dict, filepath="aynid_data"):
    """
    Sauvegarde tous les datasets dans des fichiers CSV
    """
    print(f"\n💾 Sauvegarde des données dans {filepath}...")
    
    for name, df in data_dict.items():
        filename = f"{filepath}_{name}.csv"
        df.to_csv(filename, index=False)
        print(f"✅ {filename} sauvegardé ({len(df)} lignes)")

def generate_sample_analysis(data_dict):
    """
    Génère une analyse rapide des données créées
    """
    print("\n📊 ANALYSE DES DONNÉES GÉNÉRÉES:")
    print("=" * 50)
    
    for name, df in data_dict.items():
        print(f"\n📁 {name.upper():20} : {len(df):6} enregistrements")
        if len(df) > 0:
            print(f"   Colonnes: {list(df.columns)}")
            if len(df) > 5:
                print(f"   Aperçu:")
                print(df.head(2).to_string(index=False))

# Exécution principale
if __name__ == "__main__":
    # Générer les données
    aynid_data = generate_aynid_data(
        n_users=1000,       # Réduit pour les tests
        n_products=500,     # Réduit pour les tests  
        n_sessions=5000     # Réduit pour les tests
    )
    
    # Ajouter les variables cibles
    aynid_data = create_prediction_targets(aynid_data)
    
    # Sauvegarder
    save_datasets(aynid_data, "ml_classique/conversion_prediction/data/raw/")
    
    # Afficher l'analyse
    generate_sample_analysis(aynid_data)
    
    print("\n🎉 Données Aynid générées avec succès!")
    print("\n📚 Cas d'usage couverts:")
    print("   • Prédiction d'achat (will_purchase)")
    print("   • Classification d'intention (searches, clicks)")
    print("   • Clustering utilisateurs (clustering_features)")
    print("   • Recommandation (transactions, reviews)")
    print("   • Analyse de comportement (sessions, clicks)")