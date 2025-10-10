# 1 ----> Initialization

# Clone ou création du projet
mkdir conversion_prediction
cd conversion_prediction

# Virtual environnement
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Installation des dépendances
pip install -r requirements.txt



# Initialisation DVC
dvc init
git add .
git commit -m "Initial commit"

# Add a remote storage if you have one
dvc remote add -d myremote s3://my-bucket/dvc-store
dvc remote modify myremote access_key_id <YOUR_KEY>
dvc remote modify myremote secret_access_key <YOUR_SECRET>

# Track large raw datasets
dvc add data/raw/sessions.csv
git add data/raw/sessions.csv.dvc .gitignore
git commit -m "Add raw dataset"

# Initialisation MLflow
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000

# 2-----> Pipeline execution
# Méthode 1: Via DVC (recommandé)
python train.py

# Méthode 2: Étape par étape
dvc repro

# Optional ycu can run steps individually
dvc repro prepare
dvc repro train
dvc repro evaluate

# Vérification des métriques
dvc metrics show

# Interface MLflow
mlflow ui

# 3----> Monitoring basique

http://localhost:5000

# Comparaison d'experiences
mlflow experiments compare

# 4----> Lancement du monitoring global
python pipelines/monitoring_pipeline.py 
streamlit run monitoring_dashboard.py