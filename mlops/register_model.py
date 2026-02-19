import mlflow
import mlflow.xgboost
import joblib
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_TRACKING_URI = PROJECT_ROOT / "mlops" / "mlflow_tracking"
MODELS_DIR = PROJECT_ROOT / "models"  # ← NOUVEAU

# Créer dossier models/ s'il n'existe pas
MODELS_DIR.mkdir(exist_ok=True)  # ← NOUVEAU

mlflow.set_tracking_uri(f"file:{MLFLOW_TRACKING_URI}")

print("=" * 70)
print("🔍 RECHERCHE DU MEILLEUR MODÈLE")
print("=" * 70)

# ============================================================================
# ÉTAPE 1 : Récupérer le meilleur run (INCHANGÉ)
# ============================================================================
experiment = mlflow.get_experiment_by_name("job-matcher-ml")

if experiment is None:
    print("❌ ERREUR : Expérience 'job-matcher-ml' introuvable")
    print("💡 Vérifiez que vous avez bien exécuté : python mlops/train_and_log.py")
    exit(1)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.test_accuracy DESC"]
)

if runs.empty:
    print("❌ ERREUR : Aucun run trouvé dans l'expérience")
    exit(1)

best_run_id = runs.iloc[0]['run_id']
best_accuracy = runs.iloc[0]['metrics.test_accuracy']

print(f"🏆 Meilleur modèle trouvé :")
print(f"   • Run ID     : {best_run_id}")
print(f"   • Accuracy   : {best_accuracy:.4f}")
print(f"   • Date       : {runs.iloc[0]['start_time']}")
print()

# ============================================================================
# ÉTAPE 2 : Enregistrer dans Model Registry (INCHANGÉ)
# ============================================================================
model_uri = f"runs:/{best_run_id}/model"
model_name = "job-matcher-classifier"

print("=" * 70)
print("📦 ENREGISTREMENT DANS MODEL REGISTRY")
print("=" * 70)

try:
    # Enregistrer le modèle
    registered_model = mlflow.register_model(model_uri, model_name)
    print(f"✅ Modèle enregistré : {model_name}")
    print(f"   • Version : {registered_model.version}")
except Exception as e:
    print(f"⚠️  Modèle déjà enregistré (version existante)")
    print(f"   Détails : {str(e)}")

# ============================================================================
# ÉTAPE 3 : Promouvoir en Production (INCHANGÉ)
# ============================================================================
client = mlflow.tracking.MlflowClient()
versions = client.search_model_versions(f"name='{model_name}'")
latest_version = versions[0].version

client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Production"
)

print(f"✅ Modèle promu en Production (version {latest_version})")
print()

# ============================================================================
# ÉTAPE 4 : EXPORT POUR DOCKER (NOUVEAU - CRITIQUE)
# ============================================================================
print("=" * 70)
print("🐳 EXPORT POUR PRODUCTION DOCKER")
print("=" * 70)

try:
    # 4.1 Charger le modèle depuis MLflow
    print("📥 Chargement du modèle depuis MLflow...")
    model = mlflow.xgboost.load_model(model_uri)
    print(f"   ✅ Type : {type(model)}")
    
    # 4.2 Exporter le modèle .pkl
    output_pkl = MODELS_DIR / "ml_classifier_clean_v1.pkl"
    joblib.dump(model, output_pkl)
    file_size_mb = output_pkl.stat().st_size / (1024 * 1024)
    print(f"✅ Modèle exporté : {output_pkl}")
    print(f"   • Taille : {file_size_mb:.2f} MB")
    
    # 4.3 Créer fichier de métadonnées (pour traçabilité)
    metadata = {
        "model_name": model_name,
        "version": latest_version,
        "mlflow_run_id": best_run_id,
        "accuracy": float(best_accuracy),
        "exported_at": datetime.now().isoformat(),
        "model_uri": model_uri,
        "stage": "Production",
        "model_type": "XGBoost",
        "features_count": 15  # Ajuster selon votre config
    }
    
    output_metadata = MODELS_DIR / "classifier_clean_metadata.json"
    with open(output_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Métadonnées exportées : {output_metadata}")
    print()
    
    # 4.4 Vérifier que le fichier est accessible
    print("🔍 Vérification de l'export...")
    loaded_model = joblib.load(output_pkl)
    print(f"   ✅ Modèle rechargeable : {type(loaded_model)}")
    
except Exception as e:
    print(f"❌ ERREUR lors de l'export : {str(e)}")
    import traceback
    traceback.print_exc()
    exit(1)

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================
print("=" * 70)
print("✅ PIPELINE MLOps COMPLET")
print("=" * 70)
print(f"1. ✅ Meilleur modèle identifié (run_id: {best_run_id[:8]}...)")
print(f"2. ✅ Enregistré dans Model Registry (version {latest_version})")
print(f"3. ✅ Promu en Production")
print(f"4. ✅ Exporté vers models/ pour Docker")
print()
print("🚀 PROCHAINES ÉTAPES :")
print("   1. Vérifier : ls -lh models/")
print("   2. Redémarrer API : docker-compose restart api")
print("   3. Tester : curl http://localhost:8000/health")
print("=" * 70)