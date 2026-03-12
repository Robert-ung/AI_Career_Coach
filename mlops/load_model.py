import joblib
import json
import pandas as pd
from pathlib import Path

model_name = "job-matcher-classifier"
models_dir = Path(__file__).parent.parent / "models" / model_name / "artifacts"

try:
    model  = joblib.load(models_dir / "model.pkl")
    scaler = joblib.load(models_dir / "scaler.pkl")

    # ✅ CHANGÉ : metadata.json au lieu de features.txt
    with open(models_dir / "metadata.json") as f:
        metadata = json.load(f)
    features = metadata['features']

    print(f"✅ Modèle chargé : {model_name}")
    print(f"   • Features  : {len(features)}")         # ✅ doit afficher 27
    print(f"   • Accuracy  : {metadata['test_accuracy']:.4f}")

except Exception as e:
    print(f"❌ Erreur : {e}")
    exit(1)