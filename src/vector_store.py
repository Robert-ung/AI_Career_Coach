"""
Module de gestion du Vector Store (FAISS)
Permet de stocker et rechercher efficacement les embeddings d'offres d'emploi
"""

import faiss
import numpy as np
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer


class JobVectorStore:
    """
    Gestionnaire de base vectorielle pour les offres d'emploi
    Utilise FAISS pour l'indexation et la recherche rapide
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """
        Initialiser le vector store
        
        Args:
            model_name: Nom du modèle Sentence-Transformers
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.jobs_metadata = []
        self.is_trained = False
    
    def build_index(self, jobs: List[Dict]) -> None:
        """
        Construire l'index FAISS à partir d'une liste d'offres
        
        Args:
            jobs: Liste de dictionnaires d'offres (avec 'description' minimum)
        """
        print(f"🔨 Construction de l'index FAISS pour {len(jobs)} offres...")
        
        # Extraire les descriptions et métadonnées
        descriptions = []
        self.jobs_metadata = []
        
        for job in jobs:
            # Description complète pour embedding
            full_desc = f"{job['title']}. {job['description']}"
            descriptions.append(full_desc)
            
            # Sauvegarder métadonnées (sans description pour économiser RAM)
            metadata = {k: v for k, v in job.items() if k != 'description'}
            self.jobs_metadata.append(metadata)
        
        # Générer embeddings
        print("📊 Génération des embeddings...")
        embeddings = self.model.encode(
            descriptions,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Créer index FAISS (IndexFlatIP pour similarité cosinus)
        print("🔧 Création de l'index FAISS...")
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normaliser les embeddings pour cosinus similarity
        faiss.normalize_L2(embeddings)
        
        # Ajouter à l'index
        self.index.add(embeddings)
        self.is_trained = True
        
        print(f"✅ Index construit : {self.index.ntotal} offres indexées")
    
    def search(
        self, 
        cv_text: str, 
        top_k: int = 10, 
        min_score: float = 0.0
    ) -> List[Dict]:
        """
        Rechercher les offres les plus similaires à un CV
        
        Args:
            cv_text: Texte du CV
            top_k: Nombre de résultats à retourner
            min_score: Score minimum (0-1)
        
        Returns:
            Liste de dict {job_metadata, score}
        """
        if not self.is_trained:
            raise ValueError("❌ L'index n'est pas construit. Appelez build_index() d'abord.")
        
        # Générer embedding du CV
        cv_embedding = self.model.encode([cv_text], convert_to_numpy=True)
        
        # Normaliser pour cosinus
        faiss.normalize_L2(cv_embedding)
        
        # Recherche dans FAISS
        scores, indices = self.index.search(cv_embedding, top_k)
        
        # Formater résultats
        results = []
        for score, idx in zip(scores[0], indices[0]):
            # Filtrer par score minimum
            if score < min_score:
                continue
            
            # Récupérer métadonnées
            job_meta = self.jobs_metadata[idx].copy()
            job_meta['faiss_score'] = float(score)
            job_meta['faiss_score_percent'] = float(score * 100)
            
            results.append(job_meta)
        
        return results
    
    def save(self, index_path: str, metadata_path: str) -> None:
        """
        Sauvegarder l'index et les métadonnées sur disque
        
        Args:
            index_path: Chemin pour l'index FAISS (.index)
            metadata_path: Chemin pour les métadonnées (.pkl)
        """
        if not self.is_trained:
            raise ValueError("❌ Rien à sauvegarder, l'index n'est pas construit.")
        
        # Sauvegarder index FAISS
        faiss.write_index(self.index, index_path)
        
        # Sauvegarder métadonnées + informations du modèle
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'jobs_metadata': self.jobs_metadata,
                'model_name': self.model_name,
                'dimension': self.dimension
            }, f)
        
        print(f"✅ Index sauvegardé : {index_path}")
        print(f"✅ Métadonnées sauvegardées : {metadata_path}")
        print(f"📌 Modèle utilisé : {self.model_name} ({self.dimension} dimensions)")
    
    def load(self, index_path: str, metadata_path: str) -> None:
        """
        Charger un index existant depuis le disque
        Détecte et charge automatiquement le bon modèle
        
        Args:
            index_path: Chemin de l'index FAISS
            metadata_path: Chemin des métadonnées
        """
        # Charger métadonnées d'abord pour connaître le modèle
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.jobs_metadata = data['jobs_metadata']
            saved_model_name = data.get('model_name', 'all-mpnet-base-v2')
            saved_dimension = data['dimension']
        
        # Vérifier la compatibilité du modèle actuel
        if self.model_name != saved_model_name:
            print(f"⚠️ Modèle différent détecté !")
            print(f"   - Index sauvegardé avec : {saved_model_name} ({saved_dimension} dim)")
            print(f"   - Modèle actuel : {self.model_name} ({self.dimension} dim)")
            
            # Si les dimensions ne correspondent pas, recharger le bon modèle
            if self.dimension != saved_dimension:
                print(f"🔄 Chargement du modèle correct...")
                self.model_name = saved_model_name
                self.model = SentenceTransformer(saved_model_name)
                self.dimension = self.model.get_sentence_embedding_dimension()
                print(f"✅ Modèle rechargé : {self.model_name}")
            else:
                # Même dimensions mais différent modèle → utiliser le sauvegardé
                print(f"✅ Dimensions identiques, utilisation du modèle sauvegardé")
                self.model_name = saved_model_name
        
        # Charger index FAISS
        self.index = faiss.read_index(index_path)
        self.is_trained = True
        
        print(f"✅ Index chargé : {self.index.ntotal} offres")
        print(f"📌 Modèle final : {self.model_name} ({self.dimension} dimensions)")
    
    def get_stats(self) -> Dict:
        """Obtenir des statistiques sur le vector store"""
        if not self.is_trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "total_jobs": self.index.ntotal,
            "dimension": self.dimension,
            "model": self.model_name
        }


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def create_vector_store_from_dataset(
    jobs_json_path: str,
    output_index_path: str = "models/faiss_jobs.index",
    output_metadata_path: str = "models/faiss_jobs_metadata.pkl",
    model_name: str = "all-mpnet-base-v2"
) -> JobVectorStore:
    """
    Créer et sauvegarder un vector store depuis un fichier JSON d'offres
    
    Args:
        jobs_json_path: Chemin du JSON des offres
        output_index_path: Où sauvegarder l'index FAISS
        output_metadata_path: Où sauvegarder les métadonnées
        model_name: Nom du modèle Sentence-Transformers
    
    Returns:
        JobVectorStore entraîné
    """
    # Charger offres
    with open(jobs_json_path, 'r', encoding='utf-8') as f:
        jobs = json.load(f)
    
    print(f"📂 Chargé {len(jobs)} offres depuis {jobs_json_path}")
    
    # Créer vector store
    vector_store = JobVectorStore(model_name=model_name)
    
    # Construire index
    vector_store.build_index(jobs)
    
    # Sauvegarder
    Path(output_index_path).parent.mkdir(parents=True, exist_ok=True)
    vector_store.save(output_index_path, output_metadata_path)
    
    return vector_store


def load_vector_store(
    index_path: str = "models/faiss_jobs.index",
    metadata_path: str = "models/faiss_jobs_metadata.pkl",
    model_name: str = None
) -> JobVectorStore:
    """
    Charger un vector store existant
    Le modèle est automatiquement détecté depuis les métadonnées
    
    Args:
        index_path: Chemin de l'index FAISS
        metadata_path: Chemin des métadonnées
        model_name: Nom du modèle (optionnel, auto-détecté)
    
    Returns:
        JobVectorStore chargé
    """
    # Si pas de modèle spécifié, utiliser le défaut (sera corrigé automatiquement)
    if model_name is None:
        model_name = "all-mpnet-base-v2"
    
    vector_store = JobVectorStore(model_name=model_name)
    vector_store.load(index_path, metadata_path)
    return vector_store


# ============================================================================
# TEST UNITAIRE
# ============================================================================

if __name__ == "__main__":
    """Test basique du module"""
    
    # Exemple de données
    test_jobs = [
        {
            "id": "1",
            "title": "Data Scientist",
            "description": "Python, Machine Learning, TensorFlow, PyTorch",
            "company": "TechCorp"
        },
        {
            "id": "2",
            "title": "Frontend Developer",
            "description": "React, JavaScript, HTML, CSS",
            "company": "WebAgency"
        }
    ]
    
    # Créer vector store
    vs = JobVectorStore(model_name="all-mpnet-base-v2")
    vs.build_index(test_jobs)
    
    # Test recherche
    cv = "Expérience en Python et Machine Learning"
    results = vs.search(cv, top_k=2)
    
    print("\n🔍 Résultats de recherche :")
    for r in results:
        print(f"  - {r['title']} (Score: {r['faiss_score_percent']:.1f}%)")
    
    # Test sauvegarde
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        idx_path = f"{tmpdir}/test.index"
        meta_path = f"{tmpdir}/test_meta.pkl"
        
        vs.save(idx_path, meta_path)
        
        # Test rechargement avec modèle différent
        vs2 = JobVectorStore(model_name="all-MiniLM-L6-v2")
        vs2.load(idx_path, meta_path)
        
        results2 = vs2.search(cv, top_k=2)
        print(f"\n✅ Rechargement OK : {len(results2)} résultats")
        print(f"📌 Modèle auto-corrigé : {vs2.model_name}")
    
    print("\n✅ Test réussi !")