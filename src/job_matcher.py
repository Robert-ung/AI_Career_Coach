"""
Module de matching sémantique CV ↔ Offres d'emploi
Utilise Sentence-Transformers pour la similarité sémantique
"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class JobMatcher:
    """
    Classe pour matcher un CV avec des offres d'emploi
    """
    
    def __init__(self, model_name: str = 'all-mpnet-base-v2'):
        """
        Initialiser le matcher avec un modèle Sentence-Transformer
        
        Args:
            model_name: Nom du modèle (par défaut all-mpnet-base-v2)
        """
        print(f"🔍 Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        print(f"✅ Modèle chargé")
    
    def calculate_skills_similarity(
        self, 
        cv_skills: List[str], 
        job_description: str
    ) -> Dict:
        """
        Calculer la similarité entre compétences CV et description d'offre
        Méthode : Découpe la description en phrases et prend le meilleur match
        
        Args:
            cv_skills: Liste de compétences du CV
            job_description: Description complète de l'offre
            
        Returns:
            Dict avec score et détails par compétence
        """
        # Découper la description en phrases pertinentes
        job_sentences = [
            s.strip() 
            for s in job_description.split('\n') 
            if s.strip() and len(s.strip()) > 10
        ]
        
        if not job_sentences:
            # Fallback : utiliser la description complète
            job_sentences = [job_description]
        
        # Vectoriser toutes les phrases
        job_embeddings = self.model.encode(job_sentences)
        
        # Analyser chaque compétence
        matches = []
        
        for skill in cv_skills:
            skill_embedding = self.model.encode([skill.lower()])
            
            # Calculer similarité avec CHAQUE phrase
            similarities = cosine_similarity(skill_embedding, job_embeddings)[0]
            
            # Prendre la MEILLEURE similarité
            max_similarity = float(max(similarities)) * 100
            
            # Trouver quelle phrase matche le mieux
            best_match_idx = similarities.argmax()
            best_sentence = job_sentences[best_match_idx]
            
            matches.append({
                'skill': skill,
                'similarity': round(max_similarity, 2),
                'match': 'high' if max_similarity >= 40 else 'medium' if max_similarity >= 30 else 'low',
                'matched_sentence': best_sentence[:60] + '...' if len(best_sentence) > 60 else best_sentence
            })
        
        # Trier par similarité décroissante
        matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
        
        # Calculer score global
        if matches:
            avg_similarity = sum(m['similarity'] for m in matches) / len(matches)
            high_matches = len([m for m in matches if m['match'] == 'high'])
        else:
            avg_similarity = 0
            high_matches = 0
        
        return {
            'overall_score': round(avg_similarity, 2),
            'high_matches': high_matches,
            'total_skills': len(cv_skills),
            'matches': matches
        }
    
    def calculate_job_match_score(
        self, 
        cv_skills: List[str], 
        job: Dict
    ) -> Dict:
        """
        Calculer un score complet de matching entre CV et offre
        
        Args:
            cv_skills: Liste de compétences du CV
            job: Dictionnaire d'une offre d'emploi
            
        Returns:
            Dict avec score global et détails
        """
        # 1️⃣ Score de compétences (similarité sémantique)
        job_requirements = job.get('requirements', [])
        job_description = job.get('description', '')
        job_text = ' '.join(job_requirements) + '\n' + job_description
        
        # Utiliser la méthode améliorée (matching par phrases)
        skills_result = self.calculate_skills_similarity(cv_skills, job_text)
        skills_score = skills_result['overall_score']
        
        # 2️⃣ Score d'expérience
        experience_match = 50  # Score par défaut
        exp_required = job.get('experience', '').lower()
        
        if 'junior' in exp_required or '0-2 ans' in exp_required:
            experience_match = 100
        elif '1-3 ans' in exp_required or '2-4 ans' in exp_required:
            experience_match = 75
        elif 'senior' in exp_required or '5+' in exp_required:
            experience_match = 30
        
        # 3️⃣ Score de localisation
        location_score = 100 if job.get('remote_ok', False) else 80
        
        # 4️⃣ Score de compétitivité
        applicants = int(job.get('applicants', 50))
        if applicants < 30:
            competition_score = 100
        elif applicants < 80:
            competition_score = 70
        else:
            competition_score = 40
        
        # 🎯 SCORE GLOBAL (pondéré)
        global_score = (
            skills_score * 0.50 +       # 50% sur les compétences
            experience_match * 0.25 +   # 25% sur l'expérience
            location_score * 0.15 +     # 15% sur la localisation
            competition_score * 0.10    # 10% sur la compétition
        )
        
        return {
            'job_id': job.get('job_id', 'unknown'),
            'title': job['title'],
            'company': job['company'],
            'location': job['location'],
            'type': job.get('type', 'CDI'),
            'experience': job.get('experience', 'N/A'),
            'salary': job.get('salary', 'N/A'),
            'remote_ok': job.get('remote_ok', False),
            'applicants': applicants,
            'url': job.get('url', ''),
            
            # Scores détaillés
            'global_score': round(float(global_score), 1),
            'skills_score': round(float(skills_score), 1),
            'experience_score': int(experience_match),
            'location_score': int(location_score),
            'competition_score': int(competition_score),
            
            # Détails des compétences matchées
            'skills_details': {
                'high_matches': skills_result['high_matches'],
                'top_skills': [m['skill'] for m in skills_result['matches'][:5]]
            },
            
            # Métadonnées
            'requirements': job.get('requirements', []),
            'nice_to_have': job.get('nice_to_have', []),
            'description_preview': job.get('description', '')[:200] + '...'
        }
    
    def rank_jobs(
        self, 
        cv_skills: List[str], 
        jobs: List[Dict]
    ) -> List[Dict]:
        """
        Classer toutes les offres par score décroissant
        
        Args:
            cv_skills: Liste de compétences du CV
            jobs: Liste d'offres d'emploi
            
        Returns:
            Liste d'offres triées par score
        """
        recommendations = []
        
        for job in jobs:
            score_data = self.calculate_job_match_score(cv_skills, job)
            recommendations.append(score_data)
        
        # Trier par score décroissant
        recommendations = sorted(
            recommendations, 
            key=lambda x: x['global_score'], 
            reverse=True
        )
        
        return recommendations


# Fonction utilitaire pour charger le matcher
def load_matcher(model_name: str = 'all-mpnet-base-v2') -> JobMatcher:
    """
    Charger un JobMatcher pré-configuré
    
    Args:
        model_name: Nom du modèle Sentence-Transformer
        
    Returns:
        Instance de JobMatcher
    """
    return JobMatcher(model_name)