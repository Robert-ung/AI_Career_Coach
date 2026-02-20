"""
Module d'extraction de compétences depuis un CV
"""

import re
import json
import spacy
from pathlib import Path
from typing import List, Dict, Tuple


class SkillsExtractor:
    """
    Extracteur de compétences techniques et soft skills
    """
    
    def __init__(self, skills_db_path: str = None):
        """
        Initialiser l'extracteur
        
        Args:
            skills_db_path: Chemin vers skills_reference.json
        """
        # Charger spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Modèle spaCy chargé")
        except OSError:
            raise RuntimeError(
                "Modèle spaCy non trouvé. Exécutez:\n"
                "python -m spacy download en_core_web_sm"
            )
        
        # Charger la base de compétences
        if skills_db_path is None:
            skills_db_path = Path(__file__).parent.parent / "data" / "skills_reference.json"
        
        with open(skills_db_path, 'r', encoding='utf-8') as f:
            self.skills_database = json.load(f)
        
        print(f"✅ Base de compétences chargée")
        print(f"   • Techniques : {len(self.skills_database['technical_skills'])}")
        print(f"   • Soft skills : {len(self.skills_database['soft_skills'])}")
    
    def extract_skills_from_text(
        self, 
        text: str, 
        skills_list: List[str]
    ) -> List[str]:
        """
        Extraire compétences depuis texte brut
        
        Args:
            text: Texte du CV
            skills_list: Liste de compétences de référence
            
        Returns:
            Liste de compétences trouvées
        """
        text_lower = text.lower()
        found_skills = set()
        
        for skill in skills_list:
            skill_lower = skill.lower()
            
            # Pattern flexible pour gérer C++, Node.js, .NET
            if re.search(r'[^a-z0-9\s]', skill_lower):
                escaped = re.escape(skill_lower)
                pattern = r'(?:^|\s|[(\[{])' + escaped + r'(?:\s|$|[.,;:)\]}])'
            else:
                pattern = r'\b' + re.escape(skill_lower) + r'\b'
            
            if re.search(pattern, text_lower):
                found_skills.add(skill)
        
        return sorted(found_skills)
    
    def extract_from_cv(self, cv_text: str) -> Dict:
        """
        Extraire toutes les compétences d'un CV
        
        Args:
            cv_text: Texte complet du CV
            
        Returns:
            Dict avec compétences techniques et soft skills
        """
        # Traiter avec spaCy (optionnel, pour analyse)
        doc = self.nlp(cv_text)
        
        # Extraire compétences techniques
        technical_skills = self.extract_skills_from_text(
            cv_text, 
            self.skills_database['technical_skills']
        )
        
        # Extraire soft skills
        soft_skills = self.extract_skills_from_text(
            cv_text,
            self.skills_database['soft_skills']
        )
        
        return {
            "technical_skills": technical_skills,
            "soft_skills": soft_skills,
            "total_skills": len(technical_skills) + len(soft_skills),
            "spacy_entities": [
                {"text": ent.text, "label": ent.label_} 
                for ent in doc.ents
            ]
        }
    
    def extract_and_save(
        self, 
        cv_text: str, 
        output_path: str,
        cv_filename: str = "CV.pdf"
    ) -> Dict:
        """
        Extraire et sauvegarder les résultats
        
        Args:
            cv_text: Texte du CV
            output_path: Chemin de sauvegarde
            cv_filename: Nom du fichier CV
            
        Returns:
            Résultats de l'extraction
        """
        from datetime import datetime
        
        # Extraire
        results = self.extract_from_cv(cv_text)
        
        # Ajouter métadonnées
        results["cv_file"] = cv_filename
        results["extraction_date"] = datetime.now().strftime("%Y-%m-%d")
        results["method"] = "spaCy + keyword matching"
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats sauvegardés : {output_path}")
        
        return results


# Fonction utilitaire
def extract_skills_from_cv_file(
    cv_text_path: str,
    output_path: str = None,
    skills_db_path: str = None
) -> Dict:
    """
    Extraire compétences depuis un fichier texte de CV
    
    Args:
        cv_text_path: Chemin vers le fichier texte du CV
        output_path: Chemin de sauvegarde (optionnel)
        skills_db_path: Chemin vers skills_reference.json
        
    Returns:
        Résultats de l'extraction
    """
    # Charger le texte
    with open(cv_text_path, 'r', encoding='utf-8') as f:
        cv_text = f.read()
    
    # Créer l'extracteur
    extractor = SkillsExtractor(skills_db_path)
    
    # Extraire et sauvegarder
    if output_path:
        return extractor.extract_and_save(cv_text, output_path)
    else:
        return extractor.extract_from_cv(cv_text)