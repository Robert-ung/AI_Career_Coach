"""
🎯 AI Career Coach - Dashboard Streamlit
Application de recommandation d'offres d'emploi basée sur l'analyse de CV
"""

import streamlit as st
import json
from pathlib import Path
import sys
import tempfile
from datetime import datetime

# Ajouter le dossier src au PATH
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configuration de la page
st.set_page_config(
    page_title="AI Career Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .upload-box {
        border: 2px dashed #1f77b4;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 2rem 0;
        background-color: #f0f8ff;
    }
    .job-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .job-card-excellent {
        border-color: #4CAF50;
        background-color: #f1f8f4;
    }
    .job-card-good {
        border-color: #FFC107;
        background-color: #fffbf0;
    }
    .job-card-medium {
        border-color: #FF9800;
        background-color: #fff8f0;
    }
    .score-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .score-excellent {
        background-color: #4CAF50;
        color: white;
    }
    .score-good {
        background-color: #FFC107;
        color: white;
    }
    .score-medium {
        background-color: #FF9800;
        color: white;
    }
    .score-low {
        background-color: #9E9E9E;
        color: white;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """
    Charger les modèles UNIQUEMENT quand nécessaire (lazy loading)
    ⚠️ Cette fonction est appelée SEULEMENT quand un CV est uploadé
    """
    from src.skills_extractor import SkillsExtractor
    from src.job_matcher import JobMatcher
    
    with st.spinner("⏳ Chargement des modèles IA (première fois seulement)..."):
        skills_extractor = SkillsExtractor()
        job_matcher = JobMatcher(model_name='all-mpnet-base-v2')
    
    return skills_extractor, job_matcher


def load_jobs():
    """Charger les offres d'emploi (rapide, pas de modèles)"""
    jobs_path = project_root / "data" / "jobs" / "jobs_dataset.json"
    
    if jobs_path.exists():
        with open(jobs_path, 'r', encoding='utf-8') as f:
            jobs_data = json.load(f)
            return jobs_data.get('jobs', [])
    return []


def process_cv(uploaded_file, all_jobs):
    """
    Pipeline complet de traitement du CV
    
    Returns:
        tuple: (cv_skills, recommendations)
    """
    # Charger les modèles SEULEMENT maintenant (lazy loading)
    skills_extractor, job_matcher = load_models()
    
    # Importer CVParser seulement quand nécessaire
    from src.cv_parser import CVParser
    
    # Étape 1 : Sauvegarder le fichier temporairement
    with st.spinner("📄 Lecture du CV..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
    
    st.success("✅ CV chargé")
    
    # Étape 2 : Parser le PDF
    with st.spinner("🔍 Extraction du texte..."):
        parser = CVParser(method='pdfplumber')
        cv_text = parser.parse(tmp_path)
        
        if not cv_text:
            st.error("❌ Impossible d'extraire le texte du CV")
            return None, None
    
    st.success(f"✅ Texte extrait ({len(cv_text)} caractères)")
    
    # Étape 3 : Extraire les compétences
    with st.spinner("🔧 Extraction des compétences..."):
        results = skills_extractor.extract_from_cv(cv_text)
        cv_skills = results['technical_skills']
        
        if not cv_skills:
            st.warning("⚠️ Aucune compétence technique détectée")
            return None, None
    
    st.success(f"✅ {len(cv_skills)} compétences détectées")
    
    # Étape 4 : Calculer les recommandations
    with st.spinner("🎯 Calcul des recommandations (30-60 secondes)..."):
        recommendations = job_matcher.rank_jobs(cv_skills, all_jobs)
    
    st.success(f"✅ {len(recommendations)} offres analysées")
    
    # Nettoyer le fichier temporaire
    Path(tmp_path).unlink()
    
    return cv_skills, recommendations


def get_score_class(score):
    """Retourner la classe CSS selon le score"""
    if score >= 70:
        return "excellent", "🟢"
    elif score >= 50:
        return "good", "🟡"
    elif score >= 40:
        return "medium", "🟠"
    else:
        return "low", "🔴"


def display_job_card(job, rank):
    """Afficher une carte d'offre d'emploi"""
    score_class, emoji = get_score_class(job['global_score'])
    
    # Classe CSS pour la carte
    card_class = f"job-card job-card-{score_class}" if score_class != "low" else "job-card"
    
    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
    
    # En-tête
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"### {emoji} #{rank} - {job['title']}")
        st.markdown(f"**🏢 {job['company']}** | 📍 {job['location']}")
    
    with col2:
        st.markdown(f'<div class="score-badge score-{score_class}">{job["global_score"]:.1f}%</div>', 
                   unsafe_allow_html=True)
    
    # Détails
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"**💼 Type** : {job['type']}")
        st.markdown(f"**⏱️ Expérience** : {job['experience']}")
    
    with col2:
        st.markdown(f"**💰 Salaire** : {job['salary']}")
        st.markdown(f"**🏠 Remote** : {'Oui ✅' if job['remote_ok'] else 'Non'}")
    
    with col3:
        st.markdown(f"**👥 Candidats** : {job['applicants']}")
        st.markdown(f"**📅 Publié** : {job.get('posted_date', 'N/A')}")
    
    # Scores détaillés
    with st.expander("📊 Voir les scores détaillés"):
        cols = st.columns(4)
        cols[0].metric("Compétences", f"{job['skills_score']:.1f}%")
        cols[1].metric("Expérience", f"{job['experience_score']}%")
        cols[2].metric("Localisation", f"{job['location_score']}%")
        cols[3].metric("Compétition", f"{job['competition_score']}%")
    
    # Compétences requises
    with st.expander("🔧 Compétences requises"):
        st.markdown("**Obligatoires :**")
        for req in job['requirements']:
            st.markdown(f"- {req}")
        
        if job.get('nice_to_have'):
            st.markdown("**Nice to have :**")
            for skill in job['nice_to_have']:
                st.markdown(f"- {skill}")
    
    # Bouton de candidature
    st.link_button("🔗 Voir l'offre", job['url'], use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def main():
    """Application principale"""
    
    # Header
    st.markdown('<div class="main-header">🎯 AI Career Coach</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Trouvez les offres d\'emploi parfaites pour votre profil</div>', 
                unsafe_allow_html=True)
    
    # Charger SEULEMENT les offres (pas les modèles IA)
    all_jobs = load_jobs()
    
    if not all_jobs:
        st.error("❌ Aucune offre d'emploi disponible")
        st.info("⚠️ Veuillez d'abord exécuter le notebook 04_job_generation.ipynb")
        st.stop()
    
    # Initialiser session state
    if 'cv_processed' not in st.session_state:
        st.session_state.cv_processed = False
        st.session_state.cv_skills = []
        st.session_state.recommendations = []
    
    # Zone d'upload
    st.markdown("---")
    st.header("📤 Upload de CV")
    
    uploaded_file = st.file_uploader(
        "Choisissez votre CV (PDF)",
        type=['pdf'],
        help="Uploadez votre CV au format PDF pour obtenir des recommandations personnalisées"
    )
    
    # Bouton d'analyse
    if uploaded_file is not None:
        st.markdown(f"**Fichier uploadé** : {uploaded_file.name}")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🚀 Analyser mon CV", type="primary", use_container_width=True):
                # Traiter le CV (les modèles seront chargés maintenant)
                cv_skills, recommendations = process_cv(uploaded_file, all_jobs)
                
                if cv_skills and recommendations:
                    # Sauvegarder dans session state
                    st.session_state.cv_processed = True
                    st.session_state.cv_skills = cv_skills
                    st.session_state.recommendations = recommendations
                    st.rerun()
        
        with col2:
            if st.button("🔄 Réinitialiser", use_container_width=True):
                st.session_state.cv_processed = False
                st.session_state.cv_skills = []
                st.session_state.recommendations = []
                st.rerun()
    
    # Si pas de CV traité, afficher les instructions
    if not st.session_state.cv_processed:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("### 📄 Comment ça marche ?")
        st.markdown("""
        1. **Uploadez votre CV** au format PDF
        2. **Cliquez sur "Analyser mon CV"**
        3. **Obtenez des recommandations personnalisées** basées sur vos compétences
        
        Notre système utilise l'IA pour :
        - ✅ Extraire automatiquement vos compétences
        - ✅ Comparer votre profil avec 25+ offres d'emploi
        - ✅ Calculer un score de matching sémantique
        - ✅ Recommander les meilleures opportunités
        
        ⏱️ **Temps de traitement estimé** : 30-60 secondes
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.stop()
    
    # Si CV traité, afficher les résultats
    cv_skills = st.session_state.cv_skills
    recommendations = st.session_state.recommendations
    
    # Sidebar - Filtres
    st.sidebar.header("🔍 Filtres")
    
    # Filtre par score minimum
    min_score = st.sidebar.slider(
        "Score minimum (%)",
        min_value=0,
        max_value=100,
        value=40,
        step=5
    )
    
    # Filtre par catégorie
    categories = sorted(set(job.get('category', 'unknown').replace('_', ' ').title() 
                           for job in all_jobs))
    selected_categories = st.sidebar.multiselect(
        "Catégories",
        options=categories,
        default=categories
    )
    
    # Filtre Remote
    remote_filter = st.sidebar.radio(
        "Type de travail",
        options=["Tous", "Remote uniquement", "On-site uniquement"],
        index=0
    )
    
    # Filtre par expérience
    exp_levels = sorted(set(job['experience'] for job in all_jobs))
    selected_exp = st.sidebar.multiselect(
        "Niveau d'expérience",
        options=exp_levels,
        default=exp_levels
    )
    
    # Appliquer les filtres
    filtered_recs = recommendations.copy()
    
    # Filtre score
    filtered_recs = [job for job in filtered_recs if job['global_score'] >= min_score]
    
    # Filtre catégorie
    if selected_categories:
        selected_categories_lower = [cat.lower().replace(' ', '_') for cat in selected_categories]
        filtered_recs = [
            job for job in filtered_recs 
            if any(
                all_job['job_id'] == job['job_id'] and 
                all_job.get('category', '') in selected_categories_lower
                for all_job in all_jobs
            )
        ]
    
    # Filtre remote
    if remote_filter == "Remote uniquement":
        filtered_recs = [job for job in filtered_recs if job['remote_ok']]
    elif remote_filter == "On-site uniquement":
        filtered_recs = [job for job in filtered_recs if not job['remote_ok']]
    
    # Filtre expérience
    if selected_exp:
        filtered_recs = [job for job in filtered_recs if job['experience'] in selected_exp]
    
    # Statistiques globales
    st.markdown("---")
    st.header("📊 Vue d'ensemble")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Compétences CV", len(cv_skills))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Offres analysées", len(recommendations))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Offres filtrées", len(filtered_recs))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if filtered_recs:
            st.metric("Meilleur score", f"{filtered_recs[0]['global_score']:.1f}%")
        else:
            st.metric("Meilleur score", "N/A")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Distribution des scores
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Distribution des matches")
        excellent = len([j for j in filtered_recs if j['global_score'] >= 70])
        good = len([j for j in filtered_recs if 50 <= j['global_score'] < 70])
        medium = len([j for j in filtered_recs if 40 <= j['global_score'] < 50])
        low = len([j for j in filtered_recs if j['global_score'] < 40])
        
        st.markdown(f"🟢 **Excellent match (≥70%)** : {excellent} offres")
        st.markdown(f"🟡 **Bon match (50-70%)** : {good} offres")
        st.markdown(f"🟠 **Match moyen (40-50%)** : {medium} offres")
        st.markdown(f"🔴 **Match faible (<40%)** : {low} offres")
    
    with col2:
        st.subheader("🔧 Vos compétences")
        for i, skill in enumerate(cv_skills[:10], 1):
            st.markdown(f"{i}. {skill}")
        
        if len(cv_skills) > 10:
            with st.expander(f"Voir les {len(cv_skills) - 10} autres compétences"):
                for i, skill in enumerate(cv_skills[10:], 11):
                    st.markdown(f"{i}. {skill}")
    
    # Liste des offres
    st.markdown("---")
    st.header(f"🏆 Top {min(10, len(filtered_recs))} Offres Recommandées")
    
    if not filtered_recs:
        st.warning("Aucune offre ne correspond aux critères sélectionnés")
        st.info("💡 Essayez de réduire le score minimum ou d'élargir les filtres")
    else:
        # Nombre d'offres à afficher
        num_to_show = st.selectbox(
            "Nombre d'offres à afficher",
            options=[5, 10, 15, 20, len(filtered_recs)],
            index=1
        )
        
        # Afficher les offres
        for i, job in enumerate(filtered_recs[:num_to_show], 1):
            display_job_card(job, i)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "🎯 AI Career Coach | Powered by Sentence-Transformers & Streamlit"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()