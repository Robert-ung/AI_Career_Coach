# 🎯 AI Career Coach - Système Intelligent de Matching CV ↔ Offres d'Emploi

## 📖 Description du Projet

**AI Career Coach** est un système intelligent d'aide à l'emploi destiné aux **profils juniors en Data Science et ML Engineering**. Le projet combine **NLP**, **embeddings sémantiques**, **machine learning** et **recherche vectorielle** pour proposer des recommandations d'emploi personnalisées basées sur l'analyse automatique de CV.

###  Objectifs Principaux

1. **Extraction automatique** des compétences techniques et soft skills depuis un CV PDF
2. **Matching sémantique** entre profil candidat et offres d'emploi
3. **Scoring intelligent** basé sur la couverture et la qualité des compétences
4. **Recommandations personnalisées** avec explication des forces et faiblesses
5. **Simulation d'entretiens** avec génération de questions contextuelles
6. **MLOps pipeline** avec tracking des expériences et déploiement de modèles

## 📁 Structure du projet

```
AI_Career_Coach/
│
├── 📁 data/                               # Données et artifacts
│   ├── 📁 jobs/                           # Offres d'emploi et embeddings
│   │   ├── jobs_dataset.json              # 25 offres d'emploi (Data Science/ML)
│   │   ├── jobs_faiss.index                # Index FAISS pour recherche vectorielle
│   │   └── jobs_embeddings.pkl             # Embeddings pré-calculés (768-dim)
│   │
│   ├── 📁 resume_fit_job/                   # Dataset CV-Job
│   │   ├── 📁 processed/                    # Données nettoyées
│   │   │   └── v2_dataset_resume_job_fit_processed.xlsx  # Dataset nettoyé (4,524 samples)
│   │   └── 📁 raw/                          # Données brutes
│   │       └── dataset_resume_job_fit.xlsx  # Dataset brut (6,241 samples)
│   │
│   ├── skills_reference.json                # Compétences techniques + soft skills
│   └── RESUME_*.pdf                         # CVs de test
│
├── 📁 mlops/                                # Pipeline MLOps
│   ├── train_and_log.py                     # Entraînement + tracking MLflow
│   ├── register_model.py                    # Enregistrement Model Registry
│   ├── serve_model.py                       # Test de prédiction
│   ├── 📁 mlflow_tracking/                   # Généré automatiquement (ignoré Git)
│   └── 📁 mlflow_models/                     # Généré automatiquement (ignoré Git)
│
├── 📁 models/                               # Modèles entraînés (metadata uniquement)
│   └── classifier_clean_metadata.json       # Métadonnées du modèle XGBoost
│
├── 📁 notebooks/                            # Notebooks de développement
│   ├── 01_cv_parser.ipynb                   # Parsing de CV PDF
│   ├── 02_skills_extraction_simple.ipynb    # Extraction de compétences CV
│   ├── 03_extraction_skills_job_offers.ipynb # Extraction de compétences jobs
│   ├── 03_semantic_matching.ipynb            # Tests de matching sémantique
│   ├── 04_job_generation.ipynb              # Génération du dataset d'offres
│   ├── 05_job_recommendation.ipynb          # Système de recommandation
│   ├── 06_faiss_indexing.ipynb              # Base vectorielle
│   ├── 07_interview_simulation.ipynb        # Simulation d'entretiens
│   ├── 08_exploration_dataset_RAW.ipynb     # Exploration dataset brute
│   └── 09_ml_model_training.ipynb           # Entraînement modèle ML (XGBoost, 70% accuracy)
│
├── 📁 src/                                   # Code source principal
│   ├── api.py                               # API FastAPI (endpoints REST)
│   ├── cv_parser.py                         # Parser CV (PyPDF2 + pdfplumber)
│   ├── skills_extractor.py                  # Extraction compétences (spaCy + regex)
│   ├── job_matcher.py                       # Matching sémantique (SentenceTransformer)
│   ├── vector_store.py                      # Recherche vectorielle (FAISS)
│   ├── interview_simulator.py               # Génération questions d'entretien
│   └── compute_features_from_huggingface.py # Calcul features ML
│
├── 📁 tests/                                 # Tests unitaires (TODO)
│   └── ...
│
├── app.py                                    # Dashboard Streamlit (frontend)
├── requirements.txt                          # Dépendances Python
├── .gitignore                                
└── README.md                                
```

## 🚀 Quick Start

### Lancer l'API

```bash

# 1. Cloner le repo
git clone https://github.com/Robert-ung/AI_Career_Coach.git
cd AI_Career_Coach

# 2. Créer l'environnement
python -m venv env
source env/bin/activate  # (ou env\Scripts\activate sur Windows)

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Exécuter les scripts (qui généreront les fichiers localement)
python mlops/train_and_log.py
python mlops/register_model.py

# 5. Lancer MLflow UI
mlflow ui --backend-store-uri file:./mlops/mlflow_tracking

Accéder à MLflow UI : http://127.0.0.1:5000

# 6. Lancer l'API
uvicorn src.api:app --reload --port 8000

Documentation interactive : http://127.0.0.1:8000/docs

# Tester l'API

# Health Check
curl http://127.0.0.1:8000/health

# Stats
curl http://127.0.0.1:8000/api/v1/stats

# 7. Lancer le dashboard
streamlit run app.py

Interface utilisateur : http://localhost:8501

## 🎯 **Modèle Entraîné**

- **Type** : XGBoost Classifier
- **Classes** : 3 (No Fit, Partial Fit, Perfect Fit)
- **Features** : 15 (coverage, quality, similarities, etc.)
- **Performance** : ~70% accuracy (Test Set)
- **Dataset** : 4,524 samples (nettoyé)

# 🎯 ROADMAP PFE - Système d'Aide à l'Emploi pour Juniors

## 📅 SEMAINE 1-2 : CORE FONCTIONNEL
- [x] Parser CV (01_cv_parser.ipynb)
- [x] Extraction compétences (02_skills_extraction_simple.ipynb)
- [x] Matching sémantique (03_semantic_matching.ipynb)
- [X] Scraping offres (04_job_scraping.ipynb) 
- [X] Matching CV ↔ Offres (05_job_recommendation.ipynb)
- [X] Dashboard Streamlit v1 (app.py)

**Livrable Semaine 2** : Système fonctionnel de bout en bout

## 📅 SEMAINE 3-4 : ENRICHISSEMENT
- [X] API FastAPI (src/api.py) 
- [x] Dashboard Streamlit avec API 
- [X] Base vectorielle FAISS (src/vector_store.py)
- [X] Simulation entretien LLM (06_interview_simulation.ipynb)
- [X] Clustering profils KMeans (07_profile_clustering.ipynb)

**Livrable Semaine 4** : API + Features ML avancées

## 📅 SEMAINE 5-6 : INDUSTRIALISATION
- [ ] Tests unitaires (tests/) ← MAINTENANT
- [ ] Dashboard Streamlit v2 (graphiques, stats)
- [ ] Scraping offres réelles via API (optionnel)
- [ ] Monitoring performances (logs, métriques)

**Livrable Semaine 6** : Code robuste et testé

## 📅 SEMAINE 7-8 : FINALISATION
- [ ] Documentation complète (README, docstrings)
- [ ] Rapport PFE
- [ ] Préparation soutenance (slides)
- [ ] Déploiement cloud (optionnel)

**Livrable Semaine 8** : PFE complet

Pipeline :

┌─────────────────────────────────────────────────────────────┐
│  1. UPLOAD CV (Frontend Streamlit)                          │
│     • Utilisateur upload CV PDF via interface               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  2. PARSING (cv_parser.py)                                  │
│     • PyPDF2 + pdfplumber                                   │
│     • Extraction texte brut (~2000 caractères)              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  3. EXTRACTION SKILLS (skills_extractor.py)                 │
│     • spaCy                                                 │
│     • Pattern matching sur skills                           │
│     • Résultat : ["python", "pandas", "numpy", ...]         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  4. PRÉ-FILTRAGE FAISS (vector_store.py) [OPTIONNEL]        │
│     • Embedding CV avec SentenceTransformer                 │
│     • Recherche Top-50 dans index FAISS                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  5. SCORING DÉTAILLÉ (job_matcher.py)                       │
│     • Calcul similarité CV ↔ Job (cosinus)                  │
│     • Score = (Coverage × 0.5) + (Quality × 0.5)            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  6. TRI & FILTRAGE (api.py)                                 │
│     • Tri par score décroissant                             │
│     • Filtrage score minimum (défaut: 40%)                  │
│     • Limitation Top-N (défaut: 10)                         │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  7. AFFICHAGE (app.py)                                      │
│     • Cards avec score + compétences matchées/manquantes    │
│     • Filtres interactifs (remote, expérience)              │
│     • Graphiques de répartition                             │
└─────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                         🎯 PFE - JOB MATCHING SYSTEM                        │
│                         Système de Recommandation d'Emplois                  │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           📱 FRONTEND (Streamlit)                            │
│                               app.py                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│  🖥️  Interface Utilisateur                                                  │
│  • Upload CV (PDF)                                                          │
│  • Affichage des recommandations                                            │
│  • Filtres (score, remote, expérience)                                      │
│  • Visualisation des compétences matchées/manquantes                        │
│  • Statistiques dashboard                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓ HTTP POST
                                    ↓ /api/v1/recommend-jobs
┌─────────────────────────────────────────────────────────────────────────────┐
│                          🚀 BACKEND API (FastAPI)                            │
│                              src/api.py                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  📡 Endpoints REST                                                           │
│  • POST /api/v1/recommend-jobs      → Recommandations                      │
│  • POST /api/v1/extract-skills      → Extraction skills CV                 │
│  • GET  /api/v1/jobs                → Liste offres (avec filtres)          │
│  • GET  /api/v1/stats               → Statistiques système                 │
│  • GET  /health                     → Health check                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
┌─────────────────────────────────┐  ┌──────────────────────────────────┐
│    🧠 MODULES CORE (src/)       │  │   📚 DATA LAYER                  │
├─────────────────────────────────┤  ├──────────────────────────────────┤
│  1️⃣  cv_parser.py               │  │  • data/jobs.json (25 offres)   │
│     • PyPDF2                    │  │  • data/skills_reference.json    │
│     • pdfplumber                │  │  • data/faiss_index.bin          │
│     • Extraction texte brut     │  │  • data/job_embeddings.pkl       │
│                                 │  │  • data/RESUME_*.pdf             │
│  2️⃣  skills_extractor.py        │  └──────────────────────────────────┘
│     • spaCy (fr_core_news_lg)   │
│     • Règles linguistiques      │
│     • Normalisation skills      │
│                                 │
│  3️⃣  job_matcher.py             │
│     • SentenceTransformer       │
│     • all-mpnet-base-v2         │
│     • Similarité sémantique     │
│     • Scoring Approche 4        │
│                                 │
│  4️⃣  vector_store.py            │
│     • FAISS indexing            │
│     • Pré-filtrage rapide       │
│     • Top-k candidats           │
│                                 │
│  5️⃣  interview_simulator.py     │
│     • Génération questions      │
│     • Évaluation réponses       │
└─────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────┐
│                    📤 ÉTAPE 1 : UPLOAD CV (Frontend)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    Utilisateur clique "Analyser mon CV"
                    uploaded_file.pdf (via st.file_uploader)
                                    │
                                    ↓
                         HTTP POST multipart/form-data
                         → http://localhost:8000/api/v1/recommend-jobs
                         params: {top_n: 25, min_score: 40}
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                  📥 ÉTAPE 2 : RÉCEPTION API (api.py)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    @app.post("/api/v1/recommend-jobs")
                    └─ Validation fichier PDF
                    └─ Sauvegarde temporaire /tmp/cv_temp.pdf
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              🔍 ÉTAPE 3 : PARSING CV (cv_parser.py)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    CVParser.parse_cv(cv_path) 
                                    │
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
            ┌──────────────┐              ┌──────────────┐
            │   PyPDF2     │              │  pdfplumber  │
            │   Fallback   │              │  Méthode 1   │
            └──────────────┘              └──────────────┘
                    │                                │
                    └────────────┬───────────────────┘
                                 ↓
                    📄 CV Texte Brut (string)
                    • "Robert UNG, Data Scientist..."
                    • ~2000 caractères
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│          🧬 ÉTAPE 4 : EXTRACTION SKILLS (skills_extractor.py)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    SkillsExtractor.extract_skills(cv_text)
                                    │
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
        ┌─────────────────────┐         ┌──────────────────────┐
        │  spaCy Processing   │         │  Pattern Matching    │
        │  • Tokenization     │         │  • skills_reference  │
        │  • POS tagging      │         │  • Regex patterns    │
        │  • Named entities   │         │  • 171 tech skills   │
        └─────────────────────┘         └──────────────────────┘
                    │                                │
                    └────────────┬───────────────────┘
                                 ↓
                    📋 Liste de Compétences Normalisées
                    cv_skills = [
                        "python", "pandas", "numpy", 
                        "scikit-learn", "tensorflow",
                        "docker", "fastapi", "git", ...
                    ] (20 skills typiquement)
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│         🎯 ÉTAPE 5 : PRÉ-FILTRAGE FAISS (vector_store.py) [OPTIONNEL]      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            IF use_faiss=True OR len(jobs) > 50:
                                    │
                    VectorStore.search(cv_skills, cv_text)
                                    │
                    ┌───────────────┴────────────────┐
                    ↓                                ↓
        ┌─────────────────────┐         ┌──────────────────────┐
        │  Embeddings CV      │         │  FAISS Index         │
        │  • SentenceTransf.  │   ←→    │  • 25 offres indexées│
        │  • all-mpnet-base-v2│         │  • Recherche rapide  │
        │  • 768 dimensions   │         │  • Cosine similarity │
        └─────────────────────┘         └──────────────────────┘
                    │
                    ↓
            🔝 Top 50 Candidats (jobs pré-filtrés)
            Temps : ~0.5s au lieu de 2.5s
                                    │
                    ELSE:
                    └─→ Tous les 25 jobs du dataset
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│         🧮 ÉTAPE 6 : SCORING DÉTAILLÉ (job_matcher.py)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            FOR EACH job IN candidate_jobs:
                                    │
                JobMatcher.calculate_job_match_score(cv_skills, job)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ↓                           ↓                           ↓
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ Extract Job      │    │ Embeddings       │    │ Compute          │
│ Skills           │    │ Calculation      │    │ Similarity       │
│                  │    │                  │    │                  │
│ • requirements[] │ →  │ SentenceTransf.  │ →  │ Cosine Sim.      │
│ • nice_to_have[] │    │ encode()         │    │ (CV ↔ Job)       │
│                  │    │ 768-dim vectors  │    │ 0.0 → 1.0        │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    ↓
                        📊 Approche 4 Scoring
                        ┌─────────────────────────┐
                        │ Coverage (50%)          │
                        │ = Skills couverts /     │
                        │   Skills requis         │
                        │                         │
                        │ Quality (50%)           │
                        │ = Moyenne similarités   │
                        │   des skills matchés    │
                        └─────────────────────────┘
                                    ↓
                        🎯 Score Final (0-100%)
                        score = (coverage × 0.5) + (quality × 0.5)
                                    │
                                    ↓
                        📋 Détails Complets
                        {
                            "score": 78.3,
                            "matching_skills": [
                                "python", "pandas", "numpy", ...
                            ],
                            "missing_skills": [
                                "spark", "airflow", "kafka", ...
                            ],
                            "skills_details": {
                                "coverage": 76.5,
                                "quality": 80.1,
                                "covered_count": 13,
                                "total_required": 17
                            }
                        }
                                    │
                                    ↓
            END FOR
            Temps : ~0.1s par job (2.5s pour 25 jobs)
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              📊 ÉTAPE 7 : TRI ET FILTRAGE (api.py)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            1. Tri par score décroissant
               detailed_results.sort(key='score', reverse=True)
                                    │
            2. Filtrage par score minimum
               jobs = [j for j in jobs if j['score'] >= min_score]
                                    │
            3. Limitation top_n
               jobs = jobs[:top_n]  # Max 25
                                    │
                                    ↓
            🎯 Recommandations Finales (JSON)
            {
                "recommendations": [
                    {
                        "job_id": "job_001",
                        "title": "ML Engineer",
                        "score": 85.5,
                        "matching_skills": [...],
                        "missing_skills": [...]
                    },
                    ...
                ],
                "total_jobs_analyzed": 25,
                "cv_skills_count": 20
            }
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              📤 ÉTAPE 8 : RÉPONSE HTTP → FRONTEND                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            HTTP 200 OK
            Content-Type: application/json
            Response Time: ~7-10 secondes
                                    │
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│              🎨 ÉTAPE 9 : AFFICHAGE STREAMLIT (app.py)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
            st.session_state.recommendations = result
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        ↓                           ↓                           ↓
┌──────────────┐        ┌──────────────────┐        ┌──────────────────┐
│  Sidebar     │        │  Main Content    │        │  Job Cards       │
│  Filtres     │        │  Statistiques    │        │  Détails         │
│              │        │                  │        │                  │
│ • Score min  │   →    │ • Total offres   │   →    │ • Score badge    │
│ • Remote     │        │ • Skills CV      │        │ • Compétences ✅ │
│ • Expérience │        │ • Graphiques     │        │ • Compétences ❌ │
└──────────────┘        └──────────────────┘        └──────────────────┘
        │                           │                           │
        └───────────────────────────┴───────────────────────────┘
                                    ↓
                        🎉 RÉSULTAT FINAL
                        Interface interactive avec :
                        • 8-25 recommandations affichées
                        • Filtrage en temps réel
                        • Détails par offre
                        • Compétences matchées/manquantes