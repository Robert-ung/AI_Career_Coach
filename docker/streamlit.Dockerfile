# ============================================================================
# BUILD STAGE (compiler les dépendances)
# ============================================================================
FROM python:3.10-slim as builder

WORKDIR /app

# Copier les requirements
COPY requirements/base.txt requirements/frontend.txt ./

# Installer les dépendances dans un virtualenv (= plus léger)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r base.txt -r frontend.txt

# ============================================================================
# RUNTIME STAGE (image finale légère)
# ============================================================================
FROM python:3.10-slim

WORKDIR /app

# Copier le virtualenv du build stage
COPY --from=builder /opt/venv /opt/venv

# Définir PATH pour utiliser le venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_CLIENT_LOGGER_LEVEL=error

# Créer un utilisateur non-root
RUN useradd -m -u 1000 streamlituser && \
    chown -R streamlituser:streamlituser /app

# Copier TOUS les fichiers nécessaires
COPY --chown=streamlituser:streamlituser app.py .
COPY --chown=streamlituser:streamlituser pages/ ./pages/
COPY --chown=streamlituser:streamlituser src/ ./src/
COPY --chown=streamlituser:streamlituser data/ ./data/

# Créer dossier cache Streamlit
RUN mkdir -p /home/streamlituser/.streamlit && \
    chown -R streamlituser:streamlituser /home/streamlituser

USER streamlituser

EXPOSE 8501

# Healthcheck (pour Docker savoir si le conteneur marche)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Démarrage
CMD ["streamlit", "run", "app.py", "--logger.level=error"]