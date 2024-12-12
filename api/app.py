from fastapi import FastAPI
from api.routes import queries
from llm.llm_manager import load_llm
from faiss_index.faiss_manager import load_faiss_index
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

# Variables globales pour les ressources
app = FastAPI(
    title="RAG LLM API",
    description="API pour interroger un système RAG basé sur un LLM.",
    version="1.0.0"
)
resources = {}  # Dictionnaire global pour les ressources partagées

@app.on_event("startup")
def startup_event():
    """
    Charger les ressources au démarrage de l'API.
    """
    logger.info("Loading application resources...")

    # Charger la configuration
    app_config = load_config("config/app_config.yaml")
    resources["config"] = app_config

    # Charger l'index FAISS et les métadonnées
    resources["faiss_index"], resources["metadata"] = load_faiss_index(
        index_path="data/faiss_index.bin",
        metadata_path="data/metadata.json"
    )
    logger.info("FAISS index and metadata loaded.")

    # Charger le modèle LLM
    resources["llm"] = load_llm(config_path="config/llm_config.yaml")
    logger.info("LLM model loaded.")

    logger.info("All resources loaded successfully.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG LLM API"}

# Inclure les routes
app.include_router(queries.router, prefix="/queries", tags=["Queries"])
