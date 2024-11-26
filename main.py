from document_processing.document_processor import process_documents
from embedding.embedding_manager import generate_embeddings
from faiss_index.faiss_manager import save_faiss_index
from utils.config_loader import load_config
from utils.logger import get_logger

logger = get_logger(__name__)

def main():
    # Charger la configuration principale
    app_config = load_config("config/app_config.yaml")
    embedding_config = load_config("config/embedding_config.yaml")

    input_dir = app_config["documents"]["input_dir"]

    # Étape 1 : Charger et traiter les documents
    logger.info(f"Processing documents from directory: {input_dir}")
    documents = process_documents(input_dir)

    if not documents:
        logger.error("No documents found. Exiting...")
        return

    # Étape 2 : Générer les embeddings et métadonnées
    logger.info("Generating embeddings...")
    embeddings, metadata = generate_embeddings(documents, embedding_config)  # Passer le dictionnaire

    # Étape 3 : Sauvegarder l'index FAISS et les métadonnées
    logger.info("Saving FAISS index and metadata...")
    save_faiss_index(embeddings, metadata, app_config["faiss"]["index_path"], app_config["faiss"]["metadata_path"])
    logger.info("Index and metadata successfully created and saved!")

if __name__ == "__main__":
    main()
