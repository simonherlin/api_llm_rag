import faiss
import json
from utils.logger import get_logger

logger = get_logger(__name__)

def save_faiss_index(embeddings, metadata, index_path, metadata_path):
    """
    Sauvegarde l'index FAISS et les métadonnées.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    logger.info("Saving FAISS index...")
    faiss.write_index(index, index_path)

    logger.info("Saving metadata...")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)

def load_faiss_index(index_path, metadata_path):
    """
    Charge l'index FAISS et les métadonnées.
    """
    logger.info("Loading FAISS index and metadata...")
    index = faiss.read_index(index_path)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return index, metadata
