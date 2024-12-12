import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)

def load_embedding_model(config_path):
    """
    Charge le modèle d'embedding à partir de la configuration.
    Args:
        config_path (str): Chemin vers le fichier de configuration.
    Returns:
        Model: Instance du modèle chargé.
    """
    # Exemple : Charger un modèle avec SentenceTransformers
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config_path)
    return model


def generate_embeddings(documents, config):
    """
    Génère des embeddings pour chaque paragraphe et conserve les métadonnées.
    Args:
        documents (dict): Dictionnaire des documents découpés en paragraphes.
        config (dict): Configuration du modèle d'embedding.
    Returns:
        tuple: (numpy array d'embeddings, métadonnées)
    """
    model_name = config["model_name"]
    model_path = config.get("model_path")

    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = SentenceTransformer(model_path if model_path else model_name)

    embeddings = []
    metadata = {}

    index = 0
    for doc_name, paragraphs in documents.items():
        for para_id, paragraph in enumerate(paragraphs):
            embedding = embedding_model.encode(paragraph).astype('float32')
            embeddings.append(embedding)
            metadata[index] = {
                "doc_name": doc_name,
                "paragraph_id": para_id,
                "content": paragraph
            }
            index += 1

    logger.info(f"Generated {len(embeddings)} embeddings.")
    return np.array(embeddings), metadata
