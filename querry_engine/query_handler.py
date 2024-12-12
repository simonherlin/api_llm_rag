from embedding.embedding_manager import load_embedding_model
from faiss_index.faiss_manager import load_faiss_index
from llm.llm_manager import load_llm

def handle_query(query, top_k=3):
    """
    Traite la requête en interrogeant l'index FAISS et le LLM.
    Args:
        query (str): Requête de l'utilisateur.
        top_k (int): Nombre de résultats pertinents.
    Returns:
        str: Réponse générée par le LLM.
    """
    # Charger le modèle d'embedding, l'index FAISS et le LLM
    embedding_model = load_embedding_model("config/embedding_config.yaml")
    index, metadata = load_faiss_index("data/faiss_index.bin", "data/metadata.json")
    llm = load_llm("config/llm_config.yaml")

    # Encoder la requête
    query_embedding = embedding_model.encode([query]).astype("float32")

    # Recherche dans l'index FAISS
    distances, indices = index.search(query_embedding, top_k)

    # Récupérer le contexte des résultats
    context = "\n".join([metadata[str(idx)]["content"] for idx in indices[0]])

    # Générer une réponse avec le LLM
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    return llm(prompt)
