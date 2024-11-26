from embeddings.embedding_manager import load_embedding_model
from faiss_index.faiss_manager import load_faiss_index
from llm.llm_manager import load_llm

def handle_query(query, top_k=3, config_path="config/app_config.yaml"):
    # Charger les configurations et les modèles
    embedding_model = load_embedding_model(config_path)
    index, metadata = load_faiss_index(
        index_path="data/faiss_index.bin",
        metadata_path="data/metadata.json"
    )
    llm = load_llm(config_path)

    # Encoder la requête
    query_embedding = embedding_model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, top_k)

    # Récupérer le contexte depuis les métadonnées
    context = "\n".join([metadata[str(idx)]["content"] for idx in indices[0]])

    # Générer une réponse avec le LLM
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = llm(prompt)
    return response
