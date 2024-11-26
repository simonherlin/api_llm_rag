from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from faiss_index.faiss_manager import load_faiss_index, load_metadata, search_in_faiss

app = Flask(__name__)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
index = load_faiss_index()
metadata = load_metadata()

@app.route("/query", methods=["POST"])
def query():
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", 3)

    results = search_in_faiss(query, embedding_model, index, metadata, top_k)
    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000, debug=True)
