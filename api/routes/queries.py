from fastapi import APIRouter
from pydantic import BaseModel
from querry_engine.query_handler import handle_query

router = APIRouter()

# Modèle pour valider les données reçues dans le corps de la requête
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@router.post("/ask")
def ask_question(request: QueryRequest):
    """
    Endpoint pour interroger le système.
    Args:
        request (QueryRequest): Question posée par l'utilisateur et le nombre de résultats à retourner.
    Returns:
        dict: Réponse générée par le LLM.
    """
    try:
        response = handle_query(request.question, request.top_k)
        return {"question": request.question, "response": response}
    except Exception as e:
        return {"error": str(e)}
